"""CV-free learned anchors with sparse, mode-conditioned semantic-map queries."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT.parent / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"


def _load_source():
    spec = importlib.util.spec_from_file_location("exp147_hierarchical_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source()


class SemanticAnchorQueryDecoder(SOURCE.HierarchicalLearnedAnchorDecoder):
    """Query only the three anchors of each focal mode, then correct them once."""

    def __init__(self, token_dim: int, pred_len: int, asset_path: Path) -> None:
        super().__init__(
            token_dim=token_dim, pred_len=pred_len, num_modes=20,
            anchor_steps=(4, 8, pred_len), use_refiner=False, top_m=20,
            hidden_dim=64,
        )
        bank = np.load(asset_path, allow_pickle=False)
        maps = torch.from_numpy(np.asarray(bank["maps"], dtype=np.uint8))
        image_wh = torch.from_numpy(np.asarray(bank["image_wh"], dtype=np.float32))
        self.register_buffer("scene_maps", maps, persistent=False)
        self.register_buffer("image_wh", image_wh, persistent=False)
        self.pixel_scale = float(np.asarray(bank["pixel_scale"]).reshape(-1)[0])
        self.scene_channels = int(maps.shape[1])
        correction_input = 64 + self.anchor_count * 2 + self.anchor_count * self.scene_channels
        self.scene_correction = nn.Sequential(
            nn.LayerNorm(correction_input), nn.Linear(correction_input, 128),
            nn.GELU(), nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, self.anchor_count * 2),
        )
        nn.init.zeros_(self.scene_correction[-1].weight)
        nn.init.zeros_(self.scene_correction[-1].bias)
        self.register_buffer(
            "scene_max_offsets", torch.tensor([0.35, 0.55, 0.90], dtype=torch.float32)
        )
        self.heading_normalized = os.environ.get("EXP147_HEADING_NORMALIZED", "1") == "1"
        self._scene_ids: Tensor | None = None
        self._absolute_past: Tensor | None = None
        self.last_corrected_anchors: Tensor | None = None

    def set_scene_batch(self, scene_ids: Tensor, absolute_past: Tensor) -> None:
        if (scene_ids < 0).any():
            raise RuntimeError("semantic scene decoder requires per-sample SDD scene IDs")
        self._scene_ids = scene_ids
        self._absolute_past = absolute_past

    def local_to_global(self, relative: Tensor) -> Tensor:
        assert self._absolute_past is not None
        origin = self._absolute_past[:, 0, -1]
        if not self.heading_normalized:
            return origin[:, None, None] + relative
        lag = min(3, self._absolute_past.shape[-2] - 1)
        displacement = self._absolute_past[:, 0, -1] - self._absolute_past[:, 0, -1 - lag]
        speed = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        default = torch.zeros_like(displacement)
        default[:, 0] = 1.0
        heading = torch.where(speed > 0.03, displacement / speed.clamp_min(1.0e-6), default)
        cosine = heading[:, 0, None, None]
        sine = heading[:, 1, None, None]
        x = relative[..., 0] * cosine - relative[..., 1] * sine
        y = relative[..., 0] * sine + relative[..., 1] * cosine
        return origin[:, None, None] + torch.stack((x, y), dim=-1)

    def query_scene(self, relative_anchors: Tensor) -> Tensor:
        assert self._scene_ids is not None
        global_points = self.local_to_global(relative_anchors)
        sizes = self.image_wh[self._scene_ids]
        pixel_points = global_points * self.pixel_scale
        x = 2.0 * pixel_points[..., 0] / (sizes[:, None, None, 0] - 1.0) - 1.0
        y = 2.0 * pixel_points[..., 1] / (sizes[:, None, None, 1] - 1.0) - 1.0
        grid = torch.stack((x, y), dim=-1).reshape(global_points.shape[0], -1, 1, 2)
        maps = self.scene_maps[self._scene_ids].to(dtype=relative_anchors.dtype) / 255.0
        sampled = F.grid_sample(
            maps, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return sampled[..., 0].transpose(1, 2).reshape(
            global_points.shape[0], self.num_modes, self.anchor_count * self.scene_channels
        )

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context = self.context(torch.cat(
            (recent, pooled, self.motion_features(past_loc)), dim=-1
        ))
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        scale = F.softplus(self.proposal_scale(context)) + 0.25
        source_anchors = scale[:, :, None, None] * (
            self.anchor_templates[None, None]
            + 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(
                *mode.shape[:-1], self.anchor_count, 2
            )
        )
        corrected = source_anchors.clone()
        focal_source = source_anchors[:, 0]
        scene_features = self.query_scene(focal_source)
        correction_features = torch.cat(
            (mode[:, 0], focal_source.flatten(-2), scene_features), dim=-1
        )
        correction = torch.tanh(self.scene_correction(correction_features)).reshape(
            -1, self.num_modes, self.anchor_count, 2
        ) * self.scene_max_offsets[None, None, :, None]
        corrected[:, 0] = focal_source + correction
        relative = torch.einsum("tp,bakpc->baktc", self.interpolation, corrected)
        trajectories = past_loc[:, :, None, -1:, :] + relative
        logits = self.score_head(
            torch.cat((mode, corrected.flatten(-2)), dim=-1)
        ).squeeze(-1)
        self.last_anchor_proposals = (
            past_loc[:, :, None, None, -1, :] + source_anchors
        ).detach()
        self.last_corrected_anchors = (
            past_loc[:, :, None, None, -1, :] + corrected
        ).detach()
        self.last_coarse_trajectories = trajectories.detach()
        self.last_selected_indices = None
        self._training_target = None
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name == "learned_anchor_control":
        model.decoder = SOURCE.HierarchicalLearnedAnchorDecoder(
            token_dim=model.token_dim, pred_len=model.pred_len, num_modes=20,
            anchor_steps=(4, 8, model.pred_len), use_refiner=False,
            top_m=20, hidden_dim=64,
        )
    elif name == "semantic_anchor_query":
        asset_text = os.environ.get("EXP147_SCENE_ASSET_PATH", "").strip()
        if not asset_text:
            raise RuntimeError("EXP147_SCENE_ASSET_PATH is required")
        asset_path = Path(asset_text)
        if not asset_path.is_file():
            raise FileNotFoundError(asset_path)
        model.decoder = SemanticAnchorQueryDecoder(
            model.token_dim, model.pred_len, asset_path
        )
    else:
        raise ValueError(f"Unknown EXP-147 decoder: {name}")
    cfg.MODEL.DECODER.NUM_MODES = 20
