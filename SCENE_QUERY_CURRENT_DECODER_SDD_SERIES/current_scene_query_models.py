"""EXP-147B: semantic correction after current refined anchors, before templates."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT.parent / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"


def _load_source():
    spec = importlib.util.spec_from_file_location("exp147b_mode_template_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source()


class CurrentAnchoredModeTemplateDecoder(SOURCE.GroupedModeTemplateDecoder):
    """The EXP-146B architecture trained normally on the SDD protocol."""

    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(token_dim, pred_len, use_template=True, use_groups=False)
        self.cv_reference = False

    def apply_training_policy(self, owner: nn.Module) -> None:
        # This is a fresh SDD training run, not an ETH/UCY frozen-decoder probe.
        owner.train(True)


class SemanticCurrentAnchorDecoder(CurrentAnchoredModeTemplateDecoder):
    """Correct refined focal anchors once using sparse offline semantic features."""

    def __init__(self, token_dim: int, pred_len: int, asset_path: Path) -> None:
        super().__init__(token_dim, pred_len)
        bank = np.load(asset_path, allow_pickle=False)
        maps = torch.from_numpy(np.asarray(bank["maps"], dtype=np.uint8))
        image_wh = torch.from_numpy(np.asarray(bank["image_wh"], dtype=np.float32))
        self.register_buffer("scene_maps", maps, persistent=False)
        self.register_buffer("image_wh", image_wh, persistent=False)
        self.pixel_scale = float(np.asarray(bank["pixel_scale"]).reshape(-1)[0])
        self.scene_channels = int(maps.shape[1])
        input_dim = 64 + 6 + 3 * self.scene_channels
        self.scene_correction = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 6),
        )
        # S1 begins as the same trajectory model as S0.
        nn.init.zeros_(self.scene_correction[-1].weight)
        nn.init.zeros_(self.scene_correction[-1].bias)
        self.register_buffer("scene_max_offsets", torch.tensor([0.35, 0.55, 0.90]))
        self.heading_normalized = os.environ.get("EXP147B_HEADING_NORMALIZED", "1") == "1"
        self._scene_ids: Tensor | None = None
        self._absolute_past: Tensor | None = None
        self.last_scene_correction: Tensor | None = None

    def set_scene_batch(self, scene_ids: Tensor, absolute_past: Tensor) -> None:
        if (scene_ids < 0).any():
            raise RuntimeError("EXP-147B requires per-sample SDD scene IDs")
        self._scene_ids = scene_ids
        self._absolute_past = absolute_past

    def _global_points(self, relative: Tensor) -> Tensor:
        assert self._absolute_past is not None
        origin = self._absolute_past[:, 0, -1]
        if not self.heading_normalized:
            return origin[:, None, None] + relative
        lag = min(3, self._absolute_past.shape[-2] - 1)
        displacement = self._absolute_past[:, 0, -1] - self._absolute_past[:, 0, -1 - lag]
        speed = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        fallback = torch.zeros_like(displacement); fallback[:, 0] = 1.0
        heading = torch.where(speed > 0.03, displacement / speed.clamp_min(1e-6), fallback)
        cosine, sine = heading[:, 0, None, None], heading[:, 1, None, None]
        x = relative[..., 0] * cosine - relative[..., 1] * sine
        y = relative[..., 0] * sine + relative[..., 1] * cosine
        return origin[:, None, None] + torch.stack((x, y), dim=-1)

    def _query_scene(self, anchors: Tensor) -> Tensor:
        assert self._scene_ids is not None
        points = self._global_points(anchors)
        sizes = self.image_wh[self._scene_ids]
        pixels = points * self.pixel_scale
        x = 2.0 * pixels[..., 0] / (sizes[:, None, None, 0] - 1.0) - 1.0
        y = 2.0 * pixels[..., 1] / (sizes[:, None, None, 1] - 1.0) - 1.0
        grid = torch.stack((x, y), dim=-1).reshape(points.shape[0], -1, 1, 2)
        maps = self.scene_maps[self._scene_ids].to(dtype=anchors.dtype) / 255.0
        sampled = F.grid_sample(maps, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled[..., 0].transpose(1, 2).reshape(
            points.shape[0], self.num_modes, 3 * self.scene_channels
        )

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent, pooled = encoded_tokens[:, :, -1], encoded_tokens.mean(dim=2)
        context = self.context(torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1))
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        scale = F.softplus(self.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(*mode.shape[:-1], 3, 2)
        source = scale[:, :, None, None] * (self.anchor_templates[None, None] + delta)
        refined = source + self.refiner(mode, source)

        focal_refined = refined[:, 0]
        scene = self._query_scene(focal_refined)
        correction_input = torch.cat((mode[:, 0], focal_refined.flatten(-2), scene), dim=-1)
        correction = torch.tanh(self.scene_correction(correction_input)).reshape(-1, self.num_modes, 3, 2)
        correction = correction * self.scene_max_offsets[None, None, :, None]
        scene_anchors = focal_refined + correction
        final_base = torch.cat((scene_anchors[:, None], refined[:, 1:]), dim=1)

        base_relative = torch.einsum("tp,bakpc->baktc", self.interpolation, final_base)
        focal_relative, focal_final = self.template_head(
            context[:, :1], mode[:, :1], final_base[:, :1], self.interpolation, None
        )
        final_relative = torch.cat((focal_relative, base_relative[:, 1:]), dim=1)
        final_anchors = torch.cat((focal_final, final_base[:, 1:]), dim=1)
        logits = self.score_head(torch.cat((mode, source.flatten(-2)), dim=-1)).squeeze(-1)
        last = past_loc[:, :, None, -1:, :]
        trajectories = last + final_relative
        batch, agents, modes = logits.shape
        selected = torch.arange(modes, device=logits.device).view(1, 1, modes)
        self.last_selected_indices = selected.expand(batch, agents, modes).detach()
        self.last_coarse_trajectories = (last + base_relative).detach()
        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + refined).detach()
        self.last_corrected_anchors = (past_loc[:, :, None, None, -1, :] + final_anchors).detach()
        self.last_scene_correction = correction.detach()
        self._training_target = None
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name == "s0_current":
        model.decoder = CurrentAnchoredModeTemplateDecoder(model.token_dim, model.pred_len)
    elif name == "s1_scene_anchor_query":
        asset = Path(os.environ.get("EXP147B_SCENE_ASSET_PATH", ""))
        if not asset.is_file():
            raise FileNotFoundError("EXP147B_SCENE_ASSET_PATH must point to sdd_semantic_bank.npz")
        model.decoder = SemanticCurrentAnchorDecoder(model.token_dim, model.pred_len, asset)
    else:
        raise ValueError(f"Unknown EXP-147B variant: {name}")
    cfg.MODEL.DECODER.NUM_MODES = 20
