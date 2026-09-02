"""EXP-175B-R2: observation-conditioned appearance-to-mode anchor correction.

The accepted EXP-175S0 F16+H3+Joint-V2 checkpoint is frozen.  A tiny,
zero-initialized residual branch queries the cached SDD reference image at the
three proposed anchor positions and compares those features with appearance
observed along the history.  No future coordinates are used as model input.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
JOINT_ADAPTER = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
EXPERIMENT = "exp175br2_appearance_to_mode"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


JOINT = _load("exp175br2_joint_v2_source", JOINT_ADAPTER)


def appearance_embedding(values: Tensor) -> Tensor:
    """Match the fixed embedding that passed EXP-175A-R2."""
    rgb = values[..., :3]
    hue, saturation, value = values[..., 3], values[..., 4], values[..., 5]
    texture, gradient = values[..., 6], values[..., 7]
    angle = 2.0 * torch.pi * hue
    return torch.cat(
        (
            0.45 * rgb,
            torch.sin(angle)[..., None],
            torch.cos(angle)[..., None],
            saturation[..., None],
            value[..., None],
            0.60 * texture[..., None],
            0.35 * gradient[..., None],
        ),
        dim=-1,
    )


class AppearanceAnchorResidual(nn.Module):
    """A 5.6K-parameter bounded correction for t4/t8/t12 anchors."""

    def __init__(self, mode_dim: int = 64, anchor_count: int = 3) -> None:
        super().__init__()
        self.anchor_count = int(anchor_count)
        # RGB(3) + cyclic hue(2) + saturation/value(2) + texture/gradient(2).
        embedding_dim = 9
        input_dim = mode_dim + anchor_count * (2 + embedding_dim + 1 + 1)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 48),
            nn.GELU(),
            nn.Linear(48, anchor_count * 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer(
            "max_offsets", torch.tensor([0.05, 0.08, 0.15], dtype=torch.float32)
        )

    def forward(
        self,
        mode: Tensor,
        source_anchors: Tensor,
        anchor_embedding: Tensor,
        similarity: Tensor,
        in_bounds: Tensor,
    ) -> Tensor:
        features = torch.cat(
            (
                mode,
                source_anchors.flatten(-2),
                anchor_embedding.flatten(-2),
                similarity,
                in_bounds.to(mode.dtype),
            ),
            dim=-1,
        )
        raw = self.network(features).reshape_as(source_anchors)
        shape = (1,) * (raw.ndim - 2) + (-1, 1)
        return torch.tanh(raw) * self.max_offsets.view(*shape)


class ObservationConditionedAppearanceDecoder(JOINT.JointTrainableV2Decoder):
    """Frozen Joint-V2 plus a trainable appearance-conditioned anchor delta."""

    def __init__(self, token_dim: int, pred_len: int, bank_path: Path) -> None:
        super().__init__(token_dim, pred_len)
        bank = np.load(bank_path, allow_pickle=False)
        maps = torch.from_numpy(np.asarray(bank["maps"], dtype=np.uint8))
        sizes = torch.from_numpy(np.asarray(bank["image_wh"], dtype=np.float32))
        self.pixel_scale = float(np.asarray(bank["pixel_scale"]).reshape(-1)[0])
        channel_names = [str(x) for x in bank["channel_names"].tolist()]
        expected = ["r", "g", "b", "h", "s", "v", "texture", "gradient"]
        if channel_names != expected or maps.ndim != 4:
            raise RuntimeError(
                f"Unexpected appearance bank: channels={channel_names}, shape={tuple(maps.shape)}"
            )
        # Cached maps are runtime assets rather than learned model state.
        self.register_buffer("appearance_maps", maps, persistent=False)
        self.register_buffer("image_wh", sizes, persistent=False)
        self.scene_residual = AppearanceAnchorResidual(mode_dim=64, anchor_count=3)
        self._scene_ids: Tensor | None = None
        self._absolute_past: Tensor | None = None
        self.last_scene_correction: Tensor | None = None
        self.last_scene_similarity: Tensor | None = None
        self.appearance_bank_path = str(bank_path)

    def set_scene_batch(self, scene_ids: Tensor, absolute_past: Tensor) -> None:
        if scene_ids.ndim != 1 or len(scene_ids) != len(absolute_past):
            raise RuntimeError(
                f"Scene batch mismatch: ids={tuple(scene_ids.shape)}, past={tuple(absolute_past.shape)}"
            )
        self._scene_ids = scene_ids
        self._absolute_past = absolute_past

    def apply_training_policy(self, owner: nn.Module) -> None:
        # Reassert the causal ablation after the generic trainer calls model.train().
        owner.eval()
        self.scene_residual.train(True)

    def _query(self, global_points: Tensor) -> tuple[Tensor, Tensor]:
        if self._scene_ids is None:
            raise RuntimeError("set_scene_batch(scene_ids, absolute_past) was not called")
        ids = self._scene_ids
        if bool((ids < 0).any()) or bool((ids >= len(self.appearance_maps)).any()):
            raise RuntimeError(f"Invalid scene IDs: min={int(ids.min())}, max={int(ids.max())}")
        sizes = self.image_wh[ids]
        pixels = global_points * self.pixel_scale
        view = (len(ids),) + (1,) * (global_points.ndim - 2) + (2,)
        sizes = sizes.view(*view)
        x = 2.0 * pixels[..., 0] / (sizes[..., 0] - 1.0) - 1.0
        y = 2.0 * pixels[..., 1] / (sizes[..., 1] - 1.0) - 1.0
        inside = (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
        original = global_points.shape[:-1]
        grid = torch.stack((x, y), dim=-1).reshape(len(ids), -1, 1, 2)
        maps = self.appearance_maps[ids].to(global_points.dtype) / 255.0
        sampled = F.grid_sample(
            maps, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        sampled = sampled[..., 0].transpose(1, 2).reshape(*original, maps.shape[1])
        return sampled, inside

    def _appearance_context(
        self, source_anchors: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self._absolute_past is None:
            raise RuntimeError("Absolute history is unavailable")
        absolute_past = self._absolute_past
        # Focal-heading normalization rotates every agent with this same heading.
        lag = min(3, absolute_past.shape[-2] - 1)
        displacement = absolute_past[:, 0, -1] - absolute_past[:, 0, -1 - lag]
        speed = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        fallback = torch.zeros_like(displacement)
        fallback[:, 0] = 1.0
        heading = torch.where(speed > 0.03, displacement / speed.clamp_min(1e-6), fallback)
        cosine = heading[:, 0, None, None, None]
        sine = heading[:, 1, None, None, None]
        local_x, local_y = source_anchors[..., 0], source_anchors[..., 1]
        global_delta = torch.stack(
            (local_x * cosine - local_y * sine, local_x * sine + local_y * cosine),
            dim=-1,
        )
        global_anchors = absolute_past[:, :, -1, None, None] + global_delta
        anchor_values, anchor_inside = self._query(global_anchors)
        history_values, history_inside = self._query(absolute_past)
        anchor_emb = appearance_embedding(anchor_values)
        history_emb = appearance_embedding(history_values)

        batch = len(absolute_past)
        candidate_flat = anchor_emb.reshape(batch, -1, anchor_emb.shape[-1])
        history_flat = history_emb.reshape(batch, -1, history_emb.shape[-1])
        # Padded agents are exactly zero in the processed arrays.  Do not let
        # their map samples enter the observation-conditioned prototype bank.
        agent_valid = absolute_past.abs().sum(dim=(-1, -2)) > 1e-6
        prototype_valid = (history_inside & agent_valid[..., None]).reshape(batch, -1)
        distances = torch.cdist(candidate_flat, history_flat) / (anchor_emb.shape[-1] ** 0.5)
        distances = distances.masked_fill(~prototype_valid[:, None], float("inf"))
        nearest = distances.amin(dim=-1).reshape(anchor_emb.shape[:-1])
        nearest = torch.where(torch.isfinite(nearest), nearest, torch.ones_like(nearest))
        similarity = torch.exp(-nearest / 0.18) * anchor_inside.to(nearest.dtype)
        return anchor_emb, similarity, anchor_inside

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context = self.context(
            torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1)
        )
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        scale = F.softplus(self.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(
            *mode.shape[:-1], self.anchor_count, 2
        )
        source_anchors = scale[:, :, None, None] * (
            self.anchor_templates[None, None] + delta
        )
        anchor_emb, similarity, inside = self._appearance_context(source_anchors)
        scene_delta = self.scene_residual(
            mode, source_anchors, anchor_emb, similarity, inside
        )
        scene_anchors = source_anchors + scene_delta
        v2_delta = self.refiner(mode, scene_anchors)
        corrected_anchors = scene_anchors + v2_delta
        relative = torch.einsum("tp,bakpc->baktc", self.interpolation, corrected_anchors)
        trajectories = past_loc[:, :, None, -1:, :] + relative

        # Preserve the frozen original scoring mechanism and its inputs.
        logits = self.score_head(
            torch.cat((mode, source_anchors.flatten(-2)), dim=-1)
        ).squeeze(-1)
        batch, agents, modes = logits.shape
        selected = torch.arange(modes, device=logits.device).view(1, 1, modes)
        self.last_selected_indices = selected.expand(batch, agents, modes).detach()
        self.last_coarse_trajectories = (
            past_loc[:, :, None, -1:, :]
            + torch.einsum("tp,bakpc->baktc", self.interpolation, source_anchors)
        ).detach()
        self.last_anchor_proposals = (
            past_loc[:, :, None, None, -1, :] + source_anchors
        ).detach()
        self.last_scene_adjusted_anchors = (
            past_loc[:, :, None, None, -1, :] + scene_anchors
        ).detach()
        self.last_corrected_anchors = (
            past_loc[:, :, None, None, -1, :] + corrected_anchors
        ).detach()
        self.last_scene_correction = scene_delta.detach()
        self.last_scene_similarity = similarity.detach()
        self._training_target = None
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name != EXPERIMENT:
        raise ValueError(f"Unknown EXP-175B-R2 experiment: {name}")
    checkpoint = Path(os.environ.get("EXP175BR2_SOURCE_CHECKPOINT", "")).resolve()
    bank_path = Path(os.environ.get("EXP175BR2_APPEARANCE_BANK", "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"EXP-175S0 checkpoint missing: {checkpoint}")
    if not bank_path.is_file():
        raise FileNotFoundError(f"Appearance bank missing: {bank_path}")

    decoder = ObservationConditionedAppearanceDecoder(
        model.token_dim, model.pred_len, bank_path
    )
    model.decoder = decoder
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {
        key for key in model.state_dict() if key.startswith("decoder.scene_residual.")
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(f"Unexpected missing keys: {sorted(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in decoder.scene_residual.parameters():
        parameter.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable > 30_000:
        raise RuntimeError(f"Scene branch exceeds 0.03M parameters: {trainable}")
    decoder.source_checkpoint = str(checkpoint)
    decoder.training_policy_name = "frozen_s0_train_scene_anchor_residual_only"
    decoder.no_future_scene_input = True
    cfg.MODEL.DECODER.NUM_MODES = 20
