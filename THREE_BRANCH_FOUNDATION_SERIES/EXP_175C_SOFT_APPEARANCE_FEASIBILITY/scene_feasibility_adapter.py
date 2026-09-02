"""EXP-175C: soft observation-conditioned appearance feasibility loss."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
from torch import Tensor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
B_ADAPTER = (
    ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175BR2_APPEARANCE_TO_MODE"
    / "scene_appearance_anchor_adapter.py"
)
EXPERIMENT = "exp175c_soft_appearance_feasibility"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B = _load("exp175c_scene_branch_source", B_ADAPTER)


class SoftAppearanceFeasibilityDecoder(B.ObservationConditionedAppearanceDecoder):
    """The B-R2 branch plus a training-only soft path feasibility objective."""

    def __init__(self, token_dim: int, pred_len: int, bank_path: Path, weight: float) -> None:
        super().__init__(token_dim, pred_len, bank_path)
        self.scene_loss_weight = float(weight)
        self.scene_loss_topk = 5
        self._last_decoder_past: Tensor | None = None
        self.last_scene_auxiliary_raw: Tensor | None = None

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        self._last_decoder_past = past_loc
        return super().forward(encoded_tokens, past_loc)

    def _global_focal_paths(self, prediction: Tensor) -> Tensor:
        if self._absolute_past is None or self._last_decoder_past is None:
            raise RuntimeError("Scene/history context is unavailable for auxiliary loss")
        absolute_past = self._absolute_past
        local_last = self._last_decoder_past[:, :1, -1:, :].unsqueeze(2)
        relative = prediction - local_last
        lag = min(3, absolute_past.shape[-2] - 1)
        displacement = absolute_past[:, 0, -1] - absolute_past[:, 0, -1 - lag]
        speed = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        fallback = torch.zeros_like(displacement)
        fallback[:, 0] = 1.0
        heading = torch.where(speed > 0.03, displacement / speed.clamp_min(1e-6), fallback)
        cosine = heading[:, 0, None, None, None]
        sine = heading[:, 1, None, None, None]
        x, y = relative[..., 0], relative[..., 1]
        global_delta = torch.stack(
            (x * cosine - y * sine, x * sine + y * cosine), dim=-1
        )
        return absolute_past[:, :1, -1:, :].unsqueeze(2) + global_delta

    def _path_similarity(self, prediction: Tensor) -> Tensor:
        if self._absolute_past is None:
            raise RuntimeError("Absolute history is unavailable")
        global_paths = self._global_focal_paths(prediction)
        candidate_values, candidate_inside = self._query(global_paths)
        history_values, history_inside = self._query(self._absolute_past)
        candidate = B.appearance_embedding(candidate_values)
        prototype = B.appearance_embedding(history_values)
        batch = len(prediction)
        candidate_flat = candidate.reshape(batch, -1, candidate.shape[-1])
        prototype_flat = prototype.reshape(batch, -1, prototype.shape[-1])
        agent_valid = self._absolute_past.abs().sum(dim=(-1, -2)) > 1e-6
        prototype_valid = (history_inside & agent_valid[..., None]).reshape(batch, -1)
        distance = torch.cdist(candidate_flat, prototype_flat) / (candidate.shape[-1] ** 0.5)
        distance = distance.masked_fill(~prototype_valid[:, None], float("inf"))
        nearest = distance.amin(dim=-1).reshape(candidate.shape[:-1])
        nearest = torch.where(torch.isfinite(nearest), nearest, torch.ones_like(nearest))
        point_similarity = torch.exp(-nearest / 0.18) * candidate_inside.to(nearest.dtype)
        return point_similarity.mean(dim=-1).squeeze(1)

    def auxiliary_training_loss(
        self, prediction: Tensor, target: Tensor, logits: Tensor
    ) -> Tensor:
        # GT is used only to select a detached near-oracle neighbourhood for the
        # training loss; it is never supplied to the scene branch forward pass.
        distance = torch.linalg.vector_norm(prediction - target[:, :, None], dim=-1)
        quality = distance.mean(dim=-1) + 0.25 * distance[..., -1]
        quality = quality.squeeze(1)
        gap = quality.detach() - quality.detach().amin(dim=-1, keepdim=True)
        scale = gap.mean(dim=-1, keepdim=True).clamp_min(0.05)
        responsibility_logits = -gap / (0.10 * scale)
        topk = min(self.scene_loss_topk, responsibility_logits.shape[-1])
        threshold = responsibility_logits.topk(topk, dim=-1).values[..., -1:]
        responsibility_logits = responsibility_logits.masked_fill(
            responsibility_logits < threshold, float("-inf")
        )
        weights = torch.softmax(responsibility_logits, dim=-1).detach()
        violation = 1.0 - self._path_similarity(prediction)
        raw_loss = (weights * violation).sum(dim=-1).mean()
        self.last_scene_auxiliary_raw = raw_loss.detach()
        return self.scene_loss_weight * raw_loss


def replace_decoder(model, cfg, name: str) -> None:
    if name != EXPERIMENT:
        raise ValueError(f"Unknown EXP-175C experiment: {name}")
    checkpoint = Path(os.environ.get("EXP175C_SOURCE_CHECKPOINT", "")).resolve()
    bank = Path(os.environ.get("EXP175C_APPEARANCE_BANK", "")).resolve()
    weight = float(os.environ.get("EXP175C_SCENE_LOSS_WEIGHT", "0.02"))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"EXP-175S0 checkpoint missing: {checkpoint}")
    if not bank.is_file():
        raise FileNotFoundError(f"Appearance bank missing: {bank}")
    if not 0.0 < weight <= 0.10:
        raise ValueError(f"Scene loss weight must be in (0, 0.10], got {weight}")

    decoder = SoftAppearanceFeasibilityDecoder(model.token_dim, model.pred_len, bank, weight)
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
        raise RuntimeError(f"Scene branch exceeds parameter cap: {trainable}")
    decoder.source_checkpoint = str(checkpoint)
    decoder.training_policy_name = "frozen_s0_scene_residual_plus_soft_feasibility"
    decoder.no_future_scene_input = True
    decoder.future_used_for_training_neighbourhood_only = True
    cfg.MODEL.DECODER.NUM_MODES = 20

