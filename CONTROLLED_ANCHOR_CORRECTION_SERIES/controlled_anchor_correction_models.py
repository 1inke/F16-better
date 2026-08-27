"""Frozen three-anchor proposals with learned, bounded second-stage anchor correction."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT.parent / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"
EXPERIMENTS = {
    "c1_controlled_anchor_correction": "standard",
    "c2_samepath_aligned_correction": "samepath",
}


def _load_source():
    spec = importlib.util.spec_from_file_location("exp139_correction_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source()


class BoundedAnchorCorrection(nn.Module):
    """Correct all three anchors per mode while limiting source displacement."""

    def __init__(self, hidden_dim: int = 64, anchor_count: int = 3) -> None:
        super().__init__()
        input_dim = hidden_dim + anchor_count * 2
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, anchor_count * 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        # Earlier anchors receive smaller corrections; the endpoint may move,
        # but remains bounded to prevent destroying source candidate coverage.
        self.register_buffer(
            "max_offsets", torch.tensor([0.20, 0.25, 0.35], dtype=torch.float32)
        )

    def forward(self, mode: Tensor, anchors: Tensor) -> Tensor:
        features = torch.cat((mode, anchors.flatten(-2)), dim=-1)
        raw = self.network(features).reshape(*anchors.shape)
        return torch.tanh(raw) * self.max_offsets.view(
            *((1,) * (raw.ndim - 2)), -1, 1
        )


class FrozenControlledAnchorDecoder(SOURCE.HierarchicalLearnedAnchorDecoder):
    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(
            token_dim=token_dim, pred_len=pred_len, num_modes=20,
            anchor_steps=(4, 8, pred_len), use_refiner=False, top_m=20,
            hidden_dim=64,
        )
        self.refiner = BoundedAnchorCorrection(hidden_dim=64, anchor_count=3)
        self.use_refiner = True
        self.top_m = 20
        self.last_corrected_anchors: Tensor | None = None

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.eval()
        self.refiner.train(True)

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context_input = torch.cat(
            (recent, pooled, self.motion_features(past_loc)), dim=-1
        )
        context = self.context(context_input)
        mode = self.mode_mlp(
            context[:, :, None] + self.mode_embedding[None, None]
        )
        scale = torch.nn.functional.softplus(self.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(
            *mode.shape[:-1], self.anchor_count, 2
        )
        source_anchors = scale[:, :, None, None] * (
            self.anchor_templates[None, None] + delta
        )
        source_relative = torch.einsum(
            "tp,bakpc->baktc", self.interpolation, source_anchors
        )
        last = past_loc[:, :, None, -1:, :]
        coarse = last + source_relative
        logits = self.score_head(
            torch.cat((mode, source_anchors.flatten(-2)), dim=-1)
        ).squeeze(-1)

        correction = self.refiner(mode, source_anchors)
        corrected_anchors = source_anchors + correction
        corrected_relative = torch.einsum(
            "tp,bakpc->baktc", self.interpolation, corrected_anchors
        )
        trajectories = last + corrected_relative

        batch, agents, modes = logits.shape
        selected = torch.arange(modes, device=logits.device).view(1, 1, modes)
        self.last_selected_indices = selected.expand(batch, agents, modes).detach()
        self.last_coarse_trajectories = coarse.detach()
        self.last_anchor_proposals = (
            past_loc[:, :, None, None, -1, :] + source_anchors
        ).detach()
        self.last_corrected_anchors = (
            past_loc[:, :, None, None, -1, :] + corrected_anchors
        ).detach()
        self._training_target = None
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown EXP-142 decoder: {name}")
    checkpoint_text = os.environ.get("EXP142_SOURCE_CHECKPOINT", "").strip()
    if not checkpoint_text:
        raise RuntimeError("EXP142_SOURCE_CHECKPOINT must point to EXP-139C best_selected.pth")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model.decoder = FrozenControlledAnchorDecoder(model.token_dim, model.pred_len)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    expected_missing = {
        key for key in model.state_dict() if key.startswith("decoder.refiner.")
    }
    if missing != expected_missing:
        raise RuntimeError(f"Unexpected missing source keys: {sorted(missing)}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected source keys: {incompatible.unexpected_keys}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.refiner.parameters():
        parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.correction_objective = EXPERIMENTS[name]

    if EXPERIMENTS[name] == "samepath":
        # Align winner selection and direct endpoint supervision more strongly
        # with the ADE + 0.5*FDE checkpoint criterion.
        cfg.MODEL.DECODER.WINNER_FDE_WEIGHT = 0.50
        cfg.MODEL.DECODER.ENDPOINT_LOSS_WEIGHT = 0.50
        cfg.MODEL.DECODER.DIVERSITY_WEIGHT = 0.08
    cfg.MODEL.DECODER.NUM_MODES = 20

