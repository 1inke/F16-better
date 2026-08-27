"""Mechanism validation around the successful EXP-142A correction model."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
CONTROLLED_PATH = ROOT.parent / "CONTROLLED_ANCHOR_CORRECTION_SERIES" / "controlled_anchor_correction_models.py"
EXPERIMENTS = {
    "v1_endpoint_only_correction": "endpoint_only",
    "v2_wide_endpoint_correction": "wide_endpoint",
    "v3_corrected_quality_rescoring": "quality_rescore",
}


def _load_controlled():
    spec = importlib.util.spec_from_file_location("exp142_validation_source", CONTROLLED_PATH)
    if spec is None or spec.loader is None: raise ImportError(CONTROLLED_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


CONTROLLED = _load_controlled()


class CorrectedQualityDecoder(CONTROLLED.FrozenControlledAnchorDecoder):
    """Use frozen corrected trajectories and learn only their quality logits."""

    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(token_dim, pred_len)
        quality_input = 64 + self.anchor_count * 4
        self.quality_head = nn.Sequential(
            nn.LayerNorm(quality_input), nn.Linear(quality_input, 64),
            nn.GELU(), nn.Linear(64, 1),
        )
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.eval(); self.quality_head.train(True)

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        trajectories, _ = super().forward(encoded_tokens, past_loc)
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context = self.context(torch.cat(
            (recent, pooled, self.motion_features(past_loc)), dim=-1
        ))
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        scale = torch.nn.functional.softplus(self.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(
            *mode.shape[:-1], self.anchor_count, 2
        )
        source_anchors = scale[:, :, None, None] * (
            self.anchor_templates[None, None] + delta
        )
        correction = self.refiner(mode, source_anchors)
        corrected_anchors = source_anchors + correction
        quality_features = torch.cat((
            mode, corrected_anchors.flatten(-2), correction.flatten(-2)
        ), dim=-1)
        logits = self.quality_head(quality_features).squeeze(-1)
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name not in EXPERIMENTS: raise ValueError(f"Unknown EXP-143 decoder: {name}")
    checkpoint_text = os.environ.get("EXP143_SOURCE_CHECKPOINT", "").strip()
    if not checkpoint_text: raise RuntimeError("EXP143_SOURCE_CHECKPOINT is required")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    kind = EXPERIMENTS[name]

    if kind == "quality_rescore":
        model.decoder = CorrectedQualityDecoder(model.token_dim, model.pred_len)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        expected = {k for k in model.state_dict() if k.startswith("decoder.quality_head.")}
        if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
            raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))
        for parameter in model.parameters(): parameter.requires_grad_(False)
        for parameter in model.decoder.quality_head.parameters(): parameter.requires_grad_(True)
        cfg.MODEL.DECODER.CLASSIFICATION_WEIGHT = 1.0
    else:
        model.decoder = CONTROLLED.FrozenControlledAnchorDecoder(model.token_dim, model.pred_len)
        offsets = [0.0, 0.0, 0.35] if kind == "endpoint_only" else [0.20, 0.25, 0.50]
        model.decoder.refiner.max_offsets.copy_(torch.tensor(offsets, dtype=torch.float32))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        expected = {k for k in model.state_dict() if k.startswith("decoder.refiner.")}
        if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
            raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))
        for parameter in model.parameters(): parameter.requires_grad_(False)
        for parameter in model.decoder.refiner.parameters(): parameter.requires_grad_(True)

    model.decoder.validation_mechanism = kind
    model.decoder.source_checkpoint = str(checkpoint)
    cfg.MODEL.DECODER.NUM_MODES = 20

