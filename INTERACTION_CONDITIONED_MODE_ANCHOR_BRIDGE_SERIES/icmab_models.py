"""IC1: a social-only, mode-specific bridge into the frozen EXP-146B decoder."""

from __future__ import annotations

import contextvars
import importlib.util
import os
from pathlib import Path

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT.parent / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"
EXPERIMENTS = {"ic1_social_mode_anchor_bridge"}
_SOCIAL_DELTA: contextvars.ContextVar[Tensor | None] = contextvars.ContextVar(
    "icmab_social_delta", default=None
)


def _load_source():
    spec = importlib.util.spec_from_file_location("icmab_exp146_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source()


class SocialDeltaTapGate(nn.Sequential):
    """Preserve the original social gate while exposing gate*social.

    Keeping the same Sequential child indices preserves all source checkpoint
    parameter names. ContextVar also makes the tap work after EMA deepcopy.
    """

    def __init__(self, *modules: nn.Module, dim: int) -> None:
        super().__init__(*modules)
        self.dim = int(dim)
        self.last_social_delta: Tensor | None = None

    def forward(self, features: Tensor) -> Tensor:
        logits = super().forward(features)
        social = features[..., self.dim:]
        delta = torch.sigmoid(logits) * social
        self.last_social_delta = delta
        _SOCIAL_DELTA.set(delta)
        return logits


class LowRankInteractionBridge(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 8) -> None:
        super().__init__()
        self.down = nn.Linear(input_dim, rank)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, output_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, social_delta: Tensor) -> Tensor:
        return self.up(self.activation(self.down(social_delta)))


class InteractionConditionedModeAnchorDecoder(SOURCE.GroupedModeTemplateDecoder):
    """Frozen EXP-146B plus a trainable 64->8->64 mode-specific bridge."""

    def __init__(
        self, token_dim: int, pred_len: int, social_dim: int, rank: int = 8
    ) -> None:
        super().__init__(token_dim, pred_len, use_template=True, use_groups=False)
        mode_dim = int(self.mode_embedding.shape[-1])
        self.social_dim = int(social_dim)
        self.mode_dim = mode_dim
        self.interaction_bridge = LowRankInteractionBridge(
            self.social_dim, mode_dim, rank
        )
        self.mode_gate = nn.Linear(mode_dim, 20)
        nn.init.zeros_(self.mode_gate.weight)
        nn.init.zeros_(self.mode_gate.bias)
        self.bridge_rank = int(rank)
        self.last_interaction_token: Tensor | None = None
        self.last_bridge_output: Tensor | None = None
        self.last_mode_gates: Tensor | None = None
        self.last_conditioned_mode_delta: Tensor | None = None

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.train(True)
        self.eval()
        self.interaction_bridge.train(True)
        self.mode_gate.train(True)

    def _condition_modes(self, mode: Tensor) -> Tensor:
        social_delta = _SOCIAL_DELTA.get()
        if social_delta is None:
            raise RuntimeError("IC1 decoder did not receive the F16 social delta")
        if (
            social_delta.shape[:2] != mode.shape[:2]
            or social_delta.shape[-1] != self.social_dim
            or mode.shape[-1] != self.mode_dim
        ):
            raise RuntimeError(
                f"social/mode shape mismatch: {tuple(social_delta.shape)} / {tuple(mode.shape)}"
            )
        bridge = self.interaction_bridge(social_delta)
        gates = torch.sigmoid(self.mode_gate(bridge))
        mode_delta = gates[..., None] * bridge[:, :, None, :]
        self.last_interaction_token = social_delta.detach()
        self.last_bridge_output = bridge.detach()
        self.last_mode_gates = gates.detach()
        self.last_conditioned_mode_delta = mode_delta.detach()
        return mode + mode_delta

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context = self.context(
            torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1)
        )
        base_mode = self.mode_mlp(
            context[:, :, None] + self.mode_embedding[None, None]
        )
        mode = self._condition_modes(base_mode)
        scale = torch.nn.functional.softplus(self.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(
            *mode.shape[:-1], 3, 2
        )
        source_anchors = scale[:, :, None, None] * (
            self.anchor_templates[None, None] + delta
        )
        base_anchors = source_anchors + self.refiner(mode, source_anchors)
        base_relative = torch.einsum(
            "tp,bakpc->baktc", self.interpolation, base_anchors
        )
        focal_relative, focal_anchors = self.template_head(
            context[:, :1], mode[:, :1], base_anchors[:, :1],
            self.interpolation, None,
        )
        final_relative = torch.cat((focal_relative, base_relative[:, 1:]), dim=1)
        final_anchors = torch.cat((focal_anchors, base_anchors[:, 1:]), dim=1)
        logits = self.score_head(
            torch.cat((mode, source_anchors.flatten(-2)), dim=-1)
        ).squeeze(-1)
        last = past_loc[:, :, None, -1:, :]
        trajectories = last + final_relative
        batch, agents, modes = logits.shape
        selected = torch.arange(modes, device=logits.device).view(1, 1, modes)
        self.last_selected_indices = selected.expand(batch, agents, modes).detach()
        self.last_coarse_trajectories = (last + base_relative).detach()
        self.last_anchor_proposals = (
            past_loc[:, :, None, None, -1, :] + base_anchors
        ).detach()
        self.last_corrected_anchors = (
            past_loc[:, :, None, None, -1, :] + final_anchors
        ).detach()
        self._training_target = None
        return trajectories, logits


def replace_decoder(model, cfg, experiment: str) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown ICMAB experiment: {experiment}")
    checkpoint = Path(os.environ.get("ICMAB_SOURCE_CHECKPOINT", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    base = model.agent_aware_trans_encoder
    if not bool(getattr(base, "use_temporal_social_cross", False)):
        raise ValueError("IC1 requires F16 temporal-social cross attention")
    if isinstance(base.social_gate, SocialDeltaTapGate):
        raise RuntimeError("social gate already instrumented")
    children = list(base.social_gate.children())
    base.social_gate = SocialDeltaTapGate(*children, dim=int(base.dim))
    model.decoder = InteractionConditionedModeAnchorDecoder(
        model.token_dim, model.pred_len, social_dim=int(base.dim), rank=8
    )

    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    expected_missing = {
        key for key in target
        if key.startswith("decoder.interaction_bridge.")
        or key.startswith("decoder.mode_gate.")
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))
    skipped_source = [key for key in source if key not in compatible]
    if skipped_source:
        raise RuntimeError(f"Source parameters were skipped: {skipped_source}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.agent_aware_trans_encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in model.decoder.interaction_bridge.parameters():
        parameter.requires_grad_(True)
    for parameter in model.decoder.mode_gate.parameters():
        parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.cv_reference = False
    cfg.MODEL.DECODER.NUM_MODES = 20
