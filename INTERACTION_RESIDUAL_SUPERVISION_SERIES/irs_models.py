"""IR1: future-response supervised social token with a semantic mode bridge."""

from __future__ import annotations

import contextvars
import importlib.util
import os
from pathlib import Path

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT.parent / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"
EXPERIMENTS = {"ir1_future_response_supervision"}
_SOCIAL_DELTA: contextvars.ContextVar[Tensor | None] = contextvars.ContextVar(
    "irs_social_delta", default=None
)
_NUM_VALID: contextvars.ContextVar[Tensor | None] = contextvars.ContextVar(
    "irs_num_valid", default=None
)


def _load_source():
    spec = importlib.util.spec_from_file_location("irs_exp146_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source()


class EncoderContextTap:
    def __call__(self, module: nn.Module, inputs) -> None:
        del inputs
        _NUM_VALID.set(getattr(module, "_num_valid", None))


class SocialDeltaTapGate(nn.Sequential):
    def __init__(self, *modules: nn.Module, dim: int) -> None:
        super().__init__(*modules)
        self.dim = int(dim)

    def forward(self, features: Tensor) -> Tensor:
        logits = super().forward(features)
        _SOCIAL_DELTA.set(torch.sigmoid(logits) * features[..., self.dim:])
        return logits


class FutureResponseHead(nn.Module):
    def __init__(self, social_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(social_dim)
        self.down = nn.Linear(social_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, 6)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, social: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.activation(self.down(self.norm(social)))
        response = self.output(hidden).reshape(*social.shape[:-1], 3, 2)
        return hidden, response


class FutureResponseSupervisedDecoder(SOURCE.GroupedModeTemplateDecoder):
    """Frozen EXP-146B read through a supervised six-dimensional bottleneck."""

    def __init__(
        self, token_dim: int, pred_len: int, social_dim: int,
        frequency: float, response_hidden: int = 16,
        auxiliary_weight: float = 0.10,
    ) -> None:
        super().__init__(token_dim, pred_len, use_template=True, use_groups=False)
        mode_dim = int(self.mode_embedding.shape[-1])
        self.social_dim = int(social_dim)
        self.mode_dim = mode_dim
        self.frequency = float(frequency)
        self.auxiliary_weight = float(auxiliary_weight)
        self.response_head = FutureResponseHead(self.social_dim, response_hidden)
        self.response_to_mode = nn.Linear(6, mode_dim)
        nn.init.zeros_(self.response_to_mode.weight)
        nn.init.zeros_(self.response_to_mode.bias)
        self.mode_gate = nn.Linear(mode_dim + response_hidden, 1)
        nn.init.zeros_(self.mode_gate.weight)
        nn.init.constant_(self.mode_gate.bias, -2.0)
        self.last_predicted_response: Tensor | None = None
        self.last_response_hidden: Tensor | None = None
        self.last_interaction_weight: Tensor | None = None
        self.last_inertia_anchors: Tensor | None = None
        self.last_mode_gates: Tensor | None = None
        self.last_semantic_mode_delta: Tensor | None = None
        self.last_auxiliary_unweighted: Tensor | None = None
        self.last_auxiliary_weighted: Tensor | None = None

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.train(True)
        self.eval()
        self.response_head.train(True)
        self.response_to_mode.train(True)
        self.mode_gate.train(True)

    def _interaction_strength(self, past_loc: Tensor) -> Tensor:
        batch, agents, _, _ = past_loc.shape
        num_valid = _NUM_VALID.get()
        if num_valid is None:
            num_valid = torch.full(
                (batch,), agents, dtype=torch.long, device=past_loc.device
            )
        else:
            num_valid = num_valid.to(device=past_loc.device).long()
        if agents <= 1:
            return past_loc.new_zeros(batch, agents)
        frequency = self.frequency
        last = past_loc[:, :, -1]
        velocity = (past_loc[:, :, -1] - past_loc[:, :, -2]) * frequency
        relative = last[:, None] - last[:, :, None]
        relative_velocity = velocity[:, None] - velocity[:, :, None]
        distance = torch.linalg.vector_norm(relative, dim=-1)
        dot = (relative * relative_velocity).sum(dim=-1)
        speed_sq = relative_velocity.square().sum(dim=-1)
        closing = -dot / distance.clamp_min(1.0e-3)
        raw_ttc = -dot / speed_sq.clamp_min(1.0e-4)
        ttc = raw_ttc.clamp(0.0, 10.0)
        dca = torch.linalg.vector_norm(
            relative + ttc[..., None] * relative_velocity, dim=-1
        )
        opportunity = torch.exp(-distance / 3.0) * torch.sigmoid(closing / 0.5)
        conflict = (
            torch.exp(-ttc / 3.0) * torch.exp(-dca / 1.5)
            * (raw_ttc > 0.0).to(past_loc.dtype)
        )
        pair_strength = 0.5 * (opportunity + conflict)
        ids = torch.arange(agents, device=past_loc.device)
        valid = ids[None] < num_valid[:, None]
        pair_valid = valid[:, :, None] & valid[:, None, :]
        pair_valid = pair_valid & ~torch.eye(
            agents, dtype=torch.bool, device=past_loc.device
        )[None]
        pair_strength = pair_strength.masked_fill(~pair_valid, 0.0)
        return pair_strength.amax(dim=-1).clamp(0.0, 1.0)

    def _condition_modes(
        self, base_mode: Tensor, past_loc: Tensor
    ) -> Tensor:
        social = _SOCIAL_DELTA.get()
        if social is None or social.shape[-1] != self.social_dim:
            raise RuntimeError("IR1 decoder did not receive the F16 social delta")
        hidden, response = self.response_head(social)
        semantic = self.response_to_mode(response.flatten(-2))
        gate_input = torch.cat(
            (
                base_mode,
                hidden[:, :, None].expand(-1, -1, base_mode.shape[2], -1),
            ),
            dim=-1,
        )
        gates = torch.sigmoid(self.mode_gate(gate_input)).squeeze(-1)
        delta = gates[..., None] * semantic[:, :, None]
        last = past_loc[:, :, -1]
        last_step = past_loc[:, :, -1] - past_loc[:, :, -2]
        steps = past_loc.new_tensor((4.0, 8.0, 12.0)).view(1, 1, 3, 1)
        inertia = last[:, :, None] + steps * last_step[:, :, None]
        self.last_predicted_response = response
        self.last_response_hidden = hidden.detach()
        self.last_interaction_weight = self._interaction_strength(past_loc).detach()
        self.last_inertia_anchors = inertia.detach()
        self.last_mode_gates = gates.detach()
        self.last_semantic_mode_delta = delta.detach()
        return base_mode + delta

    def auxiliary_training_loss(
        self, prediction: Tensor, target: Tensor, logits: Tensor
    ) -> Tensor:
        del prediction, logits
        if self.last_predicted_response is None or self.last_inertia_anchors is None:
            return target.new_zeros(())
        target_anchors = target[:, 0, [3, 7, 11]]
        response_target = target_anchors - self.last_inertia_anchors[:, 0]
        response_prediction = self.last_predicted_response[:, 0]
        per_sample = (response_prediction - response_target).abs().mean(dim=(-1, -2))
        weight = self.last_interaction_weight[:, 0]
        unweighted = per_sample.mean()
        weighted = (weight * per_sample).mean()
        self.last_auxiliary_unweighted = unweighted.detach()
        self.last_auxiliary_weighted = weighted.detach()
        return self.auxiliary_weight * weighted

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context = self.context(
            torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1)
        )
        base_mode = self.mode_mlp(
            context[:, :, None] + self.mode_embedding[None, None]
        )
        mode = self._condition_modes(base_mode, past_loc)
        scale = torch.nn.functional.softplus(self.proposal_scale(context)) + 0.25
        proposal_delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(
            *mode.shape[:-1], 3, 2
        )
        source_anchors = scale[:, :, None, None] * (
            self.anchor_templates[None, None] + proposal_delta
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
        raise ValueError(experiment)
    checkpoint = Path(os.environ.get("IRS_SOURCE_CHECKPOINT", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    base = model.agent_aware_trans_encoder
    if not bool(getattr(base, "use_temporal_social_cross", False)):
        raise ValueError("IR1 requires F16 temporal-social cross attention")
    base.social_gate = SocialDeltaTapGate(
        *list(base.social_gate.children()), dim=int(base.dim)
    )
    base.register_forward_pre_hook(EncoderContextTap())
    model.decoder = FutureResponseSupervisedDecoder(
        model.token_dim, model.pred_len, social_dim=int(base.dim),
        frequency=float(cfg.DATA.FREQUENCY), response_hidden=16,
        auxiliary_weight=0.10,
    )
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    prefixes = (
        "decoder.response_head.", "decoder.response_to_mode.", "decoder.mode_gate."
    )
    expected_missing = {key for key in target if key.startswith(prefixes)}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))
    skipped = [key for key in source if key not in compatible]
    if skipped:
        raise RuntimeError(f"Skipped source keys: {skipped}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.agent_aware_trans_encoder.parameters():
        parameter.requires_grad_(True)
    for prefix in prefixes:
        for name, parameter in model.named_parameters():
            if name.startswith(prefix):
                parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.cv_reference = False
    cfg.MODEL.DECODER.NUM_MODES = 20
