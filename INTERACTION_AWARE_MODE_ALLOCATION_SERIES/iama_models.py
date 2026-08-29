"""MA1: social-conditioned Top-5 mode matching without post-hoc correction."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SOURCE_PATH = WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"
IR1_PATH = WORKSPACE / "INTERACTION_RESIDUAL_SUPERVISION_SERIES" / "irs_models.py"
EXPERIMENTS = {"ma1_interaction_aware_top5"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load("iama_grouped_source", SOURCE_PATH)
IR1 = _load("iama_social_tap", IR1_PATH)


class SocialCompatibility(nn.Module):
    def __init__(self, social_dim: int, mode_dim: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(social_dim + mode_dim),
            nn.Linear(social_dim + mode_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Exact source parity at initialization and no random target corruption.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, social: Tensor, mode: Tensor) -> Tensor:
        expanded = social[:, :, None].expand(-1, -1, mode.shape[2], -1)
        return self.net(torch.cat((mode, expanded), dim=-1)).squeeze(-1)


class InteractionAwareDecoder(SOURCE.GroupedModeTemplateDecoder):
    """Unchanged candidate generator plus a training-only compatibility head."""

    def __init__(self, token_dim: int, pred_len: int, social_dim: int) -> None:
        super().__init__(token_dim, pred_len, use_template=True, use_groups=False)
        self.compatibility = SocialCompatibility(social_dim, int(self.mode_embedding.shape[-1]))
        self.last_compatibility_logits: Tensor | None = None
        self.last_allocation_target: Tensor | None = None
        self.last_geometry_target: Tensor | None = None
        self.last_allocation_kl: Tensor | None = None
        self.last_compatibility_ce: Tensor | None = None
        self.cv_reference = False

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.train(True)
        self.eval()
        self.compatibility.train(True)

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        social = IR1._SOCIAL_DELTA.get()
        if social is None:
            raise RuntimeError("MA1 did not receive the F16 social-specific representation")
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context = self.context(torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1))
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        scale = F.softplus(self.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(*mode.shape[:-1], 3, 2)
        source_anchors = scale[:, :, None, None] * (self.anchor_templates[None, None] + delta)
        base_anchors = source_anchors + self.refiner(mode, source_anchors)
        base_relative = torch.einsum("tp,bakpc->baktc", self.interpolation, base_anchors)
        focal_relative, focal_anchors = self.template_head(
            context[:, :1], mode[:, :1], base_anchors[:, :1], self.interpolation, None
        )
        relative = torch.cat((focal_relative, base_relative[:, 1:]), dim=1)
        anchors = torch.cat((focal_anchors, base_anchors[:, 1:]), dim=1)
        logits = self.score_head(torch.cat((mode, source_anchors.flatten(-2)), dim=-1)).squeeze(-1)
        self.last_compatibility_logits = self.compatibility(social, mode)
        last = past_loc[:, :, None, -1:, :]
        trajectories = last + relative
        modes = logits.shape[-1]
        selected = torch.arange(modes, device=logits.device).view(1, 1, modes)
        self.last_selected_indices = selected.expand_as(logits).detach()
        self.last_coarse_trajectories = (last + base_relative).detach()
        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + base_anchors).detach()
        self.last_corrected_anchors = (past_loc[:, :, None, None, -1, :] + anchors).detach()
        self._training_target = None
        return trajectories, logits


class InteractionAwareModeAllocationLoss(nn.Module):
    """Top-5 soft matching with a separately supervised social compatibility prior."""

    def __init__(self, cfg, decoder: InteractionAwareDecoder) -> None:
        super().__init__()
        d = cfg.MODEL.DECODER
        self.decoder = decoder
        self.winner_fde_weight = float(d.WINNER_FDE_WEIGHT)
        self.classification_weight = float(d.CLASSIFICATION_WEIGHT)
        self.endpoint_weight = float(d.ENDPOINT_LOSS_WEIGHT)
        self.hard_classification_weight = float(d.HARD_CLASSIFICATION_WEIGHT)
        self.diversity_weight = float(d.DIVERSITY_WEIGHT)
        self.diversity_margin = float(d.DIVERSITY_MARGIN)
        self.short_horizon_weight = float(d.SHORT_HORIZON_WEIGHT)
        self.short_horizon_steps = int(d.SHORT_HORIZON_STEPS)
        self.temperature = float(d.SOFT_TEMPERATURE)
        self.top_m = 5
        self.anchor_weight = 0.25
        self.social_strength = 0.25
        self.compatibility_weight = 0.10
        pred_len = int(round(cfg.PREDICTION.PRED_TIME * cfg.DATA.FREQUENCY))
        start, end = float(cfg.LOSS.START_WEIGHT), float(cfg.LOSS.END_WEIGHT)
        if cfg.LOSS.WEIGHTED_PENALTY == "exponential":
            weights = torch.exp(torch.linspace(torch.log(torch.tensor(start)), torch.log(torch.tensor(end)), pred_len))
        elif cfg.LOSS.WEIGHTED_PENALTY == "linear":
            weights = torch.linspace(start, end, pred_len)
        elif cfg.LOSS.WEIGHTED_PENALTY == "quadratic":
            weights = torch.linspace(start, end, pred_len).square()
        else:
            weights = torch.ones(pred_len)
        self.register_buffer("weights", weights)

    @staticmethod
    def _weighted(values: Tensor, sample_weight: Tensor | None) -> Tensor:
        per_sample = values.reshape(values.shape[0], -1).mean(-1)
        if sample_weight is None:
            return per_sample.mean()
        weight = sample_weight.to(values).reshape(values.shape[0])
        weight = weight / weight.mean().clamp_min(1e-8)
        return (per_sample * weight).mean()

    def forward(self, predictions, target, logits, cfg=None, sample_weight=None):
        del cfg
        if self.decoder.last_compatibility_logits is None:
            raise RuntimeError("MA1 compatibility state missing")
        distance = torch.linalg.vector_norm(predictions - target.unsqueeze(-3), dim=-1)
        trajectory_cost = distance.mean(-1) + self.winner_fde_weight * distance[..., -1]
        anchor_cost = distance[..., [3, 7, 11]].mean(-1)
        geometry_cost = trajectory_cost + self.anchor_weight * anchor_cost
        gap = geometry_cost.detach() - geometry_cost.detach().min(-1, keepdim=True).values
        scale = gap.mean(-1, keepdim=True).clamp_min(0.05)
        geometry_logits = -gap / (self.temperature * scale)
        geometry_target = torch.softmax(geometry_logits, dim=-1)

        compatibility = self.decoder.last_compatibility_logits[:, : predictions.shape[1]]
        compatibility_centered = compatibility - compatibility.mean(-1, keepdim=True)
        compatibility_scaled = compatibility_centered / compatibility_centered.detach().std(-1, keepdim=True).clamp_min(1.0)
        assignment_logits = geometry_logits + self.social_strength * compatibility_scaled.detach()
        top = assignment_logits.topk(self.top_m, dim=-1).indices
        mask = torch.zeros_like(assignment_logits, dtype=torch.bool).scatter_(-1, top, True)
        target_responsibility = torch.softmax(assignment_logits.masked_fill(~mask, -1e4), dim=-1).detach()

        regression_values = F.smooth_l1_loss(
            predictions, target.unsqueeze(-3), reduction="none"
        ).sum(-1) * self.weights
        regression_per_mode = regression_values.mean(-1)
        regression = self._weighted((target_responsibility * regression_per_mode).sum(-1), sample_weight)
        endpoint_per_mode = F.smooth_l1_loss(
            predictions[..., -1, :], target.unsqueeze(-3)[..., -1, :], reduction="none"
        ).mean(-1)
        endpoint = self._weighted((target_responsibility * endpoint_per_mode).sum(-1), sample_weight)
        short = min(max(self.short_horizon_steps, 1), predictions.shape[-2])
        short_per_mode = F.smooth_l1_loss(
            predictions[..., :short, :], target.unsqueeze(-3)[..., :short, :], reduction="none"
        ).mean((-1, -2))
        short_horizon = self._weighted((target_responsibility * short_per_mode).sum(-1), sample_weight)

        allocation_values = -(target_responsibility * torch.log_softmax(logits, dim=-1)).sum(-1)
        hard_target = target_responsibility.argmax(-1)
        hard_values = F.cross_entropy(logits.flatten(0, -2), hard_target.flatten(), reduction="none").reshape_as(hard_target)
        allocation = self._weighted(allocation_values + self.hard_classification_weight * hard_values, sample_weight)
        compatibility_values = -(geometry_target * torch.log_softmax(compatibility, dim=-1)).sum(-1)
        compatibility_loss = self._weighted(compatibility_values, sample_weight)

        endpoints = predictions[..., -1, :]
        pairwise = torch.cdist(endpoints, endpoints)
        upper = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
        diversity_values = F.relu(self.diversity_margin - pairwise) * upper
        pair_count = upper.reshape(upper.shape[0], -1).sum(-1).clamp_min(1)
        diversity = self._weighted(diversity_values.reshape(upper.shape[0], -1).sum(-1) / pair_count, sample_weight)
        total = (
            regression + self.classification_weight * allocation
            + self.endpoint_weight * endpoint + self.short_horizon_weight * short_horizon
            + self.diversity_weight * diversity + self.compatibility_weight * compatibility_loss
        )
        self.decoder.last_allocation_target = target_responsibility.detach()
        self.decoder.last_geometry_target = geometry_target.detach()
        self.decoder.last_allocation_kl = (
            target_responsibility * (target_responsibility.clamp_min(1e-12).log() - torch.log_softmax(logits, -1))
        ).sum(-1).mean().detach()
        self.decoder.last_compatibility_ce = compatibility_loss.detach()
        winner = geometry_cost.detach().argmin(-1)
        return total, {
            "regression": regression.detach(), "classification": allocation.detach(),
            "endpoint": endpoint.detach(), "diversity": diversity.detach(),
            "short_horizon": short_horizon.detach(), "winner": winner,
        }


def build_loss(cfg, model) -> nn.Module:
    if not isinstance(model.decoder, InteractionAwareDecoder):
        raise TypeError(type(model.decoder))
    return InteractionAwareModeAllocationLoss(cfg, model.decoder)


def replace_decoder(model, cfg, experiment: str) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(experiment)
    checkpoint = Path(os.environ.get("IAMA_SOURCE_CHECKPOINT", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    base = model.agent_aware_trans_encoder
    if not bool(getattr(base, "use_temporal_social_cross", False)):
        raise ValueError("MA1 requires F16 temporal-social cross attention")
    base.social_gate = IR1.SocialDeltaTapGate(*list(base.social_gate.children()), dim=int(base.dim))
    base.register_forward_pre_hook(IR1.EncoderContextTap())
    model.decoder = InteractionAwareDecoder(model.token_dim, model.pred_len, int(base.dim))
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    target = model.state_dict()
    compatible = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
    incompatible = model.load_state_dict(compatible, strict=False)
    expected = {k for k in target if k.startswith("decoder.compatibility.")}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))
    skipped = [k for k in source if k not in compatible]
    if skipped:
        raise RuntimeError(f"Skipped source keys: {skipped}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.agent_aware_trans_encoder.parameters():
        parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if name.startswith("decoder.compatibility."):
            parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint)
    cfg.MODEL.DECODER.NUM_MODES = 20
