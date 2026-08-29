"""SDRE1 external encoder plugin for the locked F16 + mode/three-anchor decoder."""
from __future__ import annotations

import types
from typing import Optional

import torch
from torch import Tensor, nn


EXPERIMENTS = {"sdre1_scale_decoupled"}


def _zero_last(module: nn.Sequential) -> None:
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)


def _masked_local_distance(past: Tensor, num_valid: Optional[Tensor], fallback: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return final-step median/min neighbour distance and valid neighbour count."""
    batch, agents = past.shape[:2]
    distance = torch.linalg.vector_norm(past[:, :, None, -1] - past[:, None, :, -1], dim=-1)
    valid = torch.ones(batch, agents, dtype=torch.bool, device=past.device)
    if num_valid is not None:
        valid = torch.arange(agents, device=past.device)[None] < num_valid[:, None]
    neighbour = valid[:, :, None] & valid[:, None, :]
    eye = torch.eye(agents, dtype=torch.bool, device=past.device)[None]
    neighbour = neighbour & ~eye
    count = neighbour.sum(-1)
    ordered = distance.masked_fill(~neighbour, float("inf")).sort(dim=-1).values
    median_index = ((count - 1).clamp_min(0) // 2)[..., None]
    median = ordered.gather(-1, median_index).squeeze(-1)
    minimum = ordered[..., 0]
    default = fallback.to(device=past.device, dtype=past.dtype)
    median = torch.where(count > 0, median, default.expand_as(median))
    minimum = torch.where(count > 0, minimum, default.expand_as(minimum))
    return median.clamp_min(1.0e-4), minimum.clamp_min(1.0e-4), count.float()


def _motion_scale_and_condition(base, past: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Build explicit absolute scale/density condition plus scale-normalized motion shape."""
    velocity = past[:, :, 1:] - past[:, :, :-1]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    scale = speed.median(dim=-1).values.clamp_min(1.0e-3)
    mean_speed = speed.mean(-1)
    acceleration = velocity[:, :, 1:] - velocity[:, :, :-1]
    mean_acceleration = torch.linalg.vector_norm(acceleration, dim=-1).mean(-1)
    median_distance, minimum_distance, neighbour_count = _masked_local_distance(
        past, base._num_valid, base.sdre_local_distance_fallback,
    )
    condition = torch.stack((
        torch.log1p(scale), torch.log1p(mean_speed), torch.log1p(mean_acceleration),
        torch.log1p(neighbour_count), torch.log1p(median_distance), torch.log1p(minimum_distance),
    ), dim=-1)

    norm_velocity = velocity / scale[..., None, None]
    norm_acceleration = acceleration / scale[..., None, None]
    if velocity.shape[2] >= 2:
        first, second = velocity[:, :, :-1], velocity[:, :, 1:]
        cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
        dot = (first * second).sum(-1)
        turn = torch.atan2(cross, dot + 1.0e-6).abs().mean(-1)
    else:
        turn = torch.zeros_like(scale)
    speed_cv = speed.std(-1, unbiased=False) / mean_speed.clamp_min(1.0e-3)
    shape = torch.cat((
        norm_velocity.mean(2), norm_velocity[:, :, -1], norm_acceleration.mean(2),
        turn[..., None], speed_cv[..., None],
    ), dim=-1)
    return scale, condition, shape


def _relation_tokens_scale_decoupled(self, past: Tensor):
    features, last_distance, last_closing = self.relation_features(past)
    raw_encoded = self.relation_encoder(features)
    if self.use_multiscale_relation:
        raw_scales = torch.stack((raw_encoded[:, :, :, -2:].mean(3), raw_encoded[:, :, :, -4:].mean(3), raw_encoded.mean(3)), dim=3)
        raw_tokens = (torch.softmax(self.relation_scale_score(raw_scales), dim=3) * raw_scales).sum(3)
    else:
        raw_weights = torch.softmax(self.relation_time_score(raw_encoded), dim=3)
        raw_tokens = (raw_weights * raw_encoded).sum(dim=3)

    scale, condition, _shape = _motion_scale_and_condition(self, past)
    local_distance, _minimum_distance, _count = _masked_local_distance(
        past, self._num_valid, self.sdre_local_distance_fallback,
    )
    receiver_distance = local_distance[:, :, None, None, None]
    pair_speed = (scale[:, :, None] + scale[:, None, :]).clamp_min(1.0e-3)[..., None, None]
    normalized = torch.cat((
        features[..., 0:2] / receiver_distance,
        features[..., 2:4] / pair_speed,
        features[..., 4:5] / receiver_distance,
        features[..., 5:6] / pair_speed,
        features[..., 6:7],
        features[..., 7:8] / receiver_distance,
    ), dim=-1)
    normalized_encoded = self.sdre_normalized_relation_encoder(normalized)
    if self.use_multiscale_relation:
        normalized_scales = torch.stack((normalized_encoded[:, :, :, -2:].mean(3), normalized_encoded[:, :, :, -4:].mean(3), normalized_encoded.mean(3)), dim=3)
        normalized_tokens = (torch.softmax(self.sdre_normalized_time_score(normalized_scales), dim=3) * normalized_scales).sum(3)
    else:
        normalized_weights = torch.softmax(self.sdre_normalized_time_score(normalized_encoded), dim=3)
        normalized_tokens = (normalized_weights * normalized_encoded).sum(dim=3)
    relation_gate = torch.sigmoid(self.sdre_relation_gate(condition))[..., None]
    tokens = raw_tokens + relation_gate * normalized_tokens

    self.last_sdre_scale = scale.detach()
    self.last_sdre_condition = condition.detach()
    self.last_sdre_relation_gate = relation_gate.detach().squeeze(-1)
    self.last_sdre_normalized_relation_norm = normalized_tokens.detach().norm(dim=-1)
    final_ttc = features[:, :, :, -1, 6] * 10.0
    final_dca = features[:, :, :, -1, 7]
    return tokens, last_distance, last_closing, final_ttc + final_dca


class ScaleShapeTemporalQueries(nn.Module):
    """EMA-safe wrapper for F16 temporal queries plus a shape residual.

    A normal forward hook closes over its original parent encoder; deepcopy used
    by EMA would then read stale training-batch context.  This module owns its
    context and is populated by a parent pre-hook, so copied EMA models remain
    batch-correct.
    """
    def __init__(self, original: nn.Module, dim: int, hidden: int) -> None:
        super().__init__()
        self.original = original
        self.sdre_shape_adapter = nn.Sequential(
            nn.LayerNorm(8), nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )
        _zero_last(self.sdre_shape_adapter)
        self.sdre_shape_gate = nn.Sequential(
            nn.LayerNorm(6), nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 1),
        )
        nn.init.zeros_(self.sdre_shape_gate[-1].weight)
        nn.init.constant_(self.sdre_shape_gate[-1].bias, -2.0)
        self.register_buffer("sdre_local_distance_fallback", torch.tensor(1.0), persistent=True)
        self._past: Optional[Tensor] = None
        self._num_valid: Optional[Tensor] = None

    def set_context(self, past: Tensor, num_valid: Optional[Tensor]) -> None:
        self._past, self._num_valid = past, num_valid

    def forward(self, tokens: Tensor):
        temporal, agent = self.original(tokens)
        if self._past is None or self._past.shape[0] != agent.shape[0]:
            # This should never occur after the parent pre-hook, but retaining
            # baseline output is safer than applying a stale residual.
            return temporal, agent
        _scale, condition, shape = _motion_scale_and_condition(self, self._past)
        residual = self.sdre_shape_adapter(shape)
        gate = torch.sigmoid(self.sdre_shape_gate(condition))
        self.last_sdre_shape_gate = gate.detach().squeeze(-1)
        self.last_sdre_shape_residual_norm = residual.detach().norm(dim=-1)
        return temporal, agent + gate * residual


def _prepare_shape_context(module: nn.Module, _inputs) -> None:
    module.temporal_queries.set_context(module._past, module._num_valid)


def _validate_f16(base: nn.Module) -> None:
    if not bool(getattr(base, "use_temporal_social_cross", False)):
        raise ValueError("SDRE1 requires f16_temporal_social_cross")
    if not bool(getattr(base, "use_relation_tokens", False)):
        raise ValueError("SDRE1 requires F16 relation tokens")
    forbidden = ("use_pair_deformable_temporal", "use_edge_list_attention")
    active = [name for name in forbidden if bool(getattr(base, name, False))]
    if active:
        raise ValueError(f"SDRE1 expects plain locked F16; active flags: {active}")


def replace_encoder(model: nn.Module, cfg, experiment: str) -> None:
    """Attach SDRE1 without editing the shared F16 implementation."""
    del cfg
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown SDRE experiment: {experiment}")
    if getattr(model, "front_variant", None) != "f16_temporal_social_cross":
        raise ValueError("SDRE1 requires f16_temporal_social_cross")
    base = model.agent_aware_trans_encoder
    _validate_f16(base)
    dim = int(base.dim)
    hidden = max(16, dim // 4)
    base.sdre_normalized_relation_encoder = nn.Sequential(
        nn.LayerNorm(8), nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, dim),
    )
    _zero_last(base.sdre_normalized_relation_encoder)
    base.sdre_normalized_time_score = nn.Linear(dim, 1)
    nn.init.zeros_(base.sdre_normalized_time_score.weight)
    nn.init.zeros_(base.sdre_normalized_time_score.bias)
    base.sdre_relation_gate = nn.Sequential(
        nn.LayerNorm(6), nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 1),
    )
    nn.init.zeros_(base.sdre_relation_gate[-1].weight)
    nn.init.constant_(base.sdre_relation_gate[-1].bias, -2.0)
    base.register_buffer("sdre_local_distance_fallback", torch.tensor(1.0), persistent=True)
    base.relation_tokens = types.MethodType(_relation_tokens_scale_decoupled, base)
    base.temporal_queries = ScaleShapeTemporalQueries(base.temporal_queries, dim, hidden)
    base.register_forward_pre_hook(_prepare_shape_context)
