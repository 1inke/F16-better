"""Response-aware temporal-social attention on the locked plain F16 encoder."""

from __future__ import annotations

import math
import types
from typing import Optional

import torch
from torch import Tensor, nn


EXPERIMENTS = {"rb1_response_aware_bias"}


class ResponseAwareBias(nn.Module):
    """Per-head bias from pair-aware observed target responses."""

    def __init__(
        self, heads: int, frequency: float,
        distance_scale: float = 3.0, closing_temperature: float = 0.5,
        max_abs_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.frequency = float(frequency)
        self.distance_scale = float(distance_scale)
        self.closing_temperature = float(closing_temperature)
        self.max_abs_bias = float(max_abs_bias)
        self.projection = nn.Linear(4, self.heads)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.last_mean_abs_bias: Optional[Tensor] = None
        self.last_max_abs_bias: Optional[Tensor] = None
        self.last_opportunity_mean: Optional[Tensor] = None
        self.last_response_mean: Optional[Tensor] = None

    def response_features(self, past: Tensor, raw_relation: Tensor) -> Tensor:
        # past [B,N,T,2], raw_relation [B,N,N,T,8].
        position = past.permute(0, 2, 1, 3)
        velocity = torch.zeros_like(position)
        velocity[:, 1:] = (position[:, 1:] - position[:, :-1]) * self.frequency
        velocity[:, 0] = velocity[:, 1]
        velocity_change = torch.zeros_like(velocity)
        velocity_change[:, 1:] = velocity[:, 1:] - velocity[:, :-1]

        target_change = velocity_change.permute(0, 2, 1, 3)[:, :, None]
        relative = raw_relation[..., :2]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        direction = relative / distance.clamp_min(1.0e-4)
        perpendicular = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
        longitudinal = -(target_change * direction).sum(dim=-1)
        lateral = (target_change * perpendicular).sum(dim=-1)

        speed = torch.linalg.vector_norm(velocity, dim=-1)
        speed_change = torch.zeros_like(speed)
        speed_change[:, 1:] = speed[:, 1:] - speed[:, :-1]
        speed_change = speed_change.permute(0, 2, 1)[:, :, None].expand_as(longitudinal)

        heading = velocity / speed[..., None].clamp_min(1.0e-4)
        cross = heading[:, :-1, :, 0] * heading[:, 1:, :, 1] - heading[:, :-1, :, 1] * heading[:, 1:, :, 0]
        dot = (heading[:, :-1] * heading[:, 1:]).sum(dim=-1).clamp(-1.0, 1.0)
        heading_change = torch.zeros_like(speed)
        heading_change[:, 1:] = torch.atan2(cross, dot).abs()
        heading_change = heading_change.permute(0, 2, 1)[:, :, None].expand_as(longitudinal)

        # Fixed normalization prevents a few noisy finite differences from
        # dominating the tiny zero-initialized linear projection.
        return torch.stack((
            torch.tanh(longitudinal / 1.0),
            torch.tanh(lateral / 1.0),
            torch.tanh(speed_change / 1.0),
            (heading_change / math.pi).clamp(0.0, 1.0),
        ), dim=-1)

    def forward(self, past: Tensor, raw_relation: Tensor) -> Tensor:
        response = self.response_features(past, raw_relation)
        distance = raw_relation[..., 4]
        closing = raw_relation[..., 5]
        opportunity = torch.exp(-distance / self.distance_scale) * torch.sigmoid(
            closing / self.closing_temperature
        )
        projected = self.max_abs_bias * torch.tanh(self.projection(response))
        bias = opportunity[..., None] * projected
        agents = bias.shape[1]
        eye = torch.eye(agents, dtype=torch.bool, device=bias.device).view(
            1, agents, agents, 1, 1
        )
        bias = bias.masked_fill(eye, 0.0).permute(0, 4, 1, 2, 3)
        self.last_mean_abs_bias = bias.detach().abs().mean()
        self.last_max_abs_bias = bias.detach().abs().amax()
        self.last_opportunity_mean = opportunity.detach().mean()
        self.last_response_mean = response.detach().abs().mean()
        return bias


def _relation_tokens_with_response_tap(self, past: Tensor):
    features, last_distance, last_closing = self.relation_features(past)
    self.last_raw_relation_features = features
    encoded = self.relation_encoder(features)
    if self.use_multiscale_relation:
        recent2 = encoded[:, :, :, -2:].mean(dim=3)
        recent4 = encoded[:, :, :, -4:].mean(dim=3)
        full = encoded.mean(dim=3)
        scales = torch.stack((recent2, recent4, full), dim=3)
        scale_weights = torch.softmax(self.relation_scale_score(scales), dim=3)
        tokens = (scale_weights * scales).sum(dim=3)
    else:
        weights = torch.softmax(self.relation_time_score(encoded), dim=3)
        tokens = (weights * encoded).sum(dim=3)
    final_ttc = features[:, :, :, -1, 6] * 10.0
    final_dca = features[:, :, :, -1, 7]
    return tokens, last_distance, last_closing, final_ttc + final_dca


def _temporal_social_attention_with_response(
    self, temporal: Tensor, agent: Tensor, relation: Tensor, allowed: Tensor,
    type_features=None, risk_score=None,
) -> Tensor:
    del type_features, risk_score
    batch, agents, steps, _ = temporal.shape
    if self.last_raw_relation_features is None or self._past is None:
        raise RuntimeError("Response bias requires trajectory and relation context")
    q = self.q_proj(agent).reshape(
        batch, agents, self.heads, self.qk_head_dim
    ).permute(0, 2, 1, 3)
    k = self.k_proj(temporal).reshape(
        batch, agents, steps, self.kv_heads, self.qk_head_dim
    ).permute(0, 3, 1, 2, 4)
    v = self.v_proj(temporal).reshape(
        batch, agents, steps, self.kv_heads, self.head_dim
    ).permute(0, 3, 1, 2, 4)
    if self.kv_heads != self.heads:
        repeats = self.heads // self.kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    scores = torch.einsum("bhid,bhjtd->bhijt", q, k)
    scores = scores / math.sqrt(float(self.qk_head_dim))
    pair_bias = self.relation_bias(relation).permute(0, 3, 1, 2)
    scores = scores + pair_bias[..., None]
    scores = scores + self.response_aware_bias(
        self._past, self.last_raw_relation_features
    )
    expanded_mask = allowed[:, None, :, :, None].expand_as(scores)
    scores = scores.masked_fill(~expanded_mask, -1.0e4)
    weights = torch.softmax(scores.flatten(-2), dim=-1).reshape_as(scores)
    base_message = v[:, :, None].expand(-1, -1, agents, -1, -1, -1)
    edge_value = self.relation_value(relation).reshape(
        batch, agents, agents, self.heads, self.head_dim
    ).permute(0, 3, 1, 2, 4)
    message = base_message + edge_value[..., None, :]
    social = (weights[..., None] * message).sum(dim=(3, 4))
    social = social.permute(0, 2, 1, 3).reshape(batch, agents, self.dim)
    return self.out_proj(social)


def _validate_plain_f16(base: nn.Module) -> None:
    if not bool(getattr(base, "use_temporal_social_cross", False)):
        raise ValueError("RAIB requires F16 temporal-social cross attention")
    if not bool(getattr(base, "use_relation_tokens", False)):
        raise ValueError("RAIB requires the original F16 relation path")
    incompatible = (
        "use_pair_deformable_temporal", "use_pair_temporal_soft_prior",
        "use_scaled_cosine_attention", "use_differential_attention",
        "use_relation_type_bias", "use_soft_risk_bias", "use_sparsemax",
        "use_entmax15", "use_temporal_relative_bias",
    )
    active = [name for name in incompatible if bool(getattr(base, name, False))]
    if active:
        raise ValueError(f"RAIB expects locked plain F16; active={active}")


def replace_encoder(model: nn.Module, cfg, experiment: str) -> None:
    del cfg
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown response-aware experiment: {experiment}")
    if getattr(model, "front_variant", None) != "f16_temporal_social_cross":
        raise ValueError("RAIB requires f16_temporal_social_cross")
    base = model.agent_aware_trans_encoder
    _validate_plain_f16(base)
    base.response_aware_bias = ResponseAwareBias(
        heads=int(base.heads), frequency=float(base.frequency)
    )
    base.last_raw_relation_features = None
    base.relation_tokens = types.MethodType(_relation_tokens_with_response_tap, base)
    base.temporal_social_attention = types.MethodType(
        _temporal_social_attention_with_response, base
    )
