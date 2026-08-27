"""Lightweight non-decoder plugins for motion-scale generalization.

The F16 sparse Top-K graph, temporal-social attention, K20 decoder and losses
remain locked.  Only the robust state representation or relation-logit bias is
changed by these experiments.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


EXPERIMENTS = {
    "g1_shape_scale_disentangled",
    "g3_scale_conditioned_relation_bias",
}


def motion_terms(past: Tensor, frequency: float) -> tuple[Tensor, ...]:
    velocity = torch.zeros_like(past)
    velocity[:, :, 1:] = (past[:, :, 1:] - past[:, :, :-1]) * frequency
    velocity[:, :, 0] = velocity[:, :, 1]
    lag = min(3, past.shape[2] - 1)
    long_velocity = torch.zeros_like(past)
    long_velocity[:, :, lag:] = (
        past[:, :, lag:] - past[:, :, :-lag]
    ) * (frequency / float(lag))
    long_velocity[:, :, :lag] = long_velocity[:, :, lag : lag + 1]
    acceleration = (velocity - long_velocity) * frequency
    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    robust_scale = speed.median(dim=2, keepdim=True).values.clamp_min(0.05)
    confidence = torch.sigmoid((speed - 0.05) / 0.05)
    heading = velocity / speed.clamp_min(1.0e-5) * confidence
    return velocity, long_velocity, acceleration, speed, robust_scale, heading, confidence


class ShapeScaleDisentangledState(nn.Module):
    """Encode scale-free motion shape and physical scale in separate branches."""

    shape_dim = 10
    scale_dim = 5

    def __init__(self, token_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        shape_hidden = max(32, hidden_dim // 2)
        scale_hidden = max(24, hidden_dim // 3)
        self.shape_encoder = nn.Sequential(
            nn.LayerNorm(self.shape_dim),
            nn.Linear(self.shape_dim, shape_hidden),
            nn.GELU(),
            nn.Linear(shape_hidden, token_dim),
        )
        self.scale_encoder = nn.Sequential(
            nn.LayerNorm(self.scale_dim),
            nn.Linear(self.scale_dim, scale_hidden),
            nn.GELU(),
            nn.Linear(scale_hidden, token_dim),
        )
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(self.scale_dim),
            nn.Linear(self.scale_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.constant_(self.scale_gate[-1].bias, -1.0)

    @staticmethod
    def features(past: Tensor, frequency: float) -> tuple[Tensor, Tensor]:
        velocity, long_velocity, acceleration, speed, scale, heading, confidence = (
            motion_terms(past, frequency)
        )
        previous_heading = torch.cat((heading[:, :, :1], heading[:, :, :-1]), dim=2)
        cross = previous_heading[..., 0] * heading[..., 1] - previous_heading[..., 1] * heading[..., 0]
        dot = (previous_heading * heading).sum(dim=-1).clamp(-1.0, 1.0)
        turn = torch.atan2(cross, dot).unsqueeze(-1) / math.pi
        normalized_speed = speed / scale
        previous_speed = torch.cat((speed[:, :, :1], speed[:, :, :-1]), dim=2)
        speed_trend = (speed - previous_speed) / scale
        acceleration_size = torch.linalg.vector_norm(
            acceleration, dim=-1, keepdim=True
        )
        shape = torch.cat(
            (
                velocity / scale,
                long_velocity / scale,
                acceleration / (scale * frequency),
                heading,
                turn * confidence,
                confidence,
            ),
            dim=-1,
        )
        scale_features = torch.cat(
            (
                torch.log1p(speed),
                normalized_speed,
                torch.log1p(scale).expand_as(speed),
                torch.log1p(acceleration_size),
                speed_trend,
            ),
            dim=-1,
        )
        return shape, scale_features

    def forward(self, past: Tensor, frequency: float) -> Tensor:
        shape, scale = self.features(past, frequency)
        shape_token = torch.tanh(self.shape_encoder(shape))
        scale_token = torch.tanh(self.scale_encoder(scale))
        return shape_token + torch.sigmoid(self.scale_gate(scale)) * scale_token


class ScaleConditionedRelationBias(nn.Module):
    """Add a bounded motion-scale correction only to sparse attention logits."""

    def __init__(self, original: nn.Module, heads: int, frequency: float) -> None:
        super().__init__()
        self.original = original
        self.heads = int(heads)
        self.frequency = float(frequency)
        self.condition = nn.Sequential(
            nn.LayerNorm(4), nn.Linear(4, 16), nn.GELU(), nn.Linear(16, heads)
        )
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        self._pair_features: Optional[Tensor] = None
        self.last_bias_correction: Optional[Tensor] = None

    def set_context(self, past: Tensor) -> None:
        velocity = (past[:, :, 1:] - past[:, :, :-1]) * self.frequency
        recent = velocity[:, :, -min(3, velocity.shape[2]):]
        speed = torch.linalg.vector_norm(recent, dim=-1).median(dim=-1).values
        receiver = speed[:, :, None]
        sender = speed[:, None, :]
        distance = torch.linalg.vector_norm(
            past[:, :, -1, None, :] - past[:, None, :, -1, :], dim=-1
        )
        pair_scale = 0.25 + 0.5 * (receiver + sender)
        self._pair_features = torch.stack(
            (
                torch.log1p(receiver.expand_as(distance)),
                torch.log1p(sender.expand_as(distance)),
                torch.log1p((receiver - sender).abs().expand_as(distance)),
                torch.log1p(distance / pair_scale),
            ),
            dim=-1,
        )

    def forward(self, relation: Tensor) -> Tensor:
        base = self.original(relation)
        if self._pair_features is None:
            raise RuntimeError("Scale-conditioned bias requires trajectory context")
        correction = 0.5 * torch.tanh(self.condition(self._pair_features))
        self.last_bias_correction = correction.detach()
        return base + correction


class RelationBiasContextWrapper(nn.Module):
    def __init__(self, base: nn.Module, frequency: float) -> None:
        super().__init__()
        self.base = base
        conditioned = ScaleConditionedRelationBias(
            base.relation_bias, base.heads, frequency
        )
        self.base.relation_bias = conditioned

    def set_context(self, past: Tensor, num_valid: Optional[Tensor]) -> None:
        self.base.set_context(past, num_valid)
        self.base.relation_bias.set_context(past)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.base(src, mask=mask)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


def replace_encoder(model: nn.Module, cfg, experiment: str) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown scale-generalization experiment: {experiment}")
    if getattr(model, "front_variant", None) != "f16_temporal_social_cross":
        raise ValueError("These plugins require the locked F16 encoder")
    if experiment == "g1_shape_scale_disentangled":
        model.robust_state = ShapeScaleDisentangledState(model.token_dim, hidden_dim=64)
    else:
        model.agent_aware_trans_encoder = RelationBiasContextWrapper(
            model.agent_aware_trans_encoder, float(cfg.DATA.FREQUENCY)
        )
