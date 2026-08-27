"""Residual non-decoder plugin for EXP-138A."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


EXPERIMENTS = {"t1_residual_shape_scale"}


class ShapeScaleResidual(nn.Module):
    """Zero-initialized scale/shape correction that preserves baseline state."""

    shape_dim = 10
    scale_dim = 5

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.shape = nn.Sequential(
            nn.LayerNorm(self.shape_dim), nn.Linear(self.shape_dim, 32),
            nn.GELU(), nn.Linear(32, token_dim),
        )
        self.scale = nn.Sequential(
            nn.LayerNorm(self.scale_dim), nn.Linear(self.scale_dim, 24),
            nn.GELU(), nn.Linear(24, token_dim),
        )
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(self.scale_dim), nn.Linear(self.scale_dim, 16),
            nn.GELU(), nn.Linear(16, 1),
        )
        nn.init.zeros_(self.shape[-1].weight)
        nn.init.zeros_(self.shape[-1].bias)
        nn.init.zeros_(self.scale[-1].weight)
        nn.init.zeros_(self.scale[-1].bias)
        nn.init.constant_(self.scale_gate[-1].bias, -1.0)

    @staticmethod
    def features(past: Tensor, frequency: float) -> tuple[Tensor, Tensor]:
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
        previous_heading = torch.cat((heading[:, :, :1], heading[:, :, :-1]), dim=2)
        cross = previous_heading[..., 0] * heading[..., 1] - previous_heading[..., 1] * heading[..., 0]
        dot = (previous_heading * heading).sum(dim=-1).clamp(-1.0, 1.0)
        turn = torch.atan2(cross, dot).unsqueeze(-1) / math.pi
        previous_speed = torch.cat((speed[:, :, :1], speed[:, :, :-1]), dim=2)
        shape = torch.cat(
            (
                velocity / robust_scale,
                long_velocity / robust_scale,
                acceleration / (robust_scale * frequency),
                heading,
                turn * confidence,
                confidence,
            ), dim=-1,
        )
        scale = torch.cat(
            (
                torch.log1p(speed), speed / robust_scale,
                torch.log1p(robust_scale).expand_as(speed),
                torch.log1p(torch.linalg.vector_norm(acceleration, dim=-1, keepdim=True)),
                (speed - previous_speed) / robust_scale,
            ), dim=-1,
        )
        return shape, scale

    def forward(self, past: Tensor, frequency: float) -> Tensor:
        shape, scale = self.features(past, frequency)
        correction = self.shape(shape) + torch.sigmoid(self.scale_gate(scale)) * self.scale(scale)
        return 0.25 * torch.tanh(correction)


class BaselinePlusShapeScaleResidual(nn.Module):
    def __init__(self, base: nn.Module, token_dim: int) -> None:
        super().__init__()
        self.base = base
        self.residual = ShapeScaleResidual(token_dim)

    def forward(self, past: Tensor, frequency: float) -> Tensor:
        return self.base(past, frequency) + self.residual(past, frequency)


def replace_encoder(model: nn.Module, cfg, experiment: str) -> None:
    del cfg
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown EXP-138 encoder experiment: {experiment}")
    if getattr(model, "front_variant", None) != "f16_temporal_social_cross":
        raise ValueError("EXP-138A requires the locked F16 encoder")
    if model.robust_state is None:
        raise ValueError("EXP-138A requires the existing robust state")
    model.robust_state = BaselinePlusShapeScaleResidual(
        model.robust_state, model.token_dim
    )
