"""Non-decoder adaptations for the locked F16 + kinematic-K20 model.

Every experiment keeps the sparse graph, temporal-social cross attention,
decoder, losses, augmentation, and heading normalization unchanged.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


EXPERIMENTS = {
    "n1_speed_scale_state",
    "n2_target_dual_temporal",
    "n3_density_self_motion",
}


def _motion_terms(past: Tensor, frequency: float) -> tuple[Tensor, ...]:
    """Return velocity, long velocity, acceleration, scale, heading and confidence."""
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
    heading = velocity / speed.clamp_min(1.0e-5)
    confidence = torch.sigmoid((speed - 0.05) / 0.05)
    heading = heading * confidence
    return velocity, long_velocity, acceleration, robust_scale, heading, confidence


class SpeedScaleNormalizedState(nn.Module):
    """Normalize motion channels per agent while preserving absolute log-speed."""

    feature_dim = 11

    def __init__(self, token_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, -0.5)

    @staticmethod
    def features(past: Tensor, frequency: float) -> Tensor:
        velocity, long_velocity, acceleration, scale, heading, confidence = (
            _motion_terms(past, frequency)
        )
        previous_heading = torch.cat((heading[:, :, :1], heading[:, :, :-1]), dim=2)
        cross = previous_heading[..., 0] * heading[..., 1] - previous_heading[..., 1] * heading[..., 0]
        dot = (previous_heading * heading).sum(dim=-1).clamp(-1.0, 1.0)
        turn = torch.atan2(cross, dot).unsqueeze(-1) / math.pi
        scale_channel = torch.log1p(scale).expand_as(confidence)
        return torch.cat(
            (
                velocity / scale,
                long_velocity / scale,
                acceleration / (scale * frequency),
                heading,
                turn * confidence,
                scale_channel,
                confidence,
            ),
            dim=-1,
        )

    def forward(self, past: Tensor, frequency: float) -> Tensor:
        features = self.features(past, frequency)
        return torch.tanh(self.encoder(features)) * torch.sigmoid(self.gate(features))


class ContextWrapper(nn.Module):
    """Base wrapper preserving the context API expected by the F16 encoder."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self._past: Optional[Tensor] = None
        self._num_valid: Optional[Tensor] = None

    def set_context(self, past: Tensor, num_valid: Optional[Tensor]) -> None:
        self._past = past
        self._num_valid = num_valid
        self.base.set_context(past, num_valid)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


class TargetDualTemporalAdapter(ContextWrapper):
    """Add short-term and full-history residuals to focal tokens before F16."""

    def __init__(self, base: nn.Module, dim: int, obs_len: int, agents: int) -> None:
        super().__init__(base)
        self.dim = dim
        self.obs_len = obs_len
        self.agents = agents
        self.short_depthwise = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.short_out = nn.Linear(dim, dim)
        self.long_mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, max(24, dim // 2)), nn.GELU(),
            nn.Linear(max(24, dim // 2), dim),
        )
        self.mix_gate = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, max(24, dim // 2)),
            nn.GELU(), nn.Linear(max(24, dim // 2), 1),
        )
        nn.init.zeros_(self.short_out.weight)
        nn.init.zeros_(self.short_out.bias)
        nn.init.zeros_(self.long_mlp[-1].weight)
        nn.init.zeros_(self.long_mlp[-1].bias)
        nn.init.constant_(self.mix_gate[-1].bias, -1.5)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch = src.shape[0]
        tokens = src.reshape(batch, self.obs_len, self.agents, self.dim)
        focal = tokens[:, :, 0]
        short = self.short_depthwise(focal.transpose(1, 2)).transpose(1, 2)
        short = self.short_out(short)
        long = self.long_mlp(focal.mean(dim=1)).unsqueeze(1).expand_as(focal)
        gate = torch.sigmoid(
            self.mix_gate(torch.cat((focal[:, -1], focal.mean(dim=1)), dim=-1))
        ).unsqueeze(1)
        enhanced_focal = focal + gate * (short + long)
        tokens = torch.cat((enhanced_focal[:, :, None], tokens[:, :, 1:]), dim=2)
        return self.base(tokens.reshape_as(src), mask=mask)


class DensitySelfMotionAdapter(ContextWrapper):
    """Supply an extra focal self-motion residual primarily in sparse scenes."""

    feature_dim = 8

    def __init__(
        self, base: nn.Module, dim: int, obs_len: int, agents: int, frequency: float
    ) -> None:
        super().__init__(base)
        self.dim = dim
        self.obs_len = obs_len
        self.agents = agents
        self.frequency = float(frequency)
        self.motion_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Linear(self.feature_dim, max(24, dim // 2)),
            nn.GELU(), nn.Linear(max(24, dim // 2), dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1),
        )
        nn.init.zeros_(self.motion_encoder[-1].weight)
        nn.init.zeros_(self.motion_encoder[-1].bias)
        nn.init.constant_(self.gate[-1].bias, -1.5)
        self.last_density_gate: Optional[Tensor] = None

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        encoded = self.base(src, mask=mask)
        if self._past is None:
            raise RuntimeError("Density adapter requires trajectory context")
        past = self._past[:, :1]
        velocity, _, acceleration, scale, heading, confidence = _motion_terms(
            past, self.frequency
        )
        motion = torch.cat(
            (
                velocity / scale,
                acceleration / (scale * self.frequency),
                heading,
                torch.log1p(scale).expand_as(confidence),
                confidence,
            ),
            dim=-1,
        )[:, 0]
        if self._num_valid is None:
            valid_agents = encoded.new_full((encoded.shape[0],), float(self.agents))
        else:
            valid_agents = self._num_valid.to(encoded.dtype).clamp(1, self.agents)
        density = (valid_agents - 1.0) / max(self.agents - 1, 1)
        low_density = 1.0 - density
        speed_stat = torch.linalg.vector_norm(velocity[:, 0], dim=-1).mean(dim=-1)
        accel_stat = torch.linalg.vector_norm(acceleration[:, 0], dim=-1).mean(dim=-1)
        gate = torch.sigmoid(self.gate(torch.stack((low_density, speed_stat, accel_stat), dim=-1)))
        gate = gate * (0.25 + 0.75 * low_density[:, None])
        self.last_density_gate = gate.detach()
        tokens = encoded.reshape(encoded.shape[0], self.obs_len, self.agents, self.dim)
        enhanced_focal = tokens[:, :, 0] + gate[:, None] * self.motion_encoder(motion)
        tokens = torch.cat((enhanced_focal[:, :, None], tokens[:, :, 1:]), dim=2)
        return tokens.reshape_as(encoded)


def replace_encoder(model: nn.Module, cfg, experiment: str) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown non-decoder experiment: {experiment}")
    if getattr(model, "front_variant", None) != "f16_temporal_social_cross":
        raise ValueError("These experiments require the locked F16 front encoder")

    if experiment == "n1_speed_scale_state":
        model.robust_state = SpeedScaleNormalizedState(model.token_dim, hidden_dim=64)
        return

    base = model.agent_aware_trans_encoder
    if experiment == "n2_target_dual_temporal":
        model.agent_aware_trans_encoder = TargetDualTemporalAdapter(
            base, model.token_dim, model.obs_len, model.num_pedestrians
        )
    else:
        model.agent_aware_trans_encoder = DensitySelfMotionAdapter(
            base, model.token_dim, model.obs_len, model.num_pedestrians,
            float(cfg.DATA.FREQUENCY),
        )
