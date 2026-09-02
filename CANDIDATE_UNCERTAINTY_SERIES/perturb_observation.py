"""Physically small, deterministic observation perturbations for EXP-179A."""

from __future__ import annotations

import torch
from torch.nn import functional as F


PRIMARY_VARIANTS = ("original", "p1_plus", "p1_minus", "p3_plus", "p3_minus")


def valid_agent_mask(past: torch.Tensor, num_valid: torch.Tensor) -> torch.Tensor:
    agents = torch.arange(past.shape[1], device=past.device)[None]
    return (agents < num_valid[:, None]).to(past.dtype)[:, :, None, None]


def coordinate_noise_pair(past: torch.Tensor, num_valid: torch.Tensor, seed: int):
    """Symmetric Gaussian perturbation, sigma=1% of focal median step length."""
    focal_step = torch.linalg.vector_norm(torch.diff(past[:, 0], dim=1), dim=-1)
    scale = focal_step.median(dim=1).values.clamp_min(1e-4) * 0.01
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(past.shape, generator=generator, dtype=torch.float32).to(past.device, past.dtype)
    noise = noise * scale[:, None, None, None] * valid_agent_mask(past, num_valid)
    return past + noise, past - noise


def recent_speed(past: torch.Tensor, num_valid: torch.Tensor, factor: float):
    """Scale only the final two velocity vectors while preserving each heading."""
    result = past.clone(); valid = valid_agent_mask(past, num_valid)
    result[:, :, -2] = past[:, :, -3] + factor * (past[:, :, -2] - past[:, :, -3])
    result[:, :, -1] = result[:, :, -2] + factor * (past[:, :, -1] - past[:, :, -2])
    return result * valid + past * (1.0 - valid)


def early_step_drop(past: torch.Tensor, num_valid: torch.Tensor):
    """Drop the first observation and linearly resample the remaining seven to eight."""
    batch, agents, _, dims = past.shape
    values = past[:, :, 1:].permute(0, 1, 3, 2).reshape(batch * agents, dims, 7)
    aligned = F.interpolate(values, size=8, mode="linear", align_corners=True)
    aligned = aligned.reshape(batch, agents, dims, 8).permute(0, 1, 3, 2)
    valid = valid_agent_mask(past, num_valid)
    return aligned * valid + past * (1.0 - valid)


def make_primary(past: torch.Tensor, num_valid: torch.Tensor, seed: int):
    plus, minus = coordinate_noise_pair(past, num_valid, seed)
    return {
        "original": past,
        "p1_plus": plus,
        "p1_minus": minus,
        "p3_plus": recent_speed(past, num_valid, 1.02),
        "p3_minus": recent_speed(past, num_valid, 0.98),
    }
