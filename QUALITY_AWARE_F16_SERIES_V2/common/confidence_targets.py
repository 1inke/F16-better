"""
Confidence Target Builders for F16 Confidence Objective Refinement V2.

Provides exact target builders for:
1. build_original_target: Exact match to locked ModeQueryLoss soft classification
2. build_fixed_quality_target: Fixed temperature on ADE + 0.25*FDE
3. build_fixed_ade_target: Fixed temperature on ADE only
"""
from __future__ import annotations

import torch
from torch import Tensor


def build_original_target(
    predictions: Tensor,
    target: Tensor,
    winner_fde_weight: float = 0.25,
    soft_temperature: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Builds the original adaptive quality target strictly matching locked ModeQueryLoss.

    predictions: [..., K, T, 2]
    target: [..., T, 2]
    Returns:
        q: [..., K] (detached soft distribution)
        winner: [...] (argmax/argmin index)
        quality: [..., K] (ADE + winner_fde_weight * FDE)
    """
    distances = torch.linalg.vector_norm(predictions - target.unsqueeze(-3), dim=-1)
    quality = distances.mean(dim=-1) + winner_fde_weight * distances[..., -1]
    detached = quality.detach()

    gap = detached - detached.min(dim=-1, keepdim=True).values
    scale = gap.mean(dim=-1, keepdim=True).clamp_min(0.05)
    q = torch.softmax(-gap / (soft_temperature * scale), dim=-1)
    winner = detached.argmin(dim=-1)

    return q, winner, quality


def build_fixed_quality_target(
    predictions: Tensor,
    target: Tensor,
    tau: float,
    fde_weight: float = 0.25,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Builds fixed-temperature physical quality target (ADE + fde_weight * FDE).
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    distances = torch.linalg.vector_norm(predictions - target.unsqueeze(-3), dim=-1)
    quality = distances.mean(dim=-1) + fde_weight * distances[..., -1]
    detached = quality.detach()

    q = torch.softmax(-detached / float(tau), dim=-1)
    winner = detached.argmin(dim=-1)

    return q, winner, quality


def build_fixed_ade_target(
    predictions: Tensor,
    target: Tensor,
    tau: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Builds fixed-temperature ADE-only quality target.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    distances = torch.linalg.vector_norm(predictions - target.unsqueeze(-3), dim=-1)
    ade = distances.mean(dim=-1)
    detached = ade.detach()

    q = torch.softmax(-detached / float(tau), dim=-1)
    winner = detached.argmin(dim=-1)

    return q, winner, ade
