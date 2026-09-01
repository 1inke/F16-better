"""Trajectory-set distance used by EXP-172A.

All distances are prediction-to-prediction and never use ground truth.
"""
from __future__ import annotations

import torch
from torch import Tensor


def pairwise_trajectory_distance(trajectories: Tensor, fde_weight: float = 0.25) -> Tensor:
    """Return ADE + fde_weight * FDE for every trajectory pair.

    Args:
        trajectories: [batch, candidates, time, xy]
    Returns:
        [batch, candidates, candidates]
    """
    delta = trajectories[:, :, None] - trajectories[:, None, :]
    point = torch.linalg.vector_norm(delta, dim=-1)
    return point.mean(dim=-1) + fde_weight * point[..., -1]


def mean_off_diagonal(distance: Tensor) -> Tensor:
    n = distance.shape[-1]
    mask = ~torch.eye(n, dtype=torch.bool, device=distance.device)
    return distance[:, mask].mean(dim=-1)
