"""
Quality target construction for F16 Quality-Aware Probability Learning.

P0 Protocol:
All targets must be strictly detached from gradient computation (no_grad).
Target q_k = softmax(-error_k / tau).
"""
from __future__ import annotations

import torch


def build_quality_target_ade(
    candidates: torch.Tensor,
    gt: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    Constructs soft quality distribution target based on candidate ADE.
    
    Args:
        candidates: [B, K=20, T=12, D=2]
        gt:         [B, T=12, D=2]
        tau:        Temperature parameter (meters)
        
    Returns:
        target_q:   [B, K=20] probability distribution, strictly detached.
    """
    with torch.no_grad():
        err = torch.linalg.vector_norm(candidates - gt[:, None], dim=-1)
        ade = err.mean(dim=-1) # [B, K]
        target_q = torch.softmax(-ade / float(tau), dim=-1)
    return target_q


def build_quality_target_ade_fde(
    candidates: torch.Tensor,
    gt: torch.Tensor,
    tau: float,
    beta: float = 0.25,
) -> torch.Tensor:
    """
    Constructs soft quality distribution target based on ADE + beta * FDE.
    """
    with torch.no_grad():
        err = torch.linalg.vector_norm(candidates - gt[:, None], dim=-1)
        ade = err.mean(dim=-1)
        fde = err[:, :, -1]
        error = ade + float(beta) * fde
        target_q = torch.softmax(-error / float(tau), dim=-1)
    return target_q
