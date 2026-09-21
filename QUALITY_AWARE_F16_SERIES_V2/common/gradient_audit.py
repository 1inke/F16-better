"""
Gradient Direction Similarity and Redundancy Audit.

Analyzes the gradient direction of logits for:
1. Native classification supervision:
   g_orig = 0.5 * [(p - q_orig) + 0.25 * (p - one_hot(winner))]
2. Proposed auxiliary soft CE:
   g_new = p - q_new
"""
from __future__ import annotations

import numpy as np


def compute_gradient_vectors(
    logits: np.ndarray,
    q_orig: np.ndarray,
    q_new: np.ndarray,
    winner: np.ndarray,
    hard_weight: float = 0.25,
    cls_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes exact analytic gradient vectors w.r.t logits.
    logits: [N, K]
    q_orig, q_new: [N, K]
    winner: [N] (winner indices)

    Returns:
        g_orig: [N, K]
        g_new: [N, K]
    """
    # p = softmax(logits)
    scaled_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(scaled_logits)
    p = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # One-hot winner
    N, K = logits.shape
    y_winner = np.zeros_like(p)
    y_winner[np.arange(N), winner] = 1.0

    # Native classification gradient
    g_soft_orig = p - q_orig
    g_hard_orig = p - y_winner
    g_orig = cls_weight * (g_soft_orig + hard_weight * g_hard_orig)

    # Auxiliary candidate gradient
    g_new = p - q_new

    return g_orig, g_new


def analyze_gradient_similarity(
    g_orig: np.ndarray,
    g_new: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """
    Computes per-sample gradient similarity metrics.
    """
    dot = np.sum(g_orig * g_new, axis=-1)
    norm_orig = np.linalg.norm(g_orig, axis=-1)
    norm_new = np.linalg.norm(g_new, axis=-1)
    denom = np.clip(norm_orig * norm_new, eps, None)

    cos_sim = dot / denom
    norm_ratio = norm_new / np.clip(norm_orig, eps, None)

    # Sign conflict ratio
    sign_orig = np.sign(g_orig)
    sign_new = np.sign(g_new)
    # Exclude exact zeros
    conflict_mask = (sign_orig != sign_new) & (sign_orig != 0) & (sign_new != 0)
    sign_conflict_ratio = np.mean(conflict_mask.astype(np.float64), axis=-1)

    return {
        "cosine": cos_sim,
        "norm_ratio": norm_ratio,
        "sign_conflict_ratio": sign_conflict_ratio,
    }
