"""
Gradient Direction Similarity and Redundancy Audit.

Enforces Butler Rule: research.semantic_objective_alignment
Computes exact analytic gradient vectors w.r.t logits for:
1. Pure Soft Distribution Gradients (strictly isolates target distribution shape):
   g_soft_orig = p - q_orig
   g_soft_cand = p - q_cand
2. Full Training Objective Gradients (strictly symmetric with true loss formulation):
   g_total_orig = w_cls * [(p - q_orig) + w_hard * (p - y_winner)]
   g_total_cand = w_cls * [(p - q_cand) + w_hard * (p - y_winner)]
3. Soft-Only Objective Gradients (when evaluating removal of hard CE):
   g_softonly_cand = w_cls * (p - q_cand)
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    scaled = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def compute_gradient_vectors_all_modes(
    logits: np.ndarray,
    q_orig: np.ndarray,
    q_cand: np.ndarray,
    winner_orig: np.ndarray,
    winner_cand: np.ndarray,
    hard_weight: float = 0.25,
    cls_weight: float = 0.5,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Computes mathematically symmetric pairs of gradient vectors w.r.t logits.
    winner_orig: [N] ground-truth winner for original objective (e.g. argmin(ADE + 0.25*FDE))
    winner_cand: [N] ground-truth winner for candidate objective (e.g. argmin(ADE))

    Returns dict mapping mode_name -> (g_orig, g_cand)
    """
    p = softmax(logits)
    N, K = logits.shape
    y_orig = np.zeros_like(p)
    y_cand = np.zeros_like(p)
    y_orig[np.arange(N), winner_orig] = 1.0
    y_cand[np.arange(N), winner_cand] = 1.0

    # 1. Pure soft gradients
    g_soft_orig = p - q_orig
    g_soft_cand = p - q_cand

    # 2. Symmetric full objective gradients (with respective hard CE on each side)
    g_full_orig = cls_weight * (g_soft_orig + hard_weight * (p - y_orig))
    g_full_cand = cls_weight * (g_soft_cand + hard_weight * (p - y_cand))

    # 3. Soft-only candidate vs full native original
    g_softonly_cand = cls_weight * g_soft_cand

    return {
        "pure_soft": (g_soft_orig, g_soft_cand),
        "full_objective": (g_full_orig, g_full_cand),
        "soft_only_vs_full": (g_full_orig, g_softonly_cand),
    }


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
    conflict_mask = (sign_orig != sign_new) & (sign_orig != 0) & (sign_new != 0)
    sign_conflict_ratio = np.mean(conflict_mask.astype(np.float64), axis=-1)

    return {
        "cosine": cos_sim,
        "norm_ratio": norm_ratio,
        "sign_conflict_ratio": sign_conflict_ratio,
    }
