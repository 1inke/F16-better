"""
Comprehensive evaluation metrics for Quality-Aware F16.

Separates:
A. Generation Quality (minADE20, minFDE20, trajectory/endpoint diversity)
B. Selection Quality (Selected ADE/FDE, Regret, Oracle Hit, Near-Oracle@5cm, Pairwise)
C. Probability Quality (Exact Oracle NLL, Soft-target CE, Brier Score, Top-1 ECE, Entropy)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F


def compute_generation_metrics(
    candidates: np.ndarray,
    ade: np.ndarray,
    fde: np.ndarray,
) -> Dict[str, float]:
    """
    candidates: [N, K=20, T=12, 2]
    ade:        [N, K=20]
    fde:        [N, K=20]
    """
    min_ade = float(np.mean(np.min(ade, axis=-1)))
    min_fde = float(np.mean(np.min(fde, axis=-1)))

    # Pairwise endpoint diversity
    endpoints = candidates[:, :, -1, :] # [N, K, 2]
    N, K, _ = endpoints.shape
    
    diff = endpoints[:, :, None, :] - endpoints[:, None, :, :] # [N, K, K, 2]
    dist = np.linalg.norm(diff, axis=-1) # [N, K, K]
    
    # Upper triangle mask without diagonal
    triu_idx = np.triu_indices(K, k=1)
    ep_div = float(np.mean(dist[:, triu_idx[0], triu_idx[1]]))

    # Pairwise trajectory diversity (mean over time)
    diff_traj = candidates[:, :, None, :, :] - candidates[:, None, :, :, :] # [N, K, K, T, 2]
    dist_traj = np.mean(np.linalg.norm(diff_traj, axis=-1), axis=-1) # [N, K, K]
    traj_div = float(np.mean(dist_traj[:, triu_idx[0], triu_idx[1]]))

    return {
        "min_ade20": min_ade,
        "min_fde20": min_fde,
        "endpoint_diversity": ep_div,
        "trajectory_diversity": traj_div,
    }


def compute_selection_metrics(
    selected_idx: np.ndarray,
    ade: np.ndarray,
    fde: np.ndarray,
    collisions: Optional[np.ndarray] = None,
    has_neighbours: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    selected_idx: [N]
    ade:          [N, K=20]
    fde:          [N, K=20]
    """
    N = len(selected_idx)
    sample_range = np.arange(N)

    sel_ade = ade[sample_range, selected_idx]
    sel_fde = fde[sample_range, selected_idx]

    min_ade = np.min(ade, axis=-1)
    min_idx = np.argmin(ade, axis=-1)

    regret = sel_ade - min_ade
    oracle_hit = float(np.mean(selected_idx == min_idx))
    near_oracle_05 = float(np.mean(regret <= 0.05))

    # Pairwise Ranking Accuracy:
    # High score (low ADE) vs lower score. For pairs (i, j) where ADE_i < ADE_j - delta,
    # correctly ranked if score_i > score_j.
    # Here selection metrics are evaluated on selected candidate.
    
    metrics = {
        "selected_ade": float(np.mean(sel_ade)),
        "selected_fde": float(np.mean(sel_fde)),
        "selection_regret": float(np.mean(regret)),
        "exact_oracle_hit": oracle_hit,
        "near_oracle_05": near_oracle_05,
    }

    if collisions is not None:
        sel_coll = collisions[sample_range, selected_idx]
        metrics["collision_all"] = float(np.mean(sel_coll))
        if has_neighbours is not None and np.any(has_neighbours):
            metrics["collision_social"] = float(np.mean(sel_coll[has_neighbours]))
        else:
            metrics["collision_social"] = float(np.mean(sel_coll))

    return metrics


def compute_pairwise_accuracy(
    scores: np.ndarray,
    ade: np.ndarray,
    delta: float = 0.05,
) -> float:
    """
    Pairwise ranking accuracy on candidates:
    For all pairs (i, j) with ade_j - ade_i > delta (i is substantially better than j):
    Acc = mean(score_i > score_j).
    """
    N, K = scores.shape
    diff_ade = ade[:, None, :] - ade[:, :, None] # [N, K, K]: ade_j - ade_i
    diff_score = scores[:, :, None] - scores[:, None, :] # [N, K, K]: score_i - score_j

    valid_mask = diff_ade > delta
    if not np.any(valid_mask):
        return 0.5

    correct = diff_score[valid_mask] > 0.0
    return float(np.mean(correct))


def compute_probability_metrics(
    probs: np.ndarray,
    ade: np.ndarray,
    tau_eval: float = 0.10,
    ece_bins: int = 10,
) -> Dict[str, float]:
    """
    probs:    [N, K=20] valid probability distribution (sum to 1)
    ade:      [N, K=20]
    """
    N, K = probs.shape
    eps = 1e-12
    probs_clamped = np.clip(probs, eps, 1.0)

    # 1. Exact Oracle NLL
    k_star = np.argmin(ade, axis=-1)
    p_k_star = probs_clamped[np.arange(N), k_star]
    oracle_nll = float(np.mean(-np.log(p_k_star)))

    # 2. Soft-target Cross Entropy (eval tau = 0.10)
    q_eval = np.exp(-ade / float(tau_eval))
    q_eval = q_eval / np.sum(q_eval, axis=-1, keepdims=True)
    soft_ce = float(np.mean(-np.sum(q_eval * np.log(probs_clamped), axis=-1)))

    # 3. Brier Score
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), k_star] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))

    # 4. Mean Max Probability & Entropy
    max_p = np.max(probs, axis=-1)
    mean_max_prob = float(np.mean(max_p))
    entropy = -np.sum(probs_clamped * np.log(probs_clamped), axis=-1)
    mean_entropy = float(np.mean(entropy))

    # 5. Top-1 ECE (Expected Calibration Error)
    # Correctness: Selected Top-1 candidate is Near-Oracle (regret <= 0.05m)
    selected_idx = np.argmax(probs, axis=-1)
    min_ade = np.min(ade, axis=-1)
    sel_ade = ade[np.arange(N), selected_idx]
    correctness = (sel_ade - min_ade <= 0.05).astype(float)
    confidence = max_p

    bin_boundaries = np.linspace(0, 1, ece_bins + 1)
    ece = 0.0
    for i in range(ece_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper) if i > 0 else (confidence >= bin_lower) & (confidence <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correctness[in_bin])
            avg_confidence_in_bin = np.mean(confidence[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return {
        "oracle_nll": oracle_nll,
        "soft_ce": soft_ce,
        "brier_score": brier,
        "top1_ece": float(ece),
        "mean_max_prob": mean_max_prob,
        "mean_entropy": mean_entropy,
    }
