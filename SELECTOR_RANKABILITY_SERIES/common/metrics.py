"""
Unified Evaluation Metrics for Rankability Series.

Computes:
1. Top-1 Accuracy: Selected ADE, Selected FDE, Regret, Oracle Hit, Near-Oracle@0.05/0.10m
2. Ranking Metrics: Pairwise ranking accuracy (multi-margins), Spearman, Kendall Tau, NDCG@5/20
3. Gate & Intervention: Intervention rate, useful intervention, false intervention, recovered regret
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def compute_top1_metrics(
    selected_indices: np.ndarray,
    ade: np.ndarray,
    fde: np.ndarray,
    min_ade20: np.ndarray,
    min_fde20: np.ndarray,
    native_indices: np.ndarray,
) -> dict[str, float]:
    """Computes Top-1 accuracy and regret metrics across all samples."""
    N = len(selected_indices)
    rows = np.arange(N)

    sel_ade = ade[rows, selected_indices]
    sel_fde = fde[rows, selected_indices]
    nat_ade = ade[rows, native_indices]

    mean_ade = float(np.mean(sel_ade))
    mean_fde = float(np.mean(sel_fde))
    ade_regret = float(np.mean(sel_ade - min_ade20))
    fde_regret = float(np.mean(sel_fde - min_fde20))

    oracle_indices = np.argmin(ade, axis=-1)
    oracle_hit = float(np.mean(selected_indices == oracle_indices))

    # Near-Oracle thresholds
    near_002 = float(np.mean(sel_ade <= min_ade20 + 0.02))
    near_005 = float(np.mean(sel_ade <= min_ade20 + 0.05))
    near_010 = float(np.mean(sel_ade <= min_ade20 + 0.10))

    # Intervention statistics vs Native
    intervened = selected_indices != native_indices
    interv_rate = float(np.mean(intervened))
    if interv_rate > 0:
        useful = intervened & (sel_ade < nat_ade - 0.01)
        false_int = intervened & (sel_ade > nat_ade + 0.01)
        useful_rate = float(np.mean(useful))
        false_rate = float(np.mean(false_int))
        recovered_regret = float(np.mean(np.maximum(0.0, nat_ade - sel_ade)))
    else:
        useful_rate = 0.0
        false_rate = 0.0
        recovered_regret = 0.0

    return {
        "selected_ade": mean_ade,
        "selected_fde": mean_fde,
        "ade_regret": ade_regret,
        "fde_regret": fde_regret,
        "oracle_hit": oracle_hit,
        "near_oracle_002": near_002,
        "near_oracle_005": near_005,
        "near_oracle_010": near_010,
        "intervention_rate": interv_rate,
        "useful_intervention_rate": useful_rate,
        "false_intervention_rate": false_rate,
        "recovered_regret": recovered_regret,
    }


def compute_pairwise_accuracy(
    predicted_scores: np.ndarray,  # (N, 20) higher is better
    candidate_ade: np.ndarray,     # (N, 20) lower is better
    margins: list[float] = (0.01, 0.05, 0.10, 0.20),
    max_samples: int = 5000,
) -> dict[str, float]:
    """Computes pairwise accuracy for candidate pairs with ADE separation >= margin."""
    N, K = predicted_scores.shape
    if N > max_samples:
        sub_idx = np.random.choice(N, max_samples, replace=False)
        scores = predicted_scores[sub_idx]
        ade = candidate_ade[sub_idx]
    else:
        scores = predicted_scores
        ade = candidate_ade

    # ade_gap[m, i, j] = ade[m, j] - ade[m, i]
    # ade_gap > 0 means candidate i has strictly lower ADE than candidate j (candidate i is better than j)
    ade_gap = ade[:, None, :] - ade[:, :, None]

    # score_gap[m, i, j] = scores[m, i] - scores[m, j]
    # score_gap > 0 means predicted score for candidate i is higher than candidate j
    score_gap = scores[:, :, None] - scores[:, None, :]

    res = {}
    for m in margins:
        # Pairs where candidate i is strictly better than candidate j by at least margin
        valid_mask = ade_gap > m
        if np.any(valid_mask):
            correct = (score_gap > 0) & valid_mask
            acc = float(np.sum(correct) / np.sum(valid_mask))
        else:
            acc = float("nan")
        res[f"pairwise_acc_margin_{int(m*100):02d}"] = acc

    return res


def compute_ndcg(
    predicted_scores: np.ndarray,
    candidate_ade: np.ndarray,
    k_list: list[int] = (5, 20),
    tau: float = 0.1,
    max_samples: int = 5000,
) -> dict[str, float]:
    """Computes Normalized Discounted Cumulative Gain (NDCG) using exponential relevance."""
    N, K = predicted_scores.shape
    if N > max_samples:
        sub_idx = np.random.choice(N, max_samples, replace=False)
        scores = predicted_scores[sub_idx]
        ade = candidate_ade[sub_idx]
    else:
        scores = predicted_scores
        ade = candidate_ade

    relevance = np.exp(-ade / tau)  # higher is better, (M, 20)
    pred_order = np.argsort(-scores, axis=-1)  # indices of sorted scores descending
    ideal_order = np.argsort(-relevance, axis=-1)

    res = {}
    M = len(scores)
    discounts = 1.0 / np.log2(np.arange(2, K + 2))

    for k in k_list:
        dcg = np.zeros(M)
        idcg = np.zeros(M)
        for i in range(k):
            pred_cand = pred_order[:, i]
            ideal_cand = ideal_order[:, i]
            dcg += relevance[np.arange(M), pred_cand] * discounts[i]
            idcg += relevance[np.arange(M), ideal_cand] * discounts[i]
        ndcg = np.mean(dcg / (idcg + 1e-12))
        res[f"ndcg_{k}"] = float(ndcg)

    return res


def compute_correlation_metrics(
    predicted_scores: np.ndarray,
    candidate_ade: np.ndarray,
    max_samples: int = 2000,
) -> dict[str, float]:
    """Computes mean Spearman and Kendall correlation between scores and negative ADE."""
    N = len(predicted_scores)
    if N > max_samples:
        sub_idx = np.random.choice(N, max_samples, replace=False)
        scores = predicted_scores[sub_idx]
        ade = candidate_ade[sub_idx]
    else:
        scores = predicted_scores
        ade = candidate_ade

    spearmans = []
    kendalls = []

    for s, a in zip(scores, ade):
        target = -a
        sp, _ = spearmanr(s, target)
        kd, _ = kendalltau(s, target)
        if not np.isnan(sp):
            spearmans.append(sp)
        if not np.isnan(kd):
            kendalls.append(kd)

    return {
        "spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
        "kendall": float(np.mean(kendalls)) if kendalls else float("nan"),
    }
