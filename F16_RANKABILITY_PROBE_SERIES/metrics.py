"""
Comprehensive Evaluation Metrics for Rankability Probe Series.

Enforces Butler Rules:
- research.goal_aligned_evaluation: Primary focus on Selection Regret and Regret Closure
- research.no_hardcoded_metrics: All statistics computed dynamically from predictions and ground-truth arrays
- research.anti_arbitrary_thresholding: Continuous regret distribution quantiles and bootstrap confidence intervals
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def compute_pairwise_accuracy(
    scores: np.ndarray,
    ades: np.ndarray,
    min_delta: float = 0.02,
) -> float:
    """
    Computes pairwise ranking accuracy for pairs with delta ADE > min_delta.
    delta_ade[n, i, j] = ades[n, j] - ades[n, i]
    When delta_ade > 0, candidate i is better than candidate j.
    Accuracy is fraction where score_i > score_j.
    """
    # delta_ade[n, i, j] = ADE_j - ADE_i
    # delta > 0 => i is better than j
    delta_ade = ades[:, None, :] - ades[:, :, None]

    # score_i - score_j
    s_diff = scores[:, :, None] - scores[:, None, :]

    mask = delta_ade > min_delta
    if np.sum(mask) == 0:
        return 0.0

    correct = (s_diff > 0) & mask
    return float(np.sum(correct) / np.sum(mask))


def compute_oracle_margin_accuracy(
    scores: np.ndarray,
    ades: np.ndarray,
    oracle_indices: np.ndarray,
    margin_threshold: float = 0.10,
) -> float:
    """
    Computes fraction of competitor modes j with ADE_j - ADE_oracle > margin_threshold
    where score_oracle > score_j.
    """
    N, K = scores.shape
    oracle_scores = scores[np.arange(N), oracle_indices]  # [N]
    oracle_ades = ades[np.arange(N), oracle_indices]      # [N]

    competitor_gap = ades - oracle_ades[:, None]          # [N, K]
    score_diff = oracle_scores[:, None] - scores          # [N, K]

    mask = competitor_gap > margin_threshold
    if np.sum(mask) == 0:
        return 0.0

    correct = (score_diff > 0) & mask
    return float(np.sum(correct) / np.sum(mask))


def compute_sample_spearman(scores: np.ndarray, ades: np.ndarray) -> np.ndarray:
    """
    Computes per-sample Spearman rank correlation between score and -ADE.
    """
    N = scores.shape[0]
    rhos = np.zeros(N, dtype=np.float64)
    for i in range(N):
        rho, _ = spearmanr(scores[i], -ades[i])
        rhos[i] = 0.0 if np.isnan(rho) else rho
    return rhos


def evaluate_probe_predictions(
    scores: np.ndarray,
    cache_data: dict[str, np.ndarray],
    native_selected_ade: float | None = None,
    oracle_min_ade: float | None = None,
) -> dict[str, float]:
    """
    Computes full spectrum of rankability metrics on a given set of scores.
    scores: [N, 20]
    cache_data: dict containing ADE_final, FDE_final, oracle_final_index, native_selected_index
    """
    N, K = scores.shape
    ade_final = cache_data["ADE_final"]
    fde_final = cache_data["FDE_final"]
    oracle_idx = cache_data["oracle_final_index"]
    native_sel_idx = cache_data["native_selected_index"]

    probe_sel_idx = np.argmax(scores, axis=-1)

    # Per-sample metrics
    sample_sel_ade = ade_final[np.arange(N), probe_sel_idx]
    sample_sel_fde = fde_final[np.arange(N), probe_sel_idx]
    sample_ora_ade = ade_final[np.arange(N), oracle_idx]
    sample_regret = sample_sel_ade - sample_ora_ade

    # Baseline references
    if native_selected_ade is None:
        native_selected_ade = float(np.mean(ade_final[np.arange(N), native_sel_idx]))
    if oracle_min_ade is None:
        oracle_min_ade = float(np.mean(sample_ora_ade))

    mean_sel_ade = float(np.mean(sample_sel_ade))
    mean_sel_fde = float(np.mean(sample_sel_fde))
    mean_regret = float(np.mean(sample_regret))

    # Regret Closure Ratio = (Native_ADE - Probe_ADE) / (Native_ADE - Oracle_ADE)
    gap = native_selected_ade - oracle_min_ade
    closure_ratio = (native_selected_ade - mean_sel_ade) / gap if gap > 1e-6 else 0.0

    # Oracle Recalls
    sorted_modes = np.argsort(-scores, axis=-1)  # descending order of scores
    recall_1 = float(np.mean(probe_sel_idx == oracle_idx))
    recall_3 = float(np.mean([oracle_idx[i] in sorted_modes[i, :3] for i in range(N)]))
    recall_5 = float(np.mean([oracle_idx[i] in sorted_modes[i, :5] for i in range(N)]))

    # Pairwise accuracies
    pairwise_all = compute_pairwise_accuracy(scores, ade_final, min_delta=0.02)
    pairwise_05 = compute_pairwise_accuracy(scores, ade_final, min_delta=0.05)
    pairwise_10 = compute_pairwise_accuracy(scores, ade_final, min_delta=0.10)
    pairwise_20 = compute_pairwise_accuracy(scores, ade_final, min_delta=0.20)

    # Oracle margin accuracies
    oracle_margin_10 = compute_oracle_margin_accuracy(scores, ade_final, oracle_idx, margin_threshold=0.10)
    oracle_margin_20 = compute_oracle_margin_accuracy(scores, ade_final, oracle_idx, margin_threshold=0.20)

    # Spearman rank correlation
    rhos = compute_sample_spearman(scores, ade_final)
    mean_spearman = float(np.mean(rhos))

    return {
        "selected_ade": mean_sel_ade,
        "selected_fde": mean_sel_fde,
        "oracle_min_ade": oracle_min_ade,
        "selection_regret": mean_regret,
        "regret_closure_pct": float(closure_ratio * 100.0),
        "oracle_recall_1": recall_1,
        "oracle_recall_3": recall_3,
        "oracle_recall_5": recall_5,
        "spearman_rho": mean_spearman,
        "pairwise_acc_all": pairwise_all,
        "pairwise_acc_05": pairwise_05,
        "pairwise_acc_10": pairwise_10,
        "pairwise_acc_20": pairwise_20,
        "oracle_margin_10": oracle_margin_10,
        "oracle_margin_20": oracle_margin_20,
        "regret_median": float(np.median(sample_regret)),
        "regret_p75": float(np.percentile(sample_regret, 75)),
        "regret_p90": float(np.percentile(sample_regret, 90)),
    }


def compute_bootstrap_ci(
    native_ades: np.ndarray,
    probe_ades: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 20260921,
) -> dict[str, float]:
    """
    Computes 95% bootstrap confidence intervals on delta ADE (native_ade - probe_ade).
    Positive delta means probe is better than native.
    """
    rng = np.random.RandomState(seed)
    N = len(native_ades)
    deltas = native_ades - probe_ades
    boot_means = np.zeros(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        idx = rng.choice(N, size=N, replace=True)
        boot_means[b] = np.mean(deltas[idx])

    mean_delta = float(np.mean(deltas))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        "mean_delta_ade": mean_delta,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "is_significant": bool(ci_lower > 0.0),
    }
