"""
Unified Local Evaluation Metrics for S6 Local Fine Reranking.

Computes comprehensive performance, safety, and discrimination metrics for Local Rerankers:
- Top-1 Selected ADE / FDE
- ADE / FDE Regret vs Global Oracle
- Exact Oracle Hit Rate
- Near-Oracle Recall@2cm, @5cm, @10cm
- Collision Rate
- Useful / False Intervention Rates
- Local Pairwise Accuracy (@5cm, @10cm) & Local Spearman Correlation
"""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from scipy.stats import spearmanr


def evaluate_local_predictions(
    selected_indices: np.ndarray,       # (N,) mode indices in [0..19]
    topk_indices: np.ndarray,           # (N, K) Top-K mode indices
    topk_scores: np.ndarray,            # (N, K) local scores
    full_ade: np.ndarray,               # (N, 20) ground truth ADE
    full_fde: np.ndarray,               # (N, 20) ground truth FDE
    full_collision: np.ndarray,         # (N, 20) ground truth collision boolean
    native_indices: np.ndarray,         # (N,) Native top-1 indices
    has_neighbors: np.ndarray | None = None, # (N,) boolean or int
) -> Dict[str, float]:
    N = len(selected_indices)
    sample_range = np.arange(N)

    # 1. Selected ADE & FDE
    sel_ade = full_ade[sample_range, selected_indices]
    sel_fde = full_fde[sample_range, selected_indices]

    # Global Oracle
    glob_oracle_ade = np.min(full_ade, axis=-1)
    glob_oracle_fde = np.min(full_fde, axis=-1)
    glob_oracle_idx = np.argmin(full_ade, axis=-1)

    # Native Top-1
    nat_ade = full_ade[sample_range, native_indices]

    # 2. Regret
    ade_regret = sel_ade - glob_oracle_ade
    fde_regret = sel_fde - glob_oracle_fde

    # 3. Oracle Hit & Near-Oracle Recall
    oracle_hit = (selected_indices == glob_oracle_idx).astype(float)
    near_02 = (sel_ade <= (glob_oracle_ade + 0.02)).astype(float)
    near_05 = (sel_ade <= (glob_oracle_ade + 0.05)).astype(float)
    near_10 = (sel_ade <= (glob_oracle_ade + 0.10)).astype(float)

    # 4. Collision Rate (All vs Social)
    collision = full_collision[sample_range, selected_indices].astype(float)
    collision_all = float(np.mean(collision))

    if has_neighbors is not None:
        social_mask = (has_neighbors > 0)
        if np.any(social_mask):
            collision_social = float(np.mean(collision[social_mask]))
        else:
            collision_social = float("nan")
    else:
        collision_social = float("nan")

    # 5. Intervention Dynamics
    intervened = (selected_indices != native_indices)
    # Useful intervention: native was NOT near-oracle (regret > 5cm), and selector improved ADE by >= 2cm
    nat_regret = nat_ade - glob_oracle_ade
    nat_was_bad = (nat_regret > 0.05)
    improved = (sel_ade < (nat_ade - 0.02))
    useful_intervention = (intervened & nat_was_bad & improved).astype(float)

    # False intervention: native was near-oracle (regret <= 5cm), but selector switched to a worse candidate
    nat_was_good = (nat_regret <= 0.05)
    degraded = (sel_ade > (nat_ade + 0.02))
    false_intervention = (intervened & nat_was_good & degraded).astype(float)

    # 6. Local Pairwise Accuracy & Spearman inside Top-K
    topk_ade = np.take_along_axis(full_ade, topk_indices, axis=-1)  # (N, K)
    N_eval = min(N, 2000)
    sub_scores = topk_scores[:N_eval]
    sub_ade = topk_ade[:N_eval]

    # Vectorized pairwise comparison
    ade_gap = sub_ade[:, None, :] - sub_ade[:, :, None]      # (N, K, K): > 0 means i has lower error than j
    score_gap = sub_scores[:, :, None] - sub_scores[:, None, :]  # (N, K, K): > 0 means i scored higher than j

    def calc_pair_acc(margin: float) -> float:
        valid = ade_gap > margin
        if not np.any(valid):
            return float("nan")
        correct = (score_gap > 0) & valid
        return float(np.sum(correct) / np.sum(valid))

    local_p05 = calc_pair_acc(0.05)
    local_p10 = calc_pair_acc(0.10)

    # Spearman
    spearmans = []
    for s, a in zip(sub_scores, sub_ade):
        sp, _ = spearmanr(s, -a)
        if not np.isnan(sp):
            spearmans.append(sp)
    local_spearman = float(np.mean(spearmans)) if spearmans else float("nan")

    return {
        "selected_ade": float(np.mean(sel_ade)),
        "selected_fde": float(np.mean(sel_fde)),
        "ade_regret": float(np.mean(ade_regret)),
        "fde_regret": float(np.mean(fde_regret)),
        "oracle_hit": float(np.mean(oracle_hit)),
        "near_oracle_02": float(np.mean(near_02)),
        "near_oracle_05": float(np.mean(near_05)),
        "near_oracle_10": float(np.mean(near_10)),
        "collision_rate": collision_social if not np.isnan(collision_social) else collision_all,
        "collision_social": collision_social,
        "collision_all": collision_all,
        "intervention_rate": float(np.mean(intervened.astype(float))),
        "useful_intervention_rate": float(np.mean(useful_intervention)),
        "false_intervention_rate": float(np.mean(false_intervention)),
        "local_pairwise_05": local_p05,
        "local_pairwise_10": local_p10,
        "local_spearman": local_spearman,
    }
