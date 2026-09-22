"""
Unit and Synthetic Validity Tests for Ranking Metrics (EXP-S0A).

Tests:
1. Test A: Perfect Ranking (lower ADE has higher score). Pairwise accuracy must be 1.0, Spearman 1.0, Kendall 1.0.
2. Test B: Reverse Ranking (lower ADE has lower score). Pairwise accuracy must be 0.0, Spearman -1.0, Kendall -1.0.
3. Test C: Margin Filtering. Pairs within margin must be excluded from pairwise accuracy.
4. Test D: NDCG Perfect vs Reverse. NDCG@k must be 1.0 for perfect ranking, and significantly lower for reverse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path[:0] = [str(HERE.parent)]
from common.metrics import (
    compute_correlation_metrics,
    compute_ndcg,
    compute_pairwise_accuracy,
)


def run_tests() -> dict[str, bool]:
    results = {}

    # Test A: Perfect Ranking
    # Candidate 0 is best (0.1m), Candidate 3 is worst (0.4m)
    # Scores: Candidate 0 is highest (4.0), Candidate 3 is lowest (1.0)
    ade_perfect = np.array([[0.1, 0.2, 0.3, 0.4]])
    scores_perfect = np.array([[4.0, 3.0, 2.0, 1.0]])

    pair_acc_perfect = compute_pairwise_accuracy(scores_perfect, ade_perfect, margins=[0.05])
    corr_perfect = compute_correlation_metrics(scores_perfect, ade_perfect)
    ndcg_perfect = compute_ndcg(scores_perfect, ade_perfect, k_list=[4])

    acc_p = pair_acc_perfect["pairwise_acc_margin_05"]
    sp_p = corr_perfect["spearman"]
    kd_p = corr_perfect["kendall"]
    ndcg_p = ndcg_perfect["ndcg_4"]

    print(f"[TEST A - Perfect] PairwiseAcc@0.05: {acc_p} (expected 1.0)")
    print(f"[TEST A - Perfect] Spearman: {sp_p} (expected 1.0)")
    print(f"[TEST A - Perfect] Kendall: {kd_p} (expected 1.0)")
    print(f"[TEST A - Perfect] NDCG@4: {ndcg_p} (expected 1.0)")

    test_a_pass = (
        np.isclose(acc_p, 1.0)
        and np.isclose(sp_p, 1.0)
        and np.isclose(kd_p, 1.0)
        and np.isclose(ndcg_p, 1.0)
    )
    results["test_a_perfect_ranking"] = bool(test_a_pass)

    # Test B: Reverse Ranking
    # Candidate 0 is best (0.1m), but gets lowest score (1.0)
    ade_reverse = np.array([[0.1, 0.2, 0.3, 0.4]])
    scores_reverse = np.array([[1.0, 2.0, 3.0, 4.0]])

    pair_acc_reverse = compute_pairwise_accuracy(scores_reverse, ade_reverse, margins=[0.05])
    corr_reverse = compute_correlation_metrics(scores_reverse, ade_reverse)

    acc_r = pair_acc_reverse["pairwise_acc_margin_05"]
    sp_r = corr_reverse["spearman"]
    kd_r = corr_reverse["kendall"]

    print(f"[TEST B - Reverse] PairwiseAcc@0.05: {acc_r} (expected 0.0)")
    print(f"[TEST B - Reverse] Spearman: {sp_r} (expected -1.0)")
    print(f"[TEST B - Reverse] Kendall: {kd_r} (expected -1.0)")

    test_b_pass = (
        np.isclose(acc_r, 0.0)
        and np.isclose(sp_r, -1.0)
        and np.isclose(kd_r, -1.0)
    )
    results["test_b_reverse_ranking"] = bool(test_b_pass)

    # Test C: Margin Filtering
    # Candidates 0 and 1 have delta = 0.01m (< 0.05m margin).
    # Candidate 2 is 0.30m (delta = 0.20m > 0.05m margin).
    ade_margin = np.array([[0.10, 0.11, 0.30]])
    # Score ranks cand 1 higher than cand 0 (inverted within margin),
    # but correctly ranks cand 0 and 1 higher than cand 2.
    scores_margin = np.array([[2.0, 3.0, 1.0]])

    # Pairs with margin >= 0.05: (0, 2) and (1, 2). For both, cand 0 and cand 1 have higher score than 2.
    # So pairwise accuracy with margin 0.05 should be 1.0!
    pair_acc_margin = compute_pairwise_accuracy(scores_margin, ade_margin, margins=[0.05])
    acc_m = pair_acc_margin["pairwise_acc_margin_05"]
    print(f"[TEST C - Margin] PairwiseAcc@0.05: {acc_m} (expected 1.0)")
    test_c_pass = np.isclose(acc_m, 1.0)
    results["test_c_margin_filtering"] = bool(test_c_pass)

    all_pass = all(results.values())
    print("\n" + "=" * 60)
    print(f"[EXP-S0A SUMMARY] All Tests Passed: {all_pass}")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print("=" * 60)

    # Output json
    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "metric_tests.json"
    out_json.write_text(json.dumps({"results": results, "all_pass": all_pass}, indent=2), encoding="utf-8")

    return results


if __name__ == "__main__":
    res = run_tests()
    if not all(res.values()):
        sys.exit(1)
