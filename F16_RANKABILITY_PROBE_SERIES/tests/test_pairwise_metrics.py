"""
Unit tests for pairwise accuracy and ranking metrics.

Enforces specification §2.3:
- Case 1: Perfect ranking -> 1.0
- Case 2: Complete inversion -> 0.0
- Case 3: Threshold filtering -> delta < min_delta excluded
- Case 4: Oracle margin accuracy
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE)]

from metrics import compute_pairwise_accuracy, compute_oracle_margin_accuracy


def test_pairwise_accuracy():
    # Case 1: Perfect ranking
    # ADE: mode 0 (0.10) is best, mode 1 (0.20) middle, mode 2 (0.30) worst
    ades_perfect = np.array([[0.10, 0.20, 0.30]])
    scores_perfect = np.array([[3.0, 2.0, 1.0]])
    acc_1 = compute_pairwise_accuracy(scores_perfect, ades_perfect, min_delta=0.02)
    assert np.isclose(acc_1, 1.0), f"Expected 1.0, got {acc_1}"

    # Case 2: Complete inversion
    scores_inverted = np.array([[1.0, 2.0, 3.0]])
    acc_2 = compute_pairwise_accuracy(scores_inverted, ades_perfect, min_delta=0.02)
    assert np.isclose(acc_2, 0.0), f"Expected 0.0, got {acc_2}"

    # Case 3: Threshold filtering
    # Modes 0 and 1 differ by only 0.01 (< 0.02) -> should be excluded
    # Modes 0 and 2 differ by 0.20 (> 0.02)
    # Modes 1 and 2 differ by 0.19 (> 0.02)
    ades_near_tie = np.array([[0.10, 0.11, 0.30]])
    # Score for mode 0 vs mode 1 is wrong (1 < 2), but mode 0 vs 2 and mode 1 vs 2 are right
    scores_tie = np.array([[2.0, 3.0, 1.0]])
    acc_3 = compute_pairwise_accuracy(scores_tie, ades_near_tie, min_delta=0.02)
    # Valid pairs with delta > 0.02: (0, 2) and (1, 2). Both have higher scores for better mode.
    assert np.isclose(acc_3, 1.0), f"Expected 1.0 after filtering near tie, got {acc_3}"

    # Case 4: Oracle margin accuracy
    ades = np.array([[0.10, 0.15, 0.35]])
    oracle_idx = np.array([0])
    # Competitor 1 has delta 0.05 (< 0.10 margin)
    # Competitor 2 has delta 0.25 (> 0.10 margin)
    scores_ora_win = np.array([[2.0, 1.0, 0.5]])
    margin_acc = compute_oracle_margin_accuracy(scores_ora_win, ades, oracle_idx, margin_threshold=0.10)
    assert np.isclose(margin_acc, 1.0), f"Expected 1.0, got {margin_acc}"

    scores_ora_lose = np.array([[0.1, 1.0, 2.0]])
    margin_acc_lose = compute_oracle_margin_accuracy(scores_ora_lose, ades, oracle_idx, margin_threshold=0.10)
    assert np.isclose(margin_acc_lose, 0.0), f"Expected 0.0, got {margin_acc_lose}"

    print("test_pairwise_metrics: ALL TESTS PASSED!")


if __name__ == "__main__":
    test_pairwise_accuracy()
