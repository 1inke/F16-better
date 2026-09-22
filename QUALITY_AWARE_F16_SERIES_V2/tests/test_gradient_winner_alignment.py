"""
Unit test for candidate gradient winner alignment.

Enforces specification §26:
Verifies that when winner_orig != winner_cand:
- g_full_orig hard CE component points toward winner_orig
- g_full_cand hard CE component points strictly toward winner_cand
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE)]

from common.gradient_audit import compute_gradient_vectors_all_modes


def test_gradient_winner_alignment():
    # Construct 1 sample with 3 modes
    logits = np.array([[0.0, 0.0, 0.0]])
    q_orig = np.array([[0.5, 0.3, 0.2]])
    q_cand = np.array([[0.2, 0.5, 0.3]])

    # Deliberately different winners:
    # winner_orig is mode 0, winner_cand is mode 1
    winner_orig = np.array([0])
    winner_cand = np.array([1])

    res = compute_gradient_vectors_all_modes(
        logits=logits,
        q_orig=q_orig,
        q_cand=q_cand,
        winner_orig=winner_orig,
        winner_cand=winner_cand,
        hard_weight=0.25,
        cls_weight=0.5,
    )

    g_full_orig, g_full_cand = res["full_objective"]

    # p = [1/3, 1/3, 1/3]
    # For g_full_orig:
    # hard CE term is 0.25 * (p - y_orig), where y_orig = [1, 0, 0]
    # mode 0 has (1/3 - 1) = -2/3, which is strongly negative (gradient pushes logits[0] up)
    # For g_full_cand:
    # hard CE term is 0.25 * (p - y_cand), where y_cand = [0, 1, 0]
    # mode 1 has (1/3 - 1) = -2/3, which is strongly negative (gradient pushes logits[1] up)

    # Check that candidate gradient pulls mode 1 (winner_cand) more than mode 0
    # Negative gradient w.r.t logits means increasing logits decreases loss
    assert g_full_cand[0, 1] < g_full_cand[0, 0], (
        f"Expected g_full_cand[0, 1] < g_full_cand[0, 0], got {g_full_cand[0, 1]} vs {g_full_cand[0, 0]}"
    )
    # Check that orig gradient pulls mode 0 (winner_orig) more than mode 1
    assert g_full_orig[0, 0] < g_full_orig[0, 1], (
        f"Expected g_full_orig[0, 0] < g_full_orig[0, 1], got {g_full_orig[0, 0]} vs {g_full_orig[0, 1]}"
    )

    print("test_gradient_winner_alignment: ALL TESTS PASSED!")


if __name__ == "__main__":
    test_gradient_winner_alignment()
