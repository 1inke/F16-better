"""
Unit tests for probability_metrics and quality_targets (Rule 5: Metric Unit Tests).
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from QUALITY_AWARE_F16_SERIES.common.probability_metrics import (
    compute_generation_metrics,
    compute_selection_metrics,
    compute_pairwise_accuracy,
    compute_probability_metrics,
)
from QUALITY_AWARE_F16_SERIES.common.quality_targets import (
    build_quality_target_ade,
)


def test_pairwise_accuracy_direction():
    # 2 samples, 4 candidates
    # ade: sample 0 has [0.1, 0.2, 0.3, 0.4] -> candidate 0 is best
    ade = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ])
    # Perfect score: higher score for lower ADE
    perfect_scores = np.array([
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
    ])
    acc_perf = compute_pairwise_accuracy(perfect_scores, ade, delta=0.05)
    assert acc_perf == 1.0, f"Expected 1.0 for perfect ranking, got {acc_perf}"

    # Reverse score: higher score for higher ADE
    reverse_scores = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
    ])
    acc_rev = compute_pairwise_accuracy(reverse_scores, ade, delta=0.05)
    assert acc_rev == 0.0, f"Expected 0.0 for reverse ranking, got {acc_rev}"


def test_probability_metrics_ideal():
    # 1 sample, 3 candidates
    ade = np.array([[0.1, 0.3, 0.5]])
    # Perfect probability (one-hot on best candidate 0)
    perf_prob = np.array([[1.0, 0.0, 0.0]])

    m_perf = compute_probability_metrics(perf_prob, ade, tau_eval=0.10)
    # Oracle NLL should be -log(1.0) = 0.0
    assert abs(m_perf["oracle_nll"]) < 1e-5
    # Brier score should be 0.0
    assert abs(m_perf["brier_score"]) < 1e-5
    # Entropy should be 0.0
    assert abs(m_perf["mean_entropy"]) < 1e-5
    assert m_perf["mean_max_prob"] == 1.0


def test_quality_targets_detach():
    cand = torch.randn(2, 20, 12, 2, requires_grad=True)
    gt = torch.randn(2, 12, 2)
    q = build_quality_target_ade(cand, gt, tau=0.10)
    assert not q.requires_grad, "Quality target must not require grad!"
    assert q.shape == (2, 20)
    assert torch.allclose(q.sum(dim=-1), torch.ones(2))


if __name__ == "__main__":
    test_pairwise_accuracy_direction()
    test_probability_metrics_ideal()
    test_quality_targets_detach()
    print("[UNIT TESTS PASSED] All probability and quality target unit tests verified successfully!")
