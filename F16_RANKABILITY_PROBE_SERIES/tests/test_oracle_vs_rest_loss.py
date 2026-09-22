"""
Unit test for Oracle-vs-Rest pairwise loss.

Enforces specification §26:
- Verifies that when s_{k*} is much higher than all competitors, loss -> 0
- Verifies that when s_{k*} is much lower than competitors, loss is positive and large
- Verifies that competitors with delta < tie_threshold do not contribute to loss
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE)]

from common.pairwise_objective import OracleVsRestPairwiseLoss, ListwiseTop1Loss


def test_oracle_vs_rest_loss():
    crit = OracleVsRestPairwiseLoss(tie_threshold=0.02, max_diff_scale=0.20)

    # Case 1: mode 0 is oracle (ADE 0.10), competitors are 0.30, 0.50
    ades = torch.tensor([[0.10, 0.30, 0.50]])
    # Scores where oracle is ranked much higher: loss should be small
    scores_good = torch.tensor([[50.0, -10.0, -20.0]])
    loss_good = crit(scores_good, ades)
    assert loss_good.item() < 1e-4, f"Expected near zero loss, got {loss_good.item()}"

    # Scores where oracle is ranked lower: loss should be large
    scores_bad = torch.tensor([[-20.0, 10.0, 20.0]])
    loss_bad = crit(scores_bad, ades)
    assert loss_bad.item() > 10.0, f"Expected large loss, got {loss_bad.item()}"

    # Case 2: Near-tie competitor should be filtered
    ades_tie = torch.tensor([[0.10, 0.11]])  # delta is 0.01 (< 0.02)
    scores_tie = torch.tensor([[0.0, 10.0]])
    loss_tie = crit(scores_tie, ades_tie)
    # Since only valid weight sum is 0 (clamped to 1), loss sum should be 0.0
    assert loss_tie.item() == 0.0, f"Expected 0.0 for near tie pair, got {loss_tie.item()}"

    # Case 3: ListwiseTop1Loss test
    l_hard = ListwiseTop1Loss(tau=None)
    loss_ce = l_hard(scores_good, ades)
    assert loss_ce.item() < 1e-4, f"Expected small CE loss for good scores, got {loss_ce.item()}"

    l_soft = ListwiseTop1Loss(tau=0.10)
    loss_soft = l_soft(scores_good, ades)
    assert loss_soft.item() > 0.0, f"Expected positive soft CE loss, got {loss_soft.item()}"

    print("test_oracle_vs_rest_loss: ALL TESTS PASSED!")


if __name__ == "__main__":
    test_oracle_vs_rest_loss()
