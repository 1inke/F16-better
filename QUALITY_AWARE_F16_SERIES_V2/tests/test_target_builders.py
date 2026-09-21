"""
Unit tests for confidence target builders.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from QUALITY_AWARE_F16_SERIES_V2.common.confidence_targets import (
    build_original_target,
    build_fixed_quality_target,
    build_fixed_ade_target,
)


def test_target_shapes_and_properties():
    B, N, K, T, D = 4, 1, 20, 12, 2
    pred = torch.randn(B, N, K, T, D, requires_grad=True)
    target = torch.randn(B, N, T, D)

    # 1. Original target
    q_orig, winner_orig, qual_orig = build_original_target(pred, target)
    assert q_orig.shape == (B, N, K)
    assert winner_orig.shape == (B, N)
    assert not q_orig.requires_grad, "q_orig must be strictly detached"
    assert torch.allclose(q_orig.sum(dim=-1), torch.ones(B, N), atol=1e-6)
    assert (q_orig >= 0).all()

    # 2. Fixed quality target
    q_fix, winner_fix, qual_fix = build_fixed_quality_target(pred, target, tau=0.10)
    assert q_fix.shape == (B, N, K)
    assert winner_fix.shape == (B, N)
    assert not q_fix.requires_grad, "q_fix must be strictly detached"
    assert torch.allclose(q_fix.sum(dim=-1), torch.ones(B, N), atol=1e-6)
    assert (winner_orig == winner_fix).all(), "Winner should match since both use ADE + 0.25*FDE"

    # 3. Fixed ADE target
    q_ade, winner_ade, ade = build_fixed_ade_target(pred, target, tau=0.10)
    assert q_ade.shape == (B, N, K)
    assert not q_ade.requires_grad, "q_ade must be strictly detached"
    assert torch.allclose(q_ade.sum(dim=-1), torch.ones(B, N), atol=1e-6)

    # 4. Temperature sensitivity: low tau should peak at winner
    q_sharp, _, _ = build_fixed_quality_target(pred, target, tau=0.001)
    sharp_winner = q_sharp.argmax(dim=-1)
    assert (sharp_winner == winner_fix).all()
    # High tau should approach 1/K
    q_flat, _, _ = build_fixed_quality_target(pred, target, tau=100.0)
    assert torch.allclose(q_flat, torch.full_like(q_flat, 1.0 / K), atol=1e-2)

    print("[TEST] test_target_shapes_and_properties PASSED!")


if __name__ == "__main__":
    test_target_shapes_and_properties()
