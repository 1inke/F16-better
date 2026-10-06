#!/usr/bin/env python3
"""
Unit test for Oracle Coverage & Chance Hit Diagnostics.
Validates metric definitions specified in introdcion.md:
- R0, R1, R2 regret bucket boundaries
- A, B, C quality breakdown inside R2
- Delta ADE 1,2 (isolation gap)
- GT endpoint support at 0.5m and 1.0m
- Consistency across synthetic edge-case arrays
"""

import unittest
import numpy as np


def compute_metrics_for_array(ade_final, fde_final, native_logits, future_relative, candidate_final, selected_idx, oracle_idx):
    N, K = ade_final.shape
    row = np.arange(N)

    regret = ade_final[row, selected_idx] - ade_final[row, oracle_idx]
    ade_min = ade_final[row, oracle_idx]
    
    # Bucket masks
    r0_mask = regret < 0.20
    r1_mask = (regret >= 0.20) & (regret < 0.50)
    r2_mask = regret >= 0.50

    a_mask = r2_mask & (ade_min < 0.50)
    b_mask = r2_mask & (ade_min >= 0.50) & (ade_min < 1.00)
    c_mask = r2_mask & (ade_min >= 1.00)

    # Isolation gap
    ade_sorted = np.sort(ade_final, axis=1)
    delta_ade_12 = ade_sorted[:, 1] - ade_sorted[:, 0]

    # Endpoint support
    gt_end = future_relative[:, -1, :]
    cand_end = candidate_final[:, :, -1, :]
    end_err = np.linalg.norm(cand_end - gt_end[:, None, :], axis=-1)
    supp_05 = np.sum(end_err < 0.5, axis=1)
    supp_10 = np.sum(end_err < 1.0, axis=1)

    return {
        "regret": regret,
        "r0_mask": r0_mask,
        "r1_mask": r1_mask,
        "r2_mask": r2_mask,
        "a_mask": a_mask,
        "b_mask": b_mask,
        "c_mask": c_mask,
        "delta_ade_12": delta_ade_12,
        "supp_05": supp_05,
        "supp_10": supp_10,
    }


class TestOracleCoverageMetrics(unittest.TestCase):
    def test_bucket_classification(self):
        # 3 synthetic samples
        ade = np.array([
            [0.10, 0.25],  # Sample 0: sel=0, ora=0 -> regret=0.0 -> R0
            [0.50, 0.15],  # Sample 1: sel=0, ora=1 -> regret=0.35 -> R1
            [1.20, 0.30],  # Sample 2: sel=0, ora=1 -> regret=0.90 -> R2, ade_min=0.30 -> A
        ])
        fde = np.zeros_like(ade)
        logits = np.zeros_like(ade)
        future = np.zeros((3, 12, 2))
        cands = np.zeros((3, 2, 12, 2))
        sel = np.array([0, 0, 0])
        ora = np.array([0, 1, 1])

        res = compute_metrics_for_array(ade, fde, logits, future, cands, sel, ora)
        self.assertTrue(res["r0_mask"][0])
        self.assertTrue(res["r1_mask"][1])
        self.assertTrue(res["r2_mask"][2])
        self.assertTrue(res["a_mask"][2])

    def test_isolation_gap(self):
        ade = np.array([
            [0.21, 0.91, 1.15],
            [0.21, 0.25, 0.31]
        ])
        ade_sorted = np.sort(ade, axis=1)
        delta_12 = ade_sorted[:, 1] - ade_sorted[:, 0]
        self.assertAlmostEqual(delta_12[0], 0.70, places=5)
        self.assertAlmostEqual(delta_12[1], 0.04, places=5)

    def test_endpoint_support(self):
        # GT endpoint at (0, 0)
        future = np.zeros((1, 12, 2))
        # Candidates endpoints: one at (0.2, 0), one at (0.8, 0), one at (2.0, 0)
        cands = np.zeros((1, 3, 12, 2))
        cands[0, 0, -1] = [0.2, 0.0]  # dist=0.2 < 0.5, < 1.0
        cands[0, 1, -1] = [0.8, 0.0]  # dist=0.8 >= 0.5, < 1.0
        cands[0, 2, -1] = [2.0, 0.0]  # dist=2.0 >= 1.0

        gt_end = future[:, -1, :]
        cand_end = cands[:, :, -1, :]
        end_err = np.linalg.norm(cand_end - gt_end[:, None, :], axis=-1)
        supp_05 = np.sum(end_err < 0.5, axis=1)[0]
        supp_10 = np.sum(end_err < 1.0, axis=1)[0]

        self.assertEqual(supp_05, 1)
        self.assertEqual(supp_10, 2)


if __name__ == "__main__":
    unittest.main()
