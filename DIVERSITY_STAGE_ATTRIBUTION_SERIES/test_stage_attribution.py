from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_stage_attribution as analysis
import run_stage_attribution as runner


class StageAttributionTests(unittest.TestCase):
    def test_config_contract(self) -> None:
        audit = runner.assert_config_contract()
        self.assertEqual(audit["allowed_difference"], "MODEL.DECODER.DIVERSITY_WEIGHT")
        self.assertEqual(float(audit["official_value"]), 0.05)
        self.assertEqual(float(audit["div000_value"]), 0.0)

    def test_diversity_activity_uses_190_pairs(self) -> None:
        candidate = np.zeros((1, 20, 12, 2), dtype=np.float32)
        candidate[0, :, -1, 0] = np.arange(20, dtype=np.float32)
        raw = {
            "sample_id": np.array([7]),
            "candidate": candidate,
            "ade": np.tile(np.arange(20, dtype=np.float32), (1, 1)),
            "fde": np.tile(np.arange(20, dtype=np.float32), (1, 1)),
            "logits": np.tile(-np.arange(20, dtype=np.float32), (1, 1)),
        }
        values = analysis.derive(raw, margin=0.4)
        self.assertEqual(values["active_pairs_count"][0], 0)
        self.assertAlmostEqual(values["endpoint_spread"][0], 7.0, places=6)

    def test_zero_endpoints_activate_all_pairs(self) -> None:
        raw = {
            "sample_id": np.array([1]),
            "candidate": np.zeros((1, 20, 12, 2), dtype=np.float32),
            "ade": np.ones((1, 20), dtype=np.float32),
            "fde": np.ones((1, 20), dtype=np.float32),
            "logits": np.zeros((1, 20), dtype=np.float32),
        }
        values = analysis.derive(raw, margin=0.4)
        self.assertEqual(values["active_pairs_count"][0], 190)
        self.assertAlmostEqual(values["active_pair_rate_sample"][0], 1.0)
        self.assertAlmostEqual(values["diversity_penalty"][0], 0.4, places=6)

    def test_bootstrap_is_deterministic(self) -> None:
        delta = np.array([-1.0, 0.0, 1.0, 2.0])
        first = analysis.bootstrap_mean_ci(delta, repetitions=100, seed=11)
        second = analysis.bootstrap_mean_ci(delta, repetitions=100, seed=11)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
