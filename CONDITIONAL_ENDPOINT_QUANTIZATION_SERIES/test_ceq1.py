from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_ceq1 as runner
from ceq_core import build_observation_features, weighted_lloyd


class CEQ1Tests(unittest.TestCase):
    def test_observation_features_are_rigid_transform_invariant(self) -> None:
        coordinates = np.zeros((2, 3, 20, 2), dtype=np.float32)
        time = np.arange(20, dtype=np.float32)
        coordinates[:, 0, :, 0] = 0.12 * time
        coordinates[:, 0, :, 1] = 0.01 * time * time
        coordinates[:, 1] = coordinates[:, 0] + np.array([0.5, 0.8], np.float32)
        counts = np.array([2, 2], dtype=np.int64)
        angle = 0.73
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        transformed = coordinates @ rotation.T + np.array([7.0, -3.0], np.float32)
        feature_a, future_a = build_observation_features(coordinates, counts, 8, 12)
        feature_b, future_b = build_observation_features(transformed, counts, 8, 12)
        np.testing.assert_allclose(feature_a, feature_b, atol=2.0e-5, rtol=2.0e-5)
        np.testing.assert_allclose(future_a, future_b, atol=2.0e-5, rtol=2.0e-5)

    def test_weighted_lloyd_probability_mass_drives_score(self) -> None:
        trajectories = torch.zeros(1, 4, 3, 2)
        trajectories[0, 1, :, 0] = 5.0
        trajectories[0, 2, :, 0] = 0.1
        trajectories[0, 3, :, 0] = 5.1
        weights = torch.tensor([[0.40, 0.30, 0.20, 0.10]])
        candidates, mass = weighted_lloyd(
            trajectories, weights, components=2, iterations=5
        )
        self.assertEqual(tuple(candidates.shape), (1, 2, 3, 2))
        torch.testing.assert_close(mass.sum(dim=1), torch.ones(1))
        self.assertGreater(float(mass.max()), 0.55)
        selected = int(mass.argmax(dim=1))
        self.assertLess(float(candidates[0, selected, -1, 0]), 1.0)

    def test_gate_advances_only_joint_geometry_and_selection_gain(self) -> None:
        metrics = {
            "all": {
                "j": -0.03, "minade": -0.025, "minfde": -0.02,
                "selected_ade": -0.04, "support_1p0": 0.40,
            },
            "high_motion": {"j": -0.02},
        }
        rows = []
        for subset, values in metrics.items():
            for metric, value in values.items():
                rows.append({
                    "subset": subset, "metric": metric,
                    "relative_effect": value if metric != "support_1p0" else 0.10,
                    "estimate": value,
                })
        ceq_summary = {"all": {"duplicate_pair_rate": 0.02}}
        self.assertEqual(
            runner.classify(rows, ceq_summary)[0],
            "CEQ1_ADVANCE_TO_NEURAL_DISTILLATION",
        )
        for row in rows:
            if row["subset"] == "all" and row["metric"] == "selected_ade":
                row["relative_effect"] = 0.01
        self.assertEqual(
            runner.classify(rows, ceq_summary)[0], "CEQ1_ORACLE_SIGNAL_ONLY"
        )

    def test_protocol_has_no_test_data_dependency(self) -> None:
        audit = runner.preflight(runner.ASTRA_DEFAULT)
        self.assertFalse(audit["new_neural_training"])
        self.assertFalse(audit["test_paths_read_or_passed"])
        train, validation = runner.split_ids()
        self.assertEqual(np.intersect1d(train, validation).size, 0)


if __name__ == "__main__":
    unittest.main()
