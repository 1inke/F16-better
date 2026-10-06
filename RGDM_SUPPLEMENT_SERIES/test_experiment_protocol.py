import json
import unittest

import numpy as np

import analyze_rgdm_factorial as analysis
import run_rgdm_factorial as runner


class ProtocolTests(unittest.TestCase):
    def test_matrix_is_exact_2x2(self):
        self.assertEqual(list(analysis.CONDITIONS), [
            "c00_g0_s0", "s_only_g0_s1", "g_only_g1_s0", "joint_g1_s1"
        ])
        spec = runner.load_spec()
        self.assertEqual(set(spec["conditions"]), set(list(analysis.CONDITIONS)[1:]))
        self.assertFalse(spec["test_access_authorized"])
        self.assertEqual(spec["gate_loss_weight"], 0.0)
        self.assertEqual(spec["rank_loss_weight"], 0.0)

    def test_training_command_has_no_test_path(self):
        spec = runner.load_spec()
        command = runner.command("s_only_g0_s1", "rgdm_b4_full", spec, "cuda:0", False)
        joined = " ".join(command).lower()
        self.assertNotIn("--test-data-path", joined)
        self.assertNotIn("--test-num-path", joined)
        self.assertIn("--validation-split-mode scene_tail", joined)

    def test_completed_selection_contract_is_verified(self):
        spec = runner.load_spec()
        run = runner.validate_completed_run("s_only_g0_s1", "rgdm_b4_full", spec)
        self.assertTrue((run / "metrics.json").is_file())

    def test_factorial_effect_signs(self):
        # C=0, S=-1, G=-2, GS=-4 -> S main=-1.5, G main=-2.5, interaction=-1.
        cells = {}
        for name, value in zip(analysis.CONDITIONS, (0.0, -1.0, -2.0, -4.0)):
            cells[name] = {metric: np.full(20, value) for metric in analysis.METRICS}
        rows = list(analysis.paired_bootstrap_factorial(
            cells, np.ones(20, dtype=bool), repetitions=100, seed=7
        ))
        selected = {(row["effect"], row["metric"]): row for row in rows}
        self.assertAlmostEqual(selected[("selection_main", "selected_ade")]["estimate"], -1.5)
        self.assertAlmostEqual(selected[("generation_main", "selected_ade")]["estimate"], -2.5)
        self.assertAlmostEqual(selected[("interaction", "selected_ade")]["estimate"], -1.0)

    def test_per_sample_metric_direction(self):
        cache = {
            "ade": np.array([[0.1, 0.5, 1.2]], dtype=np.float32),
            "fde": np.array([[0.2, 0.7, 1.5]], dtype=np.float32),
            "logits": np.array([[0.0, 2.0, 1.0]], dtype=np.float32),
            "endpoints": np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], dtype=np.float32),
        }
        values = analysis.per_sample(cache)
        self.assertAlmostEqual(values["selected_ade"][0], 0.5)
        self.assertAlmostEqual(values["minade"][0], 0.1)
        self.assertAlmostEqual(values["regret"][0], 0.4)
        self.assertEqual(values["recall_at_1"][0], 0.0)
        self.assertEqual(values["recall_at_3"][0], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
