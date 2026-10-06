import inspect
import unittest

import numpy as np
import torch

import run_formal_regime_selection as exp


class FormalProtocolTests(unittest.TestCase):
    def test_reranker_forward_is_observation_only(self):
        signature = inspect.signature(exp.HistoryCorrectionHead.forward)
        self.assertEqual(list(signature.parameters), ["self", "motion", "candidate", "native_logits", "gate"])

    def test_gate_controls_residual_and_zero_init_is_identity(self):
        model = exp.HistoryCorrectionHead(4, 3, hidden=8, dropout=0.0)
        motion = torch.randn(2, 4)
        candidate = torch.randn(2, 5, 3)
        native = torch.randn(2, 5)
        zeros = torch.zeros(2)
        ones = torch.ones(2)
        torch.testing.assert_close(model(motion, candidate, native, zeros), native)
        torch.testing.assert_close(model(motion, candidate, native, ones), native)

    def test_candidate_features_have_expected_shape_and_are_finite(self):
        past = torch.zeros(2, 1, 8, 2)
        past[:, 0, :, 0] = torch.arange(8) * 0.1
        relative = torch.randn(2, 20, 12, 2) * 0.2
        logits = torch.randn(2, 20)
        features = exp.candidate_features(past, relative, logits)
        self.assertEqual(features.shape, (2, 20, 31))
        self.assertTrue(torch.isfinite(features).all())

    def test_hierarchical_bootstrap_respects_fold_macro(self):
        delta = {"a": np.array([-1.0, -1.0]), "b": np.array([1.0] * 100)}
        point, lower, upper = exp.hierarchical_macro_bootstrap(delta, 200, 7)
        self.assertAlmostEqual(point, 0.0, places=7)
        self.assertLessEqual(lower, point)
        self.assertGreaterEqual(upper, point)

    def test_test_stage_is_order_guarded(self):
        source = inspect.getsource(exp.prepare_test)
        self.assertIn("assert_all_training_locked", source)

    def test_seed_averaged_probability_defines_one_common_quantile_mask(self):
        original = exp._load_sample
        probabilities = {
            1: np.array([0.0, 0.0, 0.0, 1.0]),
            2: np.array([0.1, 0.2, 0.3, 0.4]),
        }

        def fake_load(_fold, seed, name):
            values = np.arange(4, dtype=float) + seed if name == "left" else np.zeros(4)
            return {
                "sample_id": np.arange(4), "selected_ade": values,
                "p_d1": probabilities[seed], "change_group": np.zeros(4, dtype=int),
            }

        try:
            exp._load_sample = fake_load
            result = exp._average_seed_delta(
                {"folds": ["fold"], "head_seeds": [1, 2]},
                "left", "right", "selected_ade", "low",
            )
        finally:
            exp._load_sample = original
        np.testing.assert_allclose(result["fold"], np.array([1.5]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
