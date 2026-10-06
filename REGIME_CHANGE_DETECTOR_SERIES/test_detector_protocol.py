import inspect
import unittest

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import run_detector_experiment as experiment


class DetectorProtocolTests(unittest.TestCase):
    def test_change_score_expected_directions(self):
        past = torch.zeros(2, 1, 8, 2)
        past[:, 0, :, 0] = torch.arange(8)[None] * 0.1
        future = torch.zeros(2, 1, 12, 2)
        future[0, 0, :, 0] = 0.7 + torch.arange(1, 13) * 0.1
        future[1, 0, :, 1] = torch.arange(1, 13) * 0.1
        score = experiment.future_change_score(past, future)
        self.assertLess(float(score[0]), 0.25)
        self.assertGreaterEqual(float(score[1]), 0.5)

    def test_detector_forward_accepts_observation_features_only(self):
        signature = inspect.signature(experiment.DetectorMLP.forward)
        self.assertEqual(list(signature.parameters), ["self", "observed_features"])
        model = experiment.DetectorMLP(12, 0.1)
        self.assertEqual(model(torch.randn(5, 12)).shape, (5,))

    def test_exact_split_contract(self):
        split = experiment.exact_splits(experiment.load_spec())
        self.assertEqual(len(split["validation"]), 1516)
        self.assertEqual(np.intersect1d(split["train"], split["dev"]).size, 0)
        self.assertEqual(np.intersect1d(split["train"], split["validation"]).size, 0)
        self.assertEqual(np.intersect1d(split["dev"], split["validation"]).size, 0)

    def test_metric_direction(self):
        labels = np.array([0, 0, 1, 1])
        good = np.array([0.1, 0.2, 0.8, 0.9])
        bad = 1.0 - good
        self.assertGreater(roc_auc_score(labels, good), 0.5)
        self.assertLess(roc_auc_score(labels, bad), 0.5)

    def test_probability_bins_are_complete(self):
        probability = np.linspace(0.0, 1.0, 23)
        order = np.argsort(probability, kind="stable")
        assigned = np.concatenate(np.array_split(order, 5))
        np.testing.assert_array_equal(np.sort(assigned), np.arange(len(probability)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
