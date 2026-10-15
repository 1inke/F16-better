"""Synthetic tests only: no real-data read, fit, solver, Torch, or experiment.

Manual invocation: python -B -m unittest -v test_core.py
"""
import unittest
import numpy as np
from core_matching import (build_map, transform, landmark_rows, rbf_values,
                           witness_supervision, objective_grad, rank4, Selector,
                           model_arrays, restore_model, predict, HISTORY, CANDIDATE)
from input_contract import check_slots, split_masks
from evaluation import found_hit, describe, paired_track_ci, evaluate_screen


def fixture():
    rng = np.random.default_rng(7162)
    x = rng.normal(size=(8, 20, 136)).astype(np.float32)
    x[:, :, :96] = x[:, :1, :96]
    x[:, :, 120:124] = x[:, :1, 120:124]
    y = np.zeros((8, 20), np.int8)
    y[:, 2], y[:, 19] = 1, -1
    return x, y, np.repeat(np.arange(4), 2), np.arange(200, 208)


class CoreTests(unittest.TestCase):
    def test_shared_statistics_match_r1_reduction(self):
        x, _, _, ids = fixture()
        a, b = build_map(x, ids, "linear"), build_map(x, ids, "rbf64")
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.std, b.std)
        np.testing.assert_array_equal(a.mean, x.astype(float).mean((0, 1)))
        self.assertEqual(transform(x, b).shape, (8, 20, 200))

    def test_landmark_identity_not_row_order(self):
        _, _, _, ids = fixture()
        perm = np.array([7, 2, 0, 5, 1, 6, 4, 3])
        _, a = landmark_rows(ids)
        _, b = landmark_rows(ids[perm])
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(map(tuple, a.tolist()))), 64)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            landmark_rows(np.array([1, 1, 2, 3]))

    def test_rbf_independent_scalar_and_chunks(self):
        rng = np.random.default_rng(23)
        z, centers = rng.normal(size=(5, 136)), rng.normal(size=(64, 136))
        actual = rbf_values(z, centers, chunk=3)
        expected = np.empty_like(actual)
        for i in range(len(z)):
            for j in range(64):
                dh = sum((z[i, k] - centers[j, k]) ** 2 for k in HISTORY) / 100
                dq = sum((z[i, k] - centers[j, k]) ** 2 for k in CANDIDATE) / 36
                expected[i, j] = np.exp(-0.5 * (dh + dq))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(actual, rbf_values(z, centers, chunk=1))

    def test_rbf_can_history_condition_ranking(self):
        centers = np.zeros((64, 136))
        centers[0, [0, 96]], centers[1, [0, 96]] = [-10, -6], [10, 6]
        rows = np.zeros((4, 136))
        rows[:, 0], rows[:, 96] = [-10, -10, 10, 10], [-6, 6, -6, 6]
        score = rbf_values(rows, centers)[:, :2].sum(1)
        self.assertGreater(score[0], score[1])
        self.assertGreater(score[3], score[2])

    def test_validation_does_not_mutate_map(self):
        x, _, _, ids = fixture()
        mapping = build_map(x[:4], ids[:4], "rbf64")
        before = [v.copy() for v in (mapping.mean, mapping.std, mapping.centers, mapping.rbf_mean)]
        transform(x[4:] + 1000, mapping)
        for a, b in zip(before, (mapping.mean, mapping.std, mapping.centers, mapping.rbf_mean)):
            np.testing.assert_array_equal(a, b)

    def test_training_weights_and_unknown(self):
        _, y, _, _ = fixture()
        tracks = np.array([0, 0, 0, 0, 1, 1, 2, 3])
        target, known, weight = witness_supervision(y, tracks)
        self.assertTrue(np.all(weight[y == -1] == 0))
        np.testing.assert_array_equal(known, y != -1)
        self.assertAlmostEqual(weight.sum(), 160)
        self.assertAlmostEqual(weight[target == 1].sum(), 80)
        self.assertAlmostEqual(weight[y == 0].sum(), 80)
        for tr in range(4):
            self.assertAlmostEqual(weight[tracks == tr].sum(), 40)

    def test_objective_gradient_finite_difference(self):
        rng = np.random.default_rng(7)
        x, y, sw = rng.normal(size=(17, 9)), np.arange(17) % 2, rng.uniform(0.1, 2, 17)
        vector, direction = rng.normal(size=10), rng.normal(size=10)
        value, grad = objective_grad(x, y, sw, vector[:-1], vector[-1])
        eps = 1e-5
        plus, minus = vector + eps * direction, vector - eps * direction
        fd = (objective_grad(x, y, sw, plus[:-1], plus[-1])[0]
              - objective_grad(x, y, sw, minus[:-1], minus[-1])[0]) / (2 * eps)
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(fd, float(grad @ direction), delta=1e-8)

    def test_ties_use_physical_first_and_reject_bad_slots(self):
        self.assertEqual(rank4(np.zeros((1, 20))).tolist(), [[0, 1, 2, 3]])
        with self.assertRaises(ValueError):
            check_slots(np.array([[0, 0, 1, 2]]), 1)
        with self.assertRaises(ValueError):
            rank4(np.full((1, 20), np.nan))

    def test_unknown_not_positive_or_negative_for_bounds(self):
        y = np.zeros((2, 20), dtype=np.int8)
        y[0, 0], y[1, 0] = -1, 1
        lower, upper, op = found_hit(np.array([[0, 1, 2, 3]] * 2), y)
        self.assertEqual(lower.tolist(), [False, True])
        self.assertEqual(upper.tolist(), [True, True])
        self.assertEqual(op.tolist(), [False, True])

    def test_restore_scores_exact_without_fit(self):
        x, _, _, ids = fixture()
        for arm, dim in (("linear", 136), ("rbf64", 200)):
            mapping = build_map(x, ids, arm)
            model = Selector(mapping, np.sin(np.arange(dim)), 0.4)
            restored = restore_model(model_arrays(model))
            np.testing.assert_array_equal(predict(x, model), predict(x, restored))
        payload = model_arrays(model)
        payload["coef"][0] = np.nan
        with self.assertRaises(ValueError):
            restore_model(payload)

    def test_track_leak_rejected(self):
        d = dict(fold=np.array([0, 1]), track=np.array(["same", "same"]),
                 labels=np.tile([1] + [0] * 19, (2, 1)))
        with self.assertRaises(ValueError):
            split_masks(d, 0)

    def test_cluster_bootstrap_and_descriptive_rank(self):
        tracks = np.array(["a", "a", "b", "c"])
        ci = paired_track_ci(tracks, np.ones(4, bool), np.ones(4), 23)
        np.testing.assert_array_equal(ci["CI9875"], [1, 1])
        self.assertEqual(ci["tracks"], 3)
        y = np.zeros((1, 20), dtype=np.int8)
        y[0, 19] = 1
        report = describe(np.array([[0, 1, 2, 3]]), y, np.arange(20, dtype=float)[None])
        self.assertEqual(report["best_P_rank_median"], 1)
        self.assertEqual(report["P_greater_N_pair_fraction"], 1)

    def test_screen_keeps_four_references_and_no_auto_launch(self):
        y = np.tile([1] + [0] * 19, (90, 1)).astype(np.int8)
        d = dict(labels=y, track=np.arange(90), fold=np.repeat([0, 1, 2], 30), groups=np.full(90, 2))
        good, bad = np.tile([0, 1, 2, 3], (90, 1)), np.tile([1, 2, 3, 4], (90, 1))
        policies = dict(rbf64=good, linear=bad, envelope=bad, fixed_random=bad, bottom=bad)
        report = evaluate_screen(d, policies)
        self.assertTrue(report["statistical_screen_only"])
        self.assertEqual(len(report["comparisons"]), 4)
        self.assertFalse(report["B_started"])
        self.assertFalse(report["S2_started"])
        policies["linear"] = good
        self.assertFalse(evaluate_screen(d, policies)["statistical_screen_only"])


if __name__ == "__main__":
    unittest.main()
