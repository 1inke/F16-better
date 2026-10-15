"""Synthetic-only unittest: NO project data, fit, solver, or worker.
Run: python -B -m unittest -v test_core
"""
import inspect
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from core_ranking import (build_pairs, symmetric_rows, pair_objective, logistic_objective,
                          pointwise_weights, rank4, score, RankModel)
from evaluation import hit_arrays, describe, paired_cis, evaluate_screen, REFERENCES


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.limit = threadpool_limits(limits=1)
        self.limit.__enter__()
        rng = np.random.default_rng(31)
        self.z = rng.normal(size=(3, 20, 200)) * .2
        self.z[:, :, :96] = self.z[:, :1, :96]
        self.z[:, :, 120:124] = self.z[:, :1, 120:124]
        self.y = np.zeros((3, 20), dtype=np.int8)
        self.y[0, :2], self.y[1, :1], self.y[2, :3] = 1, 1, 1
        self.y[:, 19] = -1
        self.tracks = np.array([10, 10, 20])
        self.w = rng.normal(size=200) * .02
        self.guard = lambda: None  # synthetic fixture only, NEVER a production guard
        self.pairs = build_pairs(self.z, self.y, self.tracks, guard=self.guard)

    def tearDown(self):
        self.limit.__exit__(None, None, None)

    def test_pointwise_balance(self):
        target, known, sw = pointwise_weights(self.y, self.tracks)
        self.assertAlmostEqual(sw.sum(), 60)
        self.assertAlmostEqual(sw[self.tracks == 10].sum(), 30)
        self.assertAlmostEqual(sw[self.tracks == 20].sum(), 30)
        np.testing.assert_allclose((sw * target).sum(1), sw.sum(1) / 2)
        self.assertTrue((sw[~known] == 0).all())

    def test_pair_mass_and_indices(self):
        _, _, sw = pointwise_weights(self.y, self.tracks)
        np.testing.assert_allclose(np.bincount(self.pairs.window, weights=self.pairs.weight), sw.sum(1))
        self.assertTrue((self.y[self.pairs.window, self.pairs.positive] == 1).all())
        self.assertTrue((self.y[self.pairs.window, self.pairs.negative] == 0).all())
        self.assertEqual(len(self.pairs.delta), 2 * 17 + 1 * 18 + 3 * 16)

    def test_symmetric_exact_objective_gradient_hessian(self):
        x, y, sw = symmetric_rows(self.pairs)
        direct = pair_objective(self.pairs, self.w)
        binary = logistic_objective(x, y, sw, self.w, fit_intercept=False)
        for a, b in zip(direct, binary):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
        self.assertAlmostEqual(sw.sum(), self.pairs.weight.sum())

    def test_independent_loop_value(self):
        _, _, sw = pointwise_weights(self.y, self.tracks)
        expected = 0.0
        for i, row in enumerate(self.y):
            vals = []
            for p in np.flatnonzero(row == 1):
                for n in np.flatnonzero(row == 0):
                    vals.append(np.logaddexp(0.0, -np.dot(self.z[i, p] - self.z[i, n], self.w)))
            expected += sw[i].sum() / sw.sum() * np.mean(vals)
        expected += np.dot(self.w, self.w) / (2 * sw.sum())
        self.assertAlmostEqual(pair_objective(self.pairs, self.w)[0], expected, places=13)

    def test_pair_directional_gradient_and_hessian_FD(self):
        v = np.random.default_rng(77).normal(size=200)
        v /= np.linalg.norm(v)
        _, g, h = pair_objective(self.pairs, self.w)
        for eps in (1e-4, 1e-5):
            plus = pair_objective(self.pairs, self.w + eps * v)
            minus = pair_objective(self.pairs, self.w - eps * v)
            self.assertLess(abs((plus[0] - minus[0]) / (2 * eps) - g @ v), 1e-8)
            np.testing.assert_allclose((plus[1] - minus[1]) / (2 * eps), h @ v, rtol=0, atol=1e-8)

    def test_pointwise_directional_FD(self):
        y, known, sw = pointwise_weights(self.y, self.tracks)
        args = self.z[known], y[known], sw[known]
        theta = np.r_[self.w, .13]
        v = np.random.default_rng(78).normal(size=201)
        v /= np.linalg.norm(v)
        def obj(t):
            return logistic_objective(*args, t[:-1], t[-1])
        _, g, h = obj(theta)
        for eps in (1e-4, 1e-5):
            plus, minus = obj(theta + eps * v), obj(theta - eps * v)
            self.assertLess(abs((plus[0] - minus[0]) / (2 * eps) - g @ v), 1e-8)
            np.testing.assert_allclose((plus[1] - minus[1]) / (2 * eps), h @ v, rtol=0, atol=1e-8)

    def test_unknown_features_do_not_enter_pairs(self):
        z = self.z.copy()
        z[self.y == -1] += 300
        q = build_pairs(z, self.y, self.tracks, guard=self.guard)
        np.testing.assert_array_equal(q.delta, self.pairs.delta)
        np.testing.assert_array_equal(q.weight, self.pairs.weight)

    def test_common_history_cancels(self):
        shared = np.r_[0:96, 120:124]
        np.testing.assert_array_equal(self.pairs.delta[:, shared], np.zeros((len(self.pairs.delta), 100)))
        _, g, _ = pair_objective(self.pairs, self.w)
        np.testing.assert_allclose(g[shared], self.w[shared] / self.pairs.weight.sum(), rtol=0, atol=1e-14)

    def test_window_shift_invariance(self):
        shift = np.random.default_rng(11).normal(size=(3, 1, 200))
        q = build_pairs(self.z + shift, self.y, self.tracks, guard=self.guard)
        np.testing.assert_allclose(q.delta, self.pairs.delta, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(rank4(score(self.z, RankModel("pairwise", self.w, 0))),
                                      rank4(score(self.z + shift, RankModel("pairwise", self.w, 0))))

    def test_slot_permutation_loss_invariance(self):
        perm = np.random.default_rng(22).permutation(20)
        q = build_pairs(self.z[:, perm], self.y[:, perm], self.tracks, guard=self.guard)
        for a, b in zip(pair_objective(q, self.w), pair_objective(self.pairs, self.w)):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_ties_and_prediction_signature(self):
        np.testing.assert_array_equal(rank4(np.zeros((2, 20))), np.tile(np.arange(4), (2, 1)))
        self.assertEqual(list(inspect.signature(score).parameters), ["z", "model"])
        with self.assertRaises(ValueError):
            score(self.z, RankModel("pairwise", self.w, 1.0))

    def test_invalid_data_rejected(self):
        z = self.z.copy()
        z[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_pairs(z, self.y, self.tracks, guard=self.guard)
        bad = self.y.copy()
        bad[0] = 0
        with self.assertRaises(ValueError):
            build_pairs(self.z, bad, self.tracks, guard=self.guard)
        with self.assertRaises(ValueError):
            build_pairs(self.z, self.y, self.tracks, guard=None)

    def test_hit_unknown_and_allN_decomposition(self):
        slots = np.array([[0, 4, 5, 6], [3, 4, 5, 6], [3, 4, 5, 19]])
        out = describe(slots, self.y)
        self.assertEqual(out["hits"], 1)
        self.assertEqual(out["knownP_miss"], 2)
        self.assertEqual(out["miss_all_N"], 1)
        self.assertEqual(out["miss_contains_U"], 1)
        self.assertEqual(out["all_window_unknown_upper"], 2 / 3)
        with self.assertRaises(ValueError):
            hit_arrays(np.zeros((3, 4), dtype=int), self.y)

    def test_bootstrap_shared_draws_and_window_weighting(self):
        delta = np.tile(np.array([1., 0., -1.])[:, None], (1, 5))
        out = paired_cis(self.tracks, np.ones(3, dtype=bool), delta)
        np.testing.assert_array_equal(out["estimate"], np.zeros(5))
        np.testing.assert_array_equal(out["draws"], np.repeat(out["draws"][:, :1], 5, axis=1))
        again = paired_cis(self.tracks, np.ones(3, dtype=bool), delta)
        np.testing.assert_array_equal(again["draws"], out["draws"])
        np.testing.assert_array_equal(out["CI99"], np.quantile(out["draws"], [.005, .995], axis=0).T)

    def test_screen_requires_gain_and_change_gates(self):
        # Explicit synthetic 384-window fixture; NOT the project's true population.
        y = np.zeros((384, 20), dtype=np.int8)
        y[:263, 0] = 1
        d = dict(labels=y, track=np.arange(384), fold=np.arange(384) % 3,
                 groups=np.full(384, 2))
        hit = np.tile([0, 1, 2, 3], (384, 1))
        miss = np.tile([4, 5, 6, 7], (384, 1))
        slots = {ref: miss.copy() for ref in REFERENCES}
        slots["pairwise"] = hit
        result, draws = evaluate_screen(d, slots)
        self.assertTrue(result["statistical_screen_only"])
        self.assertEqual(draws.shape, (2000, 5))
        slots["pointwise"] = hit
        result, _ = evaluate_screen(d, slots)
        self.assertFalse(result["statistical_screen_only"])


if __name__ == "__main__":
    unittest.main()
