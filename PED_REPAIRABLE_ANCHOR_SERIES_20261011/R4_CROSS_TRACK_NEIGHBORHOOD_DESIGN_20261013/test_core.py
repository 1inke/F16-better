"""Synthetic-only tests; no project input adapter calls or experiment execution."""
import inspect
import json
import unittest
import numpy as np
from core_neighbors import (make_bank, retrieve, distance_components, calibration_weights,
    weighted_q95, geometry_support, label_support, best_positive, class_window_macro)
from summary_stats import six_statistics, json_safe
from fold_diagnostic import geometry_fold, posthoc_fold, primary_rows


def guard():
    pass


def fixture(n=7):
    z = np.broadcast_to(np.arange(n)[:, None, None], (n, 20, 200)).astype(float).copy()
    return z, np.arange(100, 100+n), np.array(["t"+str(i) for i in range(n)])


class TestGeometry(unittest.TestCase):
    def test_distinct_tracks_not_rows_and_self_exclusion(self):
        z, ids, tracks = fixture()
        b = make_bank(z, ids, tracks)
        ix, d = retrieve(z[0, :1], tracks[:1], b, guard=guard)
        np.testing.assert_array_equal(b.ids[ix[0]], ids[1:6])
        np.testing.assert_array_equal(b.slots[ix[0]], np.zeros(5))
        np.testing.assert_array_equal(d[0], np.arange(1, 6))

    def test_label_blind_signature(self):
        self.assertEqual(list(inspect.signature(retrieve).parameters),
                         ["query_features", "query_tracks", "bank", "guard", "chunk"])

    def test_ties_bank_order_and_chunk(self):
        z, ids, tracks = fixture()
        z.fill(0)
        a = make_bank(z, ids, tracks)
        perm = np.arange(7)[::-1]
        b = make_bank(z[perm], ids[perm], tracks[perm])
        ia, da = retrieve(z[:1, 0], ["query"], a, guard=guard, chunk=31)
        ib, db = retrieve(z[:1, 0], ["query"], b, guard=guard, chunk=256)
        np.testing.assert_array_equal(a.ids[ia], b.ids[ib])
        np.testing.assert_array_equal(a.slots[ia], b.slots[ib])
        np.testing.assert_array_equal(da, db)

    def test_independent_track_loop(self):
        rng = np.random.default_rng(19)
        z, ids, tracks = fixture(8)
        z = rng.normal(size=z.shape)
        tracks[-1] = tracks[0]  # two windows of one track must still count once
        bank = make_bank(z, ids, tracks)
        q = rng.normal(size=(2, 200))
        ix, dist = retrieve(q, ["new", "t0"], bank, guard=guard, chunk=31)
        for row, exclude in enumerate(("new", "t0")):
            candidates = []
            for track in np.unique(tracks):
                if track == exclude:
                    continue
                entries = []
                for w in np.flatnonzero(tracks == track):
                    for slot in range(20):
                        squared = sum(float(v)**2 for v in (q[row]-z[w, slot])) / 200
                        entries.append((squared, int(ids[w]), slot, w*20+slot))
                candidates.append(min(entries))
            chosen = sorted(candidates)[:5]
            np.testing.assert_array_equal(ix[row], [c[3] for c in chosen])
            np.testing.assert_allclose(dist[row], np.sqrt([c[0] for c in chosen]), rtol=1e-14, atol=0)

    def test_components_and_nontrivial_chunk_exact(self):
        z, ids, tracks = fixture()
        z += np.random.default_rng(1).normal(size=z.shape)
        b = make_bank(z, ids, tracks)
        ix, d = retrieve(z[0, :2], ["t0", "t0"], b, guard=guard, chunk=31)
        ix2, d2 = retrieve(z[0, :2], ["t0", "t0"], b, guard=guard)
        np.testing.assert_array_equal(ix, ix2)
        np.testing.assert_array_equal(d, d2)
        np.testing.assert_allclose(distance_components(z[0, :2], b, ix).sum(-1), d*d, rtol=1e-14)

    def test_insufficient_tracks_stop(self):
        z, ids, tracks = fixture(5)
        with self.assertRaisesRegex(ValueError, "STOP_INPUT_SUPPORT"):
            retrieve(z[0, :1], tracks[:1], make_bank(z, ids, tracks), guard=guard)

    def test_bad_inputs(self):
        z, ids, tracks = fixture()
        with self.assertRaises(ValueError):
            make_bank(z, np.zeros_like(ids), tracks)
        z[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            make_bank(z, ids, tracks)

    def test_track_weights_quantile(self):
        w = calibration_weights(np.array(["a", "a", "b"]))
        self.assertAlmostEqual(w[:40].sum(), .5)
        self.assertAlmostEqual(w[40:].sum(), .5)
        self.assertEqual(weighted_q95(np.array([0., 3.]), np.array([.96, .04])), 0.)
        self.assertEqual(weighted_q95(np.array([0., 3.]), np.array([.94, .06])), 3.)

    def test_zero_radius(self):
        s = geometry_support(np.array([[0]*5, [1]*5], float), 0.)
        np.testing.assert_array_equal(s["near"], [True, False])
        self.assertTrue(np.isnan(s["radius_ratio"]).all())

    def test_fold_geometry_only_then_labels(self):
        z, ids, tracks = fixture()
        g = geometry_fold(z, ids, tracks, z[:1]+.2, [999], ["new"], guard=guard)
        self.assertEqual(g["indices"].shape, (20, 5))
        self.assertAlmostEqual(g["train_weights"].sum(), 1.)
        y = np.zeros((7, 20), int)
        y[:, 0] = 1
        self.assertEqual(posthoc_fold(g, y)["supported"].shape, (20,))
        with self.assertRaisesRegex(ValueError, "tracks overlap"):
            geometry_fold(z, ids, tracks, z[:1], [999], tracks[:1], guard=guard)


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.z, self.ids, self.tracks = fixture()
        self.bank = make_bank(self.z, self.ids, self.tracks)
        self.y = np.zeros((7, 20), int)
        self.y[:, 0] = 1

    def test_base_rate_and_any_P(self):
        ix = np.array([[0, 21, 41, 61, 81]])
        s = label_support(ix, self.bank, self.y)
        self.assertAlmostEqual(s["observed_P"][0], .2)
        self.assertAlmostEqual(s["expected_P"][0], .05)
        self.assertAlmostEqual(s["enrichment"][0], .15)
        self.assertAlmostEqual(s["expected_anyP"][0], 1-.95**5)

    def test_U_retained_no_backfill(self):
        self.y[:4, 0] = -1
        self.y[:4, 1] = 1
        ix, _ = retrieve(self.z[:1, 0], ["new"], self.bank, guard=guard)
        s = label_support(ix, self.bank, self.y)
        self.assertEqual(s["nU"][0], 4)
        self.assertEqual(s["known"][0], 1)
        self.assertFalse(s["known_sufficient"][0])

    def test_all_U_null_not_zero(self):
        self.y[:, 0], self.y[:, 1] = -1, 1
        s = label_support(np.array([[0, 20, 40, 60, 80]]), self.bank, self.y)
        self.assertTrue(np.isnan(s["enrichment"][0]))
        self.assertTrue(np.isnan(s["observed_anyP"][0]))
        self.assertEqual(json.loads(json.dumps(json_safe(s), allow_nan=False))["enrichment"], [None])

    def test_best_P_ties_and_missing(self):
        y = np.zeros((3, 20), int)
        y[0, [4, 8]], y[1, 10] = 1, 1
        s = best_positive(np.zeros((3, 20)), y)
        np.testing.assert_array_equal(s["slot"], [4, 10, -1])
        np.testing.assert_array_equal(s["rank"], [5, 11, -1])
        np.testing.assert_array_equal(s["bin"], [1, 2, -1])

    def test_macro_window_not_candidate_weighted(self):
        y = np.zeros((2, 20), int)
        y[0, 0], y[1, :10] = 1, 1
        v = np.zeros((2, 20))
        v[0] = 1
        s = class_window_macro(v, y)
        self.assertEqual(s["P"]["mean"], .5)
        self.assertEqual(s["P"]["eligible_candidates"], 11)

    def test_primary_uses_old_score_not_neighborhood(self):
        y = np.zeros((2, 20), int)
        y[0, [4, 9]] = 1
        values = np.zeros((2, 20))
        values[0, 9] = 1.0  # nicer neighborhood must not change best-P selection
        p = primary_rows(np.zeros((2, 20)), y, {"enrichment": values})
        np.testing.assert_array_equal(p["physical_slot"], [4])
        np.testing.assert_array_equal(p["enrichment"], [0.])


class TestStatistics(unittest.TestCase):
    def fixture(self):
        # 40 tracks, each with one boundary and one deep window; all change.
        return (np.tile([.2, -.1], 40), np.ones(80, bool), np.tile([6, 12], 40),
                np.full(80, 2), np.repeat(np.arange(40), 2))

    def test_shared_bootstrap_deterministic(self):
        a = self.fixture()
        x, y = six_statistics(*a, guard=guard), six_statistics(*a, guard=guard)
        np.testing.assert_array_equal(x["draw_counts"], y["draw_counts"])
        np.testing.assert_array_equal(x["values"], y["values"])
        self.assertTrue((x["draw_counts"].sum(1) == 40).all())
        self.assertEqual(len(x["statistics"]), 6)
        for record, expected in zip(x["statistics"], [.2, -.1, .3, .2, -.1, .3]):
            self.assertEqual(record["status"], "EXPLORATORY_INTERVAL")
            np.testing.assert_allclose(record["interval"], [expected, expected], atol=1e-14)
        # Independent reconstruct of a draw using actual resampled whole tracks.
        multiplicities = x["draw_counts"][17]
        rows = np.repeat(np.arange(80), np.repeat(multiplicities, 2))
        brute = a[0][rows][a[2][rows] <= 8].mean()
        self.assertAlmostEqual(brute, x["values"][17, 0])

    def test_support_gate_and_empty_cells(self):
        e, ok, rank, group, tracks = self.fixture()
        ok[:30] = False
        x = six_statistics(e, ok, rank, group, tracks, guard=guard)
        self.assertTrue(all(s["interval"] is None for s in x["statistics"]))
        self.assertTrue(all(s["status"] == "INCONCLUSIVE_SUPPORT" for s in x["statistics"]))
        group[:] = 0
        x = six_statistics(e, ok, rank, group, tracks, guard=guard)
        self.assertEqual(x["statistics"][3]["valid_draws"], 0)
        self.assertIsNone(x["statistics"][3]["estimate"])
        json.dumps(json_safe(x), allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
