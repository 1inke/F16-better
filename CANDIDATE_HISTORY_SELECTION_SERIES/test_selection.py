import unittest
import numpy as np
import torch
from selection_core import (INPUT_KEYS, Normalizer, Selector, calibrate, features, history_for,
                            history_tokens, metrics, split_ids, utility_target)


def toy(n=12, k=4):
    r = np.random.default_rng(42)
    past = r.normal(size=(n, 8, 2)).astype(np.float32)
    past -= past[:, -1:]
    return dict(past_relative=past, candidate_final=r.normal(size=(n, k, 12, 2)).astype(np.float32),
                context_feature=r.normal(size=(n, 6)).astype(np.float32),
                mode_feature=r.normal(size=(n, k, 7)).astype(np.float32),
                native_logits=r.normal(size=(n, k)).astype(np.float32))


class Tests(unittest.TestCase):
    def test_no_future_or_label_access(self):
        class Guard(dict):
            def __getitem__(self, key):
                if key not in INPUT_KEYS:
                    raise AssertionError('Forbidden input: ' + key)
                return super().__getitem__(key)
        b, p, cv = features(Guard(toy()))
        self.assertTrue(np.isfinite(b).all())
        self.assertEqual(history_for(p, cv, 'real_history').shape[:2], p.shape[:2])

    def test_metric_direction_and_oracle(self):
        ade = np.array([[0., 1., 2.], [.5, 1.5, 2.5]])
        native = np.array([2, 2])
        m = metrics(ade, 2 * ade, np.array([0, 0]), native)
        self.assertEqual(m['selected_ade'], .25)
        self.assertEqual(m['regret'], 0.)
        self.assertEqual(m['improvement_m'], 2.)
        self.assertEqual(m['harmful_rate'], 0.)
        self.assertEqual(m['useful_rate'], 1.)
        bad = metrics(ade, ade, native, np.array([0, 0]))
        self.assertEqual(bad['harmful_rate'], 1.)
        self.assertEqual(bad['improvement_m'], -2.)

    def test_small_oracle_gap_is_not_low_native_regret(self):
        a = np.array([[0., .01, 1.]])
        m = metrics(a, a, np.array([2]), np.array([2]))
        self.assertEqual(m['small_gap_high_regret_n'], 1)
        self.assertEqual(m['small_gap_low_regret_n'], 0)

    def test_zero_alpha_and_calibration(self):
        logits = np.array([[2., 1., 0.]])
        good = np.array([[0., 0., 4.]])
        a = np.array([[2., 1., 0.]])
        self.assertEqual(calibrate(logits, good, a, 1., [0., 1.])['alpha'], 1.)
        self.assertEqual(calibrate(logits, -good, a, 1., [0., 1.])['alpha'], 0.)
        m = metrics(a, a, (logits + 0 * good).argmax(1), logits.argmax(1))
        self.assertEqual(m['improvement_m'], 0.)

    def test_id_split_stable_under_reordering(self):
        ids = np.arange(100)
        fit, cal = split_ids(ids, .15, 42)
        other = ids[::-1]
        f2, c2 = split_ids(other, .15, 42)
        self.assertEqual(set(ids[fit]), set(other[f2]))
        self.assertEqual(set(ids[cal]), set(other[c2]))
        self.assertFalse(set(ids[fit]) & set(ids[cal]))
        with self.assertRaises(ValueError):
            split_ids(np.array([1, 1]), .15, 42)

    def test_same_architecture_and_candidate_equivariance(self):
        b, p, cv = features(toy())
        h = history_tokens(p)
        cv_net = Selector(b.shape[-1], h.shape[-1], 'cv_history')
        real_net = Selector(b.shape[-1], h.shape[-1], 'real_history')
        self.assertEqual(sum(x.numel() for x in cv_net.parameters()), sum(x.numel() for x in real_net.parameters()))
        perm = np.array([2, 0, 3, 1])
        x, ht = torch.from_numpy(b), torch.from_numpy(h)
        out = real_net(x, ht)
        torch.testing.assert_close(real_net(x[:, perm], ht), out[:, perm])

    def test_shuffled_residual_preserves_endpoint_and_velocity(self):
        b, past, cv = features(toy())
        perm = np.arange(len(past))[::-1]
        hist = history_for(past, cv, 'shuffled_residual', perm)
        np.testing.assert_allclose(hist[:, -1, :2], past[:, -1], atol=1e-6)
        np.testing.assert_allclose(hist[:, -1, 2:4], past[:, -1] - past[:, -2], atol=1e-6)
        constant = history_for(cv, cv, 'real_history')
        np.testing.assert_allclose(constant, history_for(cv, cv, 'cv_history'))

    def test_utility_target_and_gradients(self):
        target = utility_target(torch.tensor([[0., 1., 2.]]))
        self.assertEqual(target.argmax(1).item(), 0)
        torch.testing.assert_close(target.mean(1), torch.zeros(1))
        b, past, cv = features(toy())
        h = history_tokens(past)
        net = Selector(b.shape[-1], h.shape[-1], 'real_history')
        loss = net(torch.from_numpy(b), torch.from_numpy(h)).square().mean()
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in net.history.parameters()), 0)

    def test_normalization_fit_only(self):
        b, past, _ = features(toy())
        h = history_tokens(past)
        norm = Normalizer(b[:6], h[:6])
        before = norm.base_mean.copy()
        norm.apply(b[6:] + 1000, h[6:] + 1000)
        np.testing.assert_array_equal(before, norm.base_mean)


if __name__ == '__main__':
    unittest.main(verbosity=2)
