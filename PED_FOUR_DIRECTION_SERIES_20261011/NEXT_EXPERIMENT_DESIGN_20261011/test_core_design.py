"""Synthetic-only checks. These do not validate scientific datasets or GPU training."""
import unittest
import numpy as np
from core_design import (
    radius, envelope_targets, identity_bucket, selector_features, select4,
    bottom4, fixed_random4, gather, scatter_native, position3_decode,
    potential_hit, accept_edit_subset, temporal_basis, temporal_readout,
)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261011)

    def test_same_candidate_not_disjoint_minima(self):
        p = np.full((1,20,12,2), 10.)
        p[0,0] = 0.; p[0,0,-1,0] = .5   # ADE good, FDE bad
        p[0,1] = 0.; p[0,1,:11,0] = .3  # FDE good, ADE bad
        labels = envelope_targets(p, np.zeros((1,12,2)))
        self.assertFalse(labels["native_good"][0])

    def test_envelope_is_radial_lower_bound(self):
        p = self.rng.normal(size=(3,20,12,2))
        gt = self.rng.normal(size=(3,12,2))
        delta = gt[:,None] - p
        e = np.linalg.norm(delta, axis=-1)
        ratio = np.minimum(1., radius()[None,None] / np.maximum(e, 1e-300))
        ideal = p + ratio[...,None] * delta
        error = np.linalg.norm(ideal - gt[:,None], axis=-1)
        np.testing.assert_allclose(error, np.maximum(e-radius()[None,None],0),
                                   atol=2e-15, rtol=0)
        lab = envelope_targets(p, gt)
        expected = np.maximum(error.mean(-1)/.2, error[...,-1]/.4)
        np.testing.assert_allclose(lab["lower_j"], expected, atol=1e-14, rtol=0)

    def test_good_not_potential_positive(self):
        l = envelope_targets(np.zeros((2,20,12,2)), np.zeros((2,12,2)))
        self.assertTrue(l["native_good"].all())
        self.assertFalse(l["potential"].any())

    def test_identity_stable(self):
        self.assertEqual(identity_bucket("seq","ped"), identity_bucket("seq","ped"))
        with self.assertRaises(ValueError):
            identity_bucket("", "ped")
        a = [identity_bucket("seq", str(i)) for i in range(100)]
        self.assertTrue(all(0 <= b < 5 for b in a))

    def test_stable_ties(self):
        np.testing.assert_array_equal(select4(np.ones((1,20))), [[0,1,2,3]])
        np.testing.assert_array_equal(bottom4(np.ones((1,20))), [[16,17,18,19]])

    def test_random_four_deterministic(self):
        a = fixed_random4(["x","y"])
        np.testing.assert_array_equal(a, fixed_random4(["x","y"]))
        self.assertTrue((np.diff(a,axis=1) > 0).all())

    def test_features_contract(self):
        f = selector_features(np.ones((2,96)), np.zeros((2,20,12,2)))
        self.assertEqual(f.shape, (2,20,120))
        np.testing.assert_array_equal(f[:,:,:96], 1.)
        with self.assertRaises(ValueError):
            select4(np.full((1,20),np.nan))

    def test_gather_scatter_fixed16(self):
        p = self.rng.normal(size=(2,20,12,2))
        s = np.array([[0,5,10,19],[1,3,9,12]])
        a = gather(p,s)
        q = a + .01
        out = scatter_native(p,s,q)
        np.testing.assert_array_equal(gather(out,s), q)
        for i in range(2):
            keep = np.setdiff1d(np.arange(20),s[i])
            np.testing.assert_array_equal(out[i,keep],p[i,keep])
        np.testing.assert_array_equal(scatter_native(p,s,a),p)
        with self.assertRaises(ValueError):
            gather(p,np.array([[0,0,2,3],[1,2,3,4]]))

    def test_position_zero_and_bound(self):
        a = self.rng.normal(size=(2,4,12,2))
        out,res,lam = position3_decode(a,np.zeros((2,4,3,2)))
        np.testing.assert_array_equal(out,a)
        np.testing.assert_array_equal(res,0.)
        out,res,lam = position3_decode(a,self.rng.normal(size=(2,4,3,2))*8)
        self.assertTrue(np.all(np.linalg.norm(res,axis=-1)<=radius()+1e-15))
        self.assertTrue(np.all((lam>=0)&(lam<=1)))
        np.testing.assert_array_equal(out,a+res)

    def test_potential_gather(self):
        p = np.zeros((2,20),bool); p[:,8] = True
        np.testing.assert_array_equal(
            potential_hit(p, np.array([[0,1,2,8],[0,1,2,3]])), [True,False])

    def test_subset_zero_budget_reject(self):
        native = np.zeros((20,12,2)); s = np.array([0,1,2,3])
        proposal = np.ones((4,12,2))
        # Fixture callback, NOT production physical metrics.
        motion = lambda x: np.full(5,np.abs(x).sum())
        out,mask = accept_edit_subset(native,s,proposal,np.ones(4),motion)
        self.assertEqual(mask,0); np.testing.assert_array_equal(out,native)

    def test_subset_full_accept_and_exact_fallback_tie(self):
        p = np.zeros((20,12,2)); s = np.array([0,1,2,3]); q = np.ones((4,12,2))
        motion = lambda x: np.ones(5)
        out,mask = accept_edit_subset(p,s,q,np.array([.1,.2,.3,.4]),motion)
        self.assertEqual(mask,15)
        out,mask = accept_edit_subset(p,s,q,np.zeros(4),motion)
        self.assertEqual(mask,0)

    def test_temporal_basis_and_reversal(self):
        h = temporal_basis()
        np.testing.assert_allclose(h@h.T,np.eye(3)/8,atol=1e-16,rtol=0)
        z = self.rng.normal(size=(2,8,32))
        a = temporal_readout(z,"temporal_dct").reshape(2,3,32)
        b = temporal_readout(z[:,::-1],"temporal_dct").reshape(2,3,32)
        np.testing.assert_allclose(b,a*np.array([1,-1,1])[None,:,None],
                                   atol=1e-15,rtol=0)
        mean = temporal_readout(z,"mean").reshape(2,3,32)
        np.testing.assert_allclose(mean[:,0],z.mean(1),atol=1e-15,rtol=0)
        constant = np.ones((1,8,32))
        c = temporal_readout(constant,"temporal_dct").reshape(1,3,32)
        np.testing.assert_allclose(c[:,1:],0.,atol=1e-15,rtol=0)
        # Constant tokens give DC-only temporal coefficients, NOT repeated mean.
        # Thus mean/DCT feature equality is NOT asserted for nonzero constants.


if __name__ == "__main__":
    unittest.main(verbosity=2)
