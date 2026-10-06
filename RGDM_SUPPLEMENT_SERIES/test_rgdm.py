import unittest

import torch

from rgdm_models import (
    BASE,
    RGDMSupplementDecoder,
    RegimeAwareGatedDeltaMemory,
    SelectionSideRGDM,
    gated_delta_step,
    observation_motion_evidence,
)


class RGDMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_delta_step_matches_explicit_formula(self):
        state = torch.randn(2, 3, 4, 5)
        key = torch.nn.functional.normalize(torch.randn(2, 3, 5), dim=-1)
        value = torch.randn(2, 3, 4)
        alpha = torch.sigmoid(torch.randn(2, 3))
        beta = torch.sigmoid(torch.randn(2, 3))
        identity = torch.eye(5).expand(2, 3, 5, 5)
        erase = identity - beta[..., None, None] * torch.einsum("...i,...j->...ij", key, key)
        expected = alpha[..., None, None] * torch.matmul(state, erase)
        expected = expected + beta[..., None, None] * torch.einsum("...i,...j->...ij", value, key)
        torch.testing.assert_close(gated_delta_step(state, key, value, alpha, beta), expected)

    def test_observation_evidence_is_finite_and_has_no_future_argument(self):
        past = torch.randn(4, 2, 8, 2).cumsum(dim=-2)
        evidence = observation_motion_evidence(past)
        self.assertEqual(evidence.shape, (4, 2, 8, 8))
        self.assertTrue(torch.isfinite(evidence).all())

    def test_identity_initialization_matches_locked_decoder_path(self):
        baseline = BASE.JointTrainableV2Decoder(64, 12).eval()
        rgdm = RGDMSupplementDecoder(64, 12, "rgdm_b4_full").eval()
        missing = rgdm.load_state_dict(baseline.state_dict(), strict=False).missing_keys
        self.assertTrue(missing)
        encoded = torch.randn(2, 3, 8, 64)
        past = torch.randn(2, 3, 8, 2).cumsum(dim=-2)
        with torch.inference_mode():
            baseline_trajectory, baseline_logits = baseline(encoded, past)
            rgdm_trajectory, rgdm_logits = rgdm(encoded, past)
        torch.testing.assert_close(rgdm_trajectory, baseline_trajectory, atol=0, rtol=0)
        torch.testing.assert_close(rgdm_logits, baseline_logits, atol=0, rtol=0)

    def test_memory_shapes_and_gate_ranges(self):
        memory = RegimeAwareGatedDeltaMemory(16)
        encoded = torch.randn(3, 2, 8, 16)
        past = torch.randn(3, 2, 8, 2).cumsum(dim=-2)
        sequence, state, diagnostics = memory(encoded, past)
        self.assertEqual(sequence.shape, (3, 2, 8, 32))
        self.assertEqual(state.shape, (3, 2, 4, 8, 8))
        self.assertTrue(((diagnostics["alpha"] > 0) & (diagnostics["alpha"] < 1)).all())
        self.assertTrue(((diagnostics["beta"] > 0) & (diagnostics["beta"] < 1)).all())

    def test_selection_is_candidate_permutation_equivariant(self):
        branch = SelectionSideRGDM(16).eval()
        native = torch.randn(2, 1, 5)
        mode = torch.randn(2, 1, 5, 64)
        anchors = torch.randn(2, 1, 5, 3, 2)
        encoded = torch.randn(2, 1, 8, 16)
        past = torch.randn(2, 1, 8, 2).cumsum(dim=-2)
        permutation = torch.tensor([2, 4, 0, 3, 1])
        first, _ = branch(native, mode, anchors, encoded, past, memory_score=True,
                          candidate_query=True, history_gate=True)
        second, _ = branch(native[..., permutation], mode[..., permutation, :],
                           anchors[..., permutation, :, :], encoded, past, memory_score=True,
                           candidate_query=True, history_gate=True)
        torch.testing.assert_close(second, first[..., permutation])

    def test_zero_identity_controls_receive_gradient(self):
        decoder = RGDMSupplementDecoder(64, 12, "rgdm_ab_joint")
        # The bare decoder constructor intentionally starts its legacy score
        # head at zero; a loaded C00 checkpoint does not.  Give this synthetic
        # fixture a non-degenerate frozen native score so calibration gradients
        # exercise the same path as the real adapter.
        torch.nn.init.normal_(decoder.score_head[-1].weight, std=0.05)
        encoded = torch.randn(2, 1, 8, 64)
        past = torch.randn(2, 1, 8, 2).cumsum(dim=-2)
        trajectory, logits = decoder(encoded, past)
        loss = trajectory.square().mean() + torch.nn.functional.cross_entropy(
            logits[:, 0], torch.tensor([0, 1])
        )
        loss.backward()
        self.assertIsNotNone(decoder.generation_rgdm.fusion_strength.grad)
        self.assertIsNotNone(decoder.selection_rgdm.history_calibration_strength.grad)
        self.assertGreater(float(decoder.selection_rgdm.score_residual[-1].weight.grad.abs().sum()), 0)

    def test_future_change_target_is_detached(self):
        past = torch.randn(3, 1, 8, 2, requires_grad=True).cumsum(dim=-2)
        target = torch.randn(3, 1, 12, 2, requires_grad=True).cumsum(dim=-2)
        result = RGDMSupplementDecoder._future_change_target(past, target)
        self.assertFalse(result.requires_grad)
        self.assertTrue(((result >= 0) & (result <= 1)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
