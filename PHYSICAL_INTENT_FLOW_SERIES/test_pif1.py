from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIF = load("pif_test_models", HERE / "physical_intent_flow_models.py")
AUDIT = load("pif_test_audit", HERE / "audit_c00_mode_utility.py")


def test_control_round_trip() -> None:
    torch.manual_seed(3)
    past = torch.zeros(4, 2, 8, 2)
    velocity = torch.tensor([0.12, 0.03])
    for step in range(8):
        past[:, :, step] = step * velocity
    controls = torch.randn(4, 2, 20, 6) * 0.15
    anchors = PIF.controls_to_anchors(controls, past, 0.02)
    reconstructed = PIF.anchors_to_controls(anchors, past[:, :, None].expand(-1, -1, 20, -1, -1), 0.02)
    error = PIF.wrap_angle(reconstructed - controls)
    assert torch.max(torch.abs(error)).item() < 2.0e-5


def test_flow_inverse_and_density_agree() -> None:
    torch.manual_seed(4)
    flow = PIF.ConditionalIntentFlow(context_dim=8, hidden_dim=16, layers=4)
    context = torch.randn(3, 5, 8)
    latent = torch.randn(3, 5, 6)
    value, sampled_log_prob = flow.sample(context, latent)
    direct_log_prob = flow.log_prob(value, context)
    assert torch.allclose(sampled_log_prob, direct_log_prob, atol=2.0e-5, rtol=1.0e-5)
    loss = -direct_log_prob.mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in flow.parameters())


def test_latent_points_are_deterministic_and_finite() -> None:
    first = PIF.fixed_latent_points()
    second = PIF.fixed_latent_points()
    assert first.shape == (20, 6)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.equal(first[0], torch.zeros(6))


def test_marginal_contribution_only_charges_winner() -> None:
    quality = np.array([[1.0, 2.0, 3.0], [4.0, 1.5, 2.0]])
    contribution = AUDIT.marginal_contribution(quality)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    assert np.allclose(contribution, expected)

