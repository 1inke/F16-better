import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


SCRIPT = Path(__file__).with_name("directional_regime_pilot.py")
SPEC = importlib.util.spec_from_file_location("directional_pilot_tested", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_maneuver_labels_cover_predeclared_states():
    cfg = MODULE.load_config()
    observed = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (7, 1))
    future = np.array([
        [1.0, 0.0], [0.5, 0.0], [0.05, 0.0], [1.6, 0.0],
        [0.8, 0.8], [0.8, -0.8], [-1.0, 0.0],
    ], dtype=np.float32)
    labels = MODULE.classify_velocity(observed, future, cfg)
    assert labels.tolist() == [0, 1, 2, 3, 4, 5, 6]


def test_explicit_candidates_are_observation_only_and_deterministic():
    past = np.zeros((2, 8, 2), dtype=np.float32)
    past[:, :, 0] = np.arange(8, dtype=np.float32)[None] * 0.2
    first = MODULE.explicit_candidates(past)
    second = MODULE.explicit_candidates(past.copy())
    assert first.shape == (2, 8, 12, 2)
    assert np.array_equal(first, second)
    assert np.allclose(first[:, 0], 0.0)  # stop proposal


def test_zero_residual_plain_head_preserves_native_argmax():
    model = MODULE.DirectionalHead(10, 12, hidden=16, dropout=0.0, attenuate=False, beta=0.75, floor=0.2)
    obs = torch.randn(3, 10)
    candidate = torch.randn(3, 5, 12)
    native = torch.randn(3, 5)
    probs = torch.softmax(torch.randn(3, 7), -1)
    output, scale = model(obs, candidate, native, probs)
    assert torch.equal(output.argmax(-1), native.argmax(-1))
    assert torch.allclose(scale, torch.ones_like(scale))


def test_high_risk_uncertainty_attenuates_native_scores():
    model = MODULE.DirectionalHead(10, 12, hidden=16, dropout=0.0, attenuate=True, beta=0.75, floor=0.2)
    obs = torch.randn(2, 10)
    candidate = torch.randn(2, 5, 12)
    native = torch.randn(2, 5)
    probs = torch.tensor([[0.95, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005], [1/7]*7])
    _, scale = model(obs, candidate, native, probs)
    assert 0.2 <= float(scale[1]) < float(scale[0]) <= 1.0


def test_candidate_feature_shape_and_finiteness():
    rng = np.random.default_rng(3)
    past = rng.normal(size=(4, 8, 2)).astype(np.float32)
    trajectories = rng.normal(size=(4, 20, 12, 2)).astype(np.float32)
    logits = rng.normal(size=(4, 20)).astype(np.float32)
    labels = rng.integers(0, 7, size=(4, 20))
    features = MODULE.candidate_features(past, trajectories, logits, labels, np.zeros((4, 20), np.float32))
    assert features.shape == (4, 20, 39)
    assert np.isfinite(features).all()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} direct tests passed")
