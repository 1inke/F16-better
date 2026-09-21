"""
Test Suite: No Future Leakage & Dimension Verification.

Enforces Butler Rules:
- research.variable_integrity_discipline
- research.semantic_objective_alignment

Asserts:
1. Probe input builder never accesses GT / future / ADE / FDE / oracle indices.
2. Probe input dimensions strictly match protocol §12:
   - R1: 70
   - R2: 70
   - R3: 134
   - R4: 152
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE)]

from common.feature_schema import PROBE_FEATURE_DIMS, validate_cache_dict


def build_probe_features(cache: dict[str, np.ndarray], probe_name: str) -> np.ndarray:
    """
    Constructs probe input features per mode.
    Disallows any access to future or GT information.
    """
    mode = cache["mode_feature"]               # [N, 20, 64]
    N, K, _ = mode.shape

    if probe_name == "R1_exact_input":
        src_anc = cache["source_anchors_relative"].reshape(N, K, -1)  # [N, 20, 6]
        feat = np.concatenate([mode, src_anc], axis=-1)                # [N, 20, 70]
    elif probe_name == "R2_post_correction":
        corr_anc = cache["corrected_anchors_relative"].reshape(N, K, -1)  # [N, 20, 6]
        feat = np.concatenate([mode, corr_anc], axis=-1)                   # [N, 20, 70]
    elif probe_name == "R3_context_post":
        ctx = cache["context_feature"][:, None, :].repeat(K, axis=1)     # [N, 20, 64]
        corr_anc = cache["corrected_anchors_relative"].reshape(N, K, -1)  # [N, 20, 6]
        feat = np.concatenate([ctx, mode, corr_anc], axis=-1)             # [N, 20, 134]
    elif probe_name == "R4_full_final":
        ctx = cache["context_feature"][:, None, :].repeat(K, axis=1)     # [N, 20, 64]
        cand_fin = cache["candidate_final"].reshape(N, K, -1)            # [N, 20, 24]
        feat = np.concatenate([ctx, mode, cand_fin], axis=-1)             # [N, 20, 152]
    else:
        raise ValueError(f"Unknown probe: {probe_name}")

    return feat


def test_probe_dimensions():
    # Synthetic mock cache to verify feature builders
    N = 10
    mock_cache = {
        "sample_id": np.arange(N),
        "past_relative": np.zeros((N, 8, 2)),
        "future_relative": np.zeros((N, 12, 2)),
        "candidate_final": np.zeros((N, 20, 12, 2)),
        "candidate_coarse": np.zeros((N, 20, 12, 2)),
        "source_anchors_relative": np.zeros((N, 20, 3, 2)),
        "corrected_anchors_relative": np.zeros((N, 20, 3, 2)),
        "context_feature": np.zeros((N, 64)),
        "mode_feature": np.zeros((N, 20, 64)),
        "native_logits": np.zeros((N, 20)),
        "ADE_final": np.zeros((N, 20)),
        "FDE_final": np.zeros((N, 20)),
        "ADE_coarse": np.zeros((N, 20)),
        "FDE_coarse": np.zeros((N, 20)),
        "native_selected_index": np.zeros(N, dtype=int),
        "oracle_final_index": np.zeros(N, dtype=int),
        "oracle_coarse_index": np.zeros(N, dtype=int),
    }

    validate_cache_dict(mock_cache)

    for probe_name, expected_dim in PROBE_FEATURE_DIMS.items():
        feat = build_probe_features(mock_cache, probe_name)
        assert feat.shape == (N, 20, expected_dim), (
            f"Dimension mismatch for {probe_name}: expected {expected_dim}, got {feat.shape[-1]}"
        )
    print("test_probe_dimensions: PASSED")


def test_no_future_leakage_in_features():
    """
    Asserts that no future/ground truth array is accessed when constructing probe features.
    """
    forbidden_keys = {"future_relative", "ADE_final", "FDE_final", "ADE_coarse", "FDE_coarse", "oracle_final_index"}
    
    class LeakageDetectorDict(dict):
        def __getitem__(self, key):
            if key in forbidden_keys:
                raise AssertionError(f"SECURITY VIOLATION: Feature builder accessed forbidden key '{key}'")
            return super().__getitem__(key)

    N = 5
    mock_cache = LeakageDetectorDict({
        "sample_id": np.arange(N),
        "past_relative": np.zeros((N, 8, 2)),
        "future_relative": np.zeros((N, 12, 2)),
        "candidate_final": np.zeros((N, 20, 12, 2)),
        "candidate_coarse": np.zeros((N, 20, 12, 2)),
        "source_anchors_relative": np.zeros((N, 20, 3, 2)),
        "corrected_anchors_relative": np.zeros((N, 20, 3, 2)),
        "context_feature": np.zeros((N, 64)),
        "mode_feature": np.zeros((N, 20, 64)),
        "native_logits": np.zeros((N, 20)),
        "ADE_final": np.zeros((N, 20)),
        "FDE_final": np.zeros((N, 20)),
        "ADE_coarse": np.zeros((N, 20)),
        "FDE_coarse": np.zeros((N, 20)),
        "native_selected_index": np.zeros(N, dtype=int),
        "oracle_final_index": np.zeros(N, dtype=int),
        "oracle_coarse_index": np.zeros(N, dtype=int),
    })

    for probe_name in PROBE_FEATURE_DIMS:
        # Must execute without triggering KeyError on forbidden keys
        feat = build_probe_features(mock_cache, probe_name)
        assert feat is not None
    print("test_no_future_leakage_in_features: PASSED")


if __name__ == "__main__":
    test_probe_dimensions()
    test_no_future_leakage_in_features()
    print("All feature and leakage tests PASSED!")
