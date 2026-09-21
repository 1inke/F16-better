"""
Feature Schema and Validation Definitions for F16 Rankability Probe Series.

Defines:
- CACHE_FEATURE_KEYS: Schema of cached numpy arrays
- PROBE_FEATURE_DIMS: Expected input feature dimensions for R1, R2, R3, R4
- validate_cache_schema: Asserts array shapes and sanity constraints
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np

# Dimension specifications per protocol §7 & §12
PROBE_FEATURE_DIMS = {
    "R1_exact_input": 70,       # mode(64) + source_anchors(6) = 70
    "R2_post_correction": 70,   # mode(64) + corrected_anchors(6) = 70
    "R3_context_post": 134,     # context(64) + mode(64) + corrected_anchors(6) = 134
    "R4_full_final": 152,       # context(64) + mode(64) + final_trajectory(24) = 152
}

EXPECTED_SHAPES = {
    "sample_id": (),
    "past_relative": (8, 2),
    "candidate_final": (20, 12, 2),
    "candidate_coarse": (20, 12, 2),
    "source_anchors_relative": (20, 3, 2),
    "corrected_anchors_relative": (20, 3, 2),
    "context_feature": (64,),
    "mode_feature": (20, 64),
    "native_logits": (20,),
    "ADE_final": (20,),
    "FDE_final": (20,),
    "ADE_coarse": (20,),
    "FDE_coarse": (20,),
    "native_selected_index": (),
    "oracle_final_index": (),
    "oracle_coarse_index": (),
    "future_relative": (12, 2),
}


def validate_cache_dict(cache: Dict[str, np.ndarray]) -> None:
    """
    Validates that all expected fields exist and match expected dimensions.
    """
    if "sample_id" not in cache:
        raise KeyError("Missing 'sample_id' in cache dict")

    N = len(cache["sample_id"])
    for key, shape in EXPECTED_SHAPES.items():
        if key not in cache:
            raise KeyError(f"Missing required key '{key}' in cache")
        arr = cache[key]
        expected_shape = (N,) + shape
        if arr.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for key '{key}': expected {expected_shape}, got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"Array '{key}' contains NaN or Inf values")

    # Sanity checks on indices
    native_sel = cache["native_selected_index"]
    oracle_fin = cache["oracle_final_index"]
    oracle_coa = cache["oracle_coarse_index"]

    assert np.all((native_sel >= 0) & (native_sel < 20)), "native_selected_index out of range"
    assert np.all((oracle_fin >= 0) & (oracle_fin < 20)), "oracle_final_index out of range"
    assert np.all((oracle_coa >= 0) & (oracle_coa < 20)), "oracle_coarse_index out of range"

    # Sanity checks on oracle definition
    ade_fin = cache["ADE_final"]
    ade_coa = cache["ADE_coarse"]
    assert np.all(np.argmin(ade_fin, axis=-1) == oracle_fin), "oracle_final_index != argmin(ADE_final)"
    assert np.all(np.argmin(ade_coa, axis=-1) == oracle_coa), "oracle_coarse_index != argmin(ADE_coarse)"
    assert np.all(np.argmax(cache["native_logits"], axis=-1) == native_sel), "native_selected_index != argmax(native_logits)"
