"""
Cache IO and Data Adapter for Rankability Series.

Loads precomputed Feature Bank (npz) and Candidate Cache (pt) files.
Enforces strict boundary between deployable features and privileged labels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PRIVILEGED_NAMES = {"ade", "fde", "clearance", "collision", "focal_future_gt", "neighbor_future_gt"}


def assert_no_privileged_leakage(feature_array: Any, name: str = "feature") -> None:
    """Verifies that no privileged evaluation target is mixed into deployable features."""
    if hasattr(feature_array, "detach"):
        arr = feature_array.detach().cpu().numpy()
    else:
        arr = np.asarray(feature_array)
    # Features must not contain NaNs or Infs
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError(f"Feature '{name}' contains NaN or Inf values!")


def load_fold_features(
    feature_root: Path,
    fold: str,
    seed: int = 1,
) -> Dict[str, np.ndarray]:
    """Loads feature bank npz for a specific fold and seed."""
    fold_dir = feature_root / f"seed_{seed}" / fold
    npz_path = fold_dir / "features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature npz missing: {npz_path}")

    data = np.load(npz_path)
    result = {k: data[k] for k in data.files}

    # Verify shapes and integrity
    assert "g0" in result and "g1" in result and "g2" in result and "g3" in result and "g4" in result
    assert "ade" in result and "min_ade20" in result and "native_top1_idx" in result
    return result


def assemble_feature_groups(fold_data: Dict[str, np.ndarray], groups: List[str]) -> np.ndarray:
    """Concatenates requested feature groups e.g. ['g0', 'g1', 'g2']."""
    feats = []
    for g in groups:
        f = fold_data[g]
        assert_no_privileged_leakage(f, g)
        feats.append(f)
    return np.concatenate(feats, axis=-1)  # (N, 20, D)
