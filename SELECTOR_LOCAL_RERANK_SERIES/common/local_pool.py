"""
Local Pool Builder for S6 Local Fine Reranking.

Selects the Native Top-K candidate pool based on generator logit rankings.
Provides strict separation between candidate indices/logits and privileged targets (ade/fde).
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np


def build_native_topk_pool(data: Dict[str, np.ndarray], k: int) -> Dict[str, np.ndarray]:
    """
    Extracts the Top-K candidate trajectories and associated metadata according to Native logits.
    
    Args:
        data: Fold data dictionary containing 'g0', 'ade', 'fde', and optional 'collision', 'candidates'.
        k: Number of top native candidates to retain (e.g. 3, 5, 10).
        
    Returns:
        Dictionary with:
            - indices: (N, K) original mode indices (0-19) in Native rank order
            - logits: (N, K) native confidence logits for Top-K
            - ade: (N, K) ground truth ADE [PRIVILEGED - evaluation/supervision only]
            - fde: (N, K) ground truth FDE [PRIVILEGED - evaluation/supervision only]
            - collision: (N, K) ground truth collision boolean [PRIVILEGED]
            - candidates: (N, K, 12, 2) trajectory coordinates if present
    """
    native_logits = data["g0"][:, :, 0]  # (N, 20)
    native_order = np.argsort(-native_logits, axis=-1)  # (N, 20) descending
    topk_idx = native_order[:, :k]  # (N, K)

    pool = {
        "indices": topk_idx,
        "logits": np.take_along_axis(native_logits, topk_idx, axis=-1),
        "ade": np.take_along_axis(data["ade"], topk_idx, axis=-1),
        "fde": np.take_along_axis(data["fde"], topk_idx, axis=-1),
    }

    if "collision" in data:
        pool["collision"] = np.take_along_axis(data["collision"], topk_idx, axis=-1)

    if "candidates" in data:
        # candidates shape (N, 20, 12, 2)
        cands = data["candidates"]
        pool["candidates"] = np.take_along_axis(cands, topk_idx[:, :, None, None], axis=1)

    return pool
