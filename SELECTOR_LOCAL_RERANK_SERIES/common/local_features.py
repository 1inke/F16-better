"""
Local Feature Extraction for S6 Local Fine Reranking.

Extracts gathered global features (G0, G1, G2, G4) restricted to Native Top-K,
and computes specialized Local Relative Features (GL).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


def compute_gl_features(
    topk_logits: np.ndarray,
    topk_cands: Optional[np.ndarray] = None,
    batch_size: int = 5000,
) -> np.ndarray:
    """
    Computes GL Local Features for Top-K candidate pool.
    
    GL includes 9 relative metrics:
    1. local_rank: index within local pool (0, 1, ..., K-1)
    2. logit_gap_to_top1: native logit difference from local rank 0
    3. logit_gap_to_mean: native logit difference from local mean
    4. prob_gap_to_top1: local softmax probability difference from local rank 0
    5. distance_to_local_medoid: ADE distance to Top-K medoid trajectory
    6. distance_to_local_barycenter: ADE distance to Top-K probability-weighted barycenter
    7. mean_distance_to_other_local_candidates: mean ADE distance to other Top-K candidates
    8. min_distance_to_other_local_candidate: minimum ADE distance to another Top-K candidate
    9. local_density: count of other Top-K candidates within 0.5m threshold
    
    Args:
        topk_logits: (N, K) native confidence logits
        topk_cands: (N, K, 12, 2) candidate trajectory coordinates
        batch_size: batch size for vectorized distance computations
        
    Returns:
        gl: (N, K, 9) array of float32 features
    """
    N, K = topk_logits.shape

    # 1. Local Rank
    ranks = np.broadcast_to(np.arange(K, dtype=np.float32)[None, :], (N, K))

    # 2. Logit gap to top-1
    logit_gap_top1 = topk_logits[:, 0:1] - topk_logits

    # 3. Logit gap to mean
    logit_gap_mean = np.mean(topk_logits, axis=-1, keepdims=True) - topk_logits

    # 4. Prob gap to top-1 (local softmax)
    shifted = topk_logits - np.max(topk_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    local_probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)
    prob_gap_top1 = local_probs[:, 0:1] - local_probs

    # Distance-based GL features
    if topk_cands is not None:
        dist_medoid_list = []
        dist_barycenter_list = []
        mean_dist_list = []
        min_dist_list = []
        density_list = []

        for start in range(0, N, batch_size):
            b_cands = topk_cands[start : start + batch_size]  # (B, K, 12, 2)
            b_probs = local_probs[start : start + batch_size]  # (B, K)
            B = len(b_cands)

            # Pairwise distance matrix between K candidates: (B, K, K)
            # diff shape: (B, K, K, 12, 2)
            diff = b_cands[:, :, None, :, :] - b_cands[:, None, :, :, :]
            # ADE distance matrix: average over 12 timesteps
            dist_mat = np.mean(np.linalg.norm(diff, axis=-1), axis=-1)  # (B, K, K)

            # Distance to medoid: candidate with minimum sum of distances
            sum_dists = np.sum(dist_mat, axis=-1)  # (B, K)
            medoid_idx = np.argmin(sum_dists, axis=-1)  # (B,)
            b_medoid_dist = np.take_along_axis(dist_mat, medoid_idx[:, None, None], axis=1).squeeze(1)  # (B, K)
            dist_medoid_list.append(b_medoid_dist)

            # Barycenter trajectory: (B, 12, 2)
            barycenter = np.sum(b_probs[:, :, None, None] * b_cands, axis=1)  # (B, 12, 2)
            # Distance of each candidate to barycenter: (B, K)
            b_bary_dist = np.mean(np.linalg.norm(b_cands - barycenter[:, None, :, :], axis=-1), axis=-1)
            dist_barycenter_list.append(b_bary_dist)

            # Pairwise metrics to OTHER candidates (fill diagonal with inf / nan)
            diag_idx = np.arange(K)
            dist_mat_no_diag = dist_mat.copy()
            dist_mat_no_diag[:, diag_idx, diag_idx] = np.nan

            b_mean_dist = np.nanmean(dist_mat_no_diag, axis=-1)  # (B, K)
            mean_dist_list.append(b_mean_dist)

            b_min_dist = np.nanmin(dist_mat_no_diag, axis=-1)  # (B, K)
            min_dist_list.append(b_min_dist)

            # Local density: count of other candidates within 0.5m
            dist_mat_no_diag_zero = dist_mat.copy()
            dist_mat_no_diag_zero[:, diag_idx, diag_idx] = 999.0
            b_density = np.sum(dist_mat_no_diag_zero < 0.5, axis=-1).astype(np.float32)  # (B, K)
            density_list.append(b_density)

        dist_to_medoid = np.concatenate(dist_medoid_list, axis=0)
        dist_to_barycenter = np.concatenate(dist_barycenter_list, axis=0)
        mean_dist_others = np.concatenate(mean_dist_list, axis=0)
        min_dist_others = np.concatenate(min_dist_list, axis=0)
        local_density = np.concatenate(density_list, axis=0)

    else:
        # Graceful fallback if coordinates are unavailable
        dist_to_medoid = np.zeros((N, K), dtype=np.float32)
        dist_to_barycenter = np.zeros((N, K), dtype=np.float32)
        mean_dist_others = np.zeros((N, K), dtype=np.float32)
        min_dist_others = np.zeros((N, K), dtype=np.float32)
        local_density = np.zeros((N, K), dtype=np.float32)

    gl = np.stack([
        ranks,
        logit_gap_top1,
        logit_gap_mean,
        prob_gap_top1,
        dist_to_medoid,
        dist_to_barycenter,
        mean_dist_others,
        min_dist_others,
        local_density,
    ], axis=-1).astype(np.float32)

    return gl


def assemble_local_features(
    fold_data: Dict[str, np.ndarray],
    topk_pool: Dict[str, np.ndarray],
    groups: List[str],
) -> np.ndarray:
    """
    Assembles gathered base feature groups (g0, g1, g2, g4) and local GL features.
    
    Args:
        fold_data: raw fold data dictionary (with g0, g1, g2, g3, g4)
        topk_pool: dictionary from build_native_topk_pool (with 'indices', 'logits', optional 'candidates')
        groups: list of group identifiers, e.g. ['g0', 'g1', 'g2', 'gl']
        
    Returns:
        concatenated feature matrix of shape (N, K, D_total)
    """
    topk_idx = topk_pool["indices"]  # (N, K)
    N, K = topk_idx.shape
    features = []

    for g in groups:
        g_lower = g.lower()
        if g_lower == "gl":
            topk_logits = topk_pool["logits"]
            topk_cands = topk_pool.get("candidates")
            gl = compute_gl_features(topk_logits, topk_cands)
            features.append(gl)
        elif g_lower in ("g0", "g1", "g2", "g3", "g4"):
            base_feat = fold_data[g_lower]  # (N, 20, D_base)
            # Gather Top-K candidates
            gathered = np.take_along_axis(base_feat, topk_idx[:, :, None], axis=1)  # (N, K, D_base)
            features.append(gathered)
        else:
            raise ValueError(f"Unknown feature group: {g}")

    return np.concatenate(features, axis=-1).astype(np.float32)
