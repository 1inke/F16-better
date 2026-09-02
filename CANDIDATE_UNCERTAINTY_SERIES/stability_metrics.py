"""Metric helpers for the EXP-179A perturbation audit."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


FEATURES = (
    "trajectory_U", "endpoint_U", "teacher_U_0p7_0p3", "logit_variance",
    "rank_variance", "v2_correction_variance", "endpoint_direction_variance",
    "p2_trajectory_U", "p2_endpoint_U",
)


def ranks_descending(logits: np.ndarray) -> np.ndarray:
    order = np.argsort(-logits, axis=-1, kind="mergesort")
    output = np.empty_like(order, dtype=np.float32)
    rows = np.arange(logits.shape[0])[:, None]
    output[rows, order] = np.arange(logits.shape[1], dtype=np.float32)[None]
    return output


def derive(primary_predictions, primary_logits, primary_corrections, primary_directions,
           p2_prediction, p2_direction):
    """All arrays use [M,B,K,...], with mode identity fixed across M."""
    reference = primary_predictions[0]
    displacement = np.linalg.norm(primary_predictions - reference[None], axis=-1)
    trajectory_u = displacement.mean(axis=-1).mean(axis=0)
    endpoint_u = displacement[..., -1].mean(axis=0)
    logit_variance = primary_logits.var(axis=0)
    ranks = np.stack([ranks_descending(x) for x in primary_logits])
    rank_variance = ranks.var(axis=0)
    correction_delta = np.linalg.norm(primary_corrections - primary_corrections[0:1], axis=-1)
    correction_variance = np.square(correction_delta).mean(axis=(0, 3))
    direction_delta = np.arctan2(
        np.sin(primary_directions - primary_directions[0:1]),
        np.cos(primary_directions - primary_directions[0:1]),
    )
    direction_variance = np.square(direction_delta).mean(axis=0)
    p2_displacement = np.linalg.norm(p2_prediction - reference, axis=-1)
    return {
        "trajectory_U": trajectory_u.astype(np.float32),
        "endpoint_U": endpoint_u.astype(np.float32),
        "teacher_U_0p7_0p3": (0.7 * trajectory_u + 0.3 * endpoint_u).astype(np.float32),
        "logit_variance": logit_variance.astype(np.float32),
        "rank_variance": rank_variance.astype(np.float32),
        "v2_correction_variance": correction_variance.astype(np.float32),
        "endpoint_direction_variance": direction_variance.astype(np.float32),
        "p2_trajectory_U": p2_displacement.mean(axis=-1).astype(np.float32),
        "p2_endpoint_U": p2_displacement[..., -1].astype(np.float32),
    }


def per_sample_spearman(feature: np.ndarray, target: np.ndarray):
    a = rankdata(feature, axis=1, method="average"); b = rankdata(target, axis=1, method="average")
    a -= a.mean(axis=1, keepdims=True); b -= b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    valid = denominator > 1e-12; output = np.full(len(a), np.nan, np.float64)
    output[valid] = (a[valid] * b[valid]).sum(axis=1) / denominator[valid]
    return output, valid


def standardized_logits(logits: np.ndarray):
    return (logits - logits.mean(axis=1, keepdims=True)) / np.maximum(logits.std(axis=1, keepdims=True), 1e-6)


def fit_score_calibrator(logits: np.ndarray, quality: np.ndarray, bins: int = 25):
    score = standardized_logits(logits).reshape(-1)
    target = (quality - quality.mean(axis=1, keepdims=True)).reshape(-1)
    edges = np.unique(np.quantile(score, np.linspace(0, 1, bins + 1)))
    index = np.clip(np.digitize(score, edges[1:-1]), 0, max(len(edges) - 2, 0))
    x, y = [], []
    for position in range(max(len(edges) - 1, 1)):
        mask = index == position
        if mask.any():
            x.append(float(score[mask].mean())); y.append(float(target[mask].mean()))
    if len(x) < 2:
        return np.asarray([-1.0, 1.0]), np.asarray([target.mean(), target.mean()])
    return np.asarray(x), np.asarray(y)


def score_control(logits: np.ndarray, calibrators):
    score = standardized_logits(logits)
    return np.mean([np.interp(score, x, y, left=y[0], right=y[-1]) for x, y in calibrators], axis=0)
