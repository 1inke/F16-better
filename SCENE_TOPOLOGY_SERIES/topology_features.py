"""Candidate-to-topology measurements used by EXP-178A."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def sample_grid(field: np.ndarray, points_px: np.ndarray, image_wh: np.ndarray) -> np.ndarray:
    height, width = field.shape[:2]
    x = np.clip(np.rint(points_px[..., 0] * (width - 1) / max(float(image_wh[0] - 1), 1)).astype(int), 0, width - 1)
    y = np.clip(np.rint(points_px[..., 1] * (height - 1) / max(float(image_wh[1] - 1), 1)).astype(int), 0, height - 1)
    return field[y, x]


def candidate_features(bank: dict[str, np.ndarray], scene_id: int, candidates_px: np.ndarray, past_px: np.ndarray):
    size = bank["image_wh"][scene_id]
    if not bank["skeleton"][scene_id].any():
        zeros = np.zeros(len(candidates_px), np.float32)
        far = np.full(len(candidates_px), float(np.hypot(*bank["skeleton"][scene_id].shape)), np.float32)
        return {
            "mean_skeleton_distance": far.copy(), "endpoint_skeleton_distance": far.copy(),
            "max_skeleton_deviation": far.copy(), "reachable_endpoint": zeros.copy(),
            "path_connectivity": zeros.copy(), "direction_consistency": zeros.copy(),
            "geodesic_path_ratio": zeros.copy(), "mean_walkability": zeros.copy(),
            "topology_consistency": zeros.copy(),
        }
    distance = sample_grid(bank["distance_to_skeleton"][scene_id], candidates_px, size)
    probability = sample_grid(bank["probability"][scene_id].astype(np.float32), candidates_px, size)
    nearest_y = sample_grid(bank["nearest_y"][scene_id], candidates_px, size).astype(int)
    nearest_x = sample_grid(bank["nearest_x"][scene_id], candidates_px, size).astype(int)
    component = bank["component"][scene_id][nearest_y, nearest_x]
    start_y = int(sample_grid(bank["nearest_y"][scene_id], past_px[-1:], size)[0])
    start_x = int(sample_grid(bank["nearest_x"][scene_id], past_px[-1:], size)[0])
    start_component = int(bank["component"][scene_id][start_y, start_x])
    endpoint_component = component[:, -1]
    close_to_topology = distance <= 6.0
    connectivity = ((component == start_component) & close_to_topology).mean(axis=1)
    reachable = ((endpoint_component == start_component) & close_to_topology[:, -1]).astype(np.float32)
    displacement = candidates_px[:, -1] - candidates_px[:, 0]
    past_heading = past_px[-1] - past_px[max(0, len(past_px) - 4)]
    heading_norm = max(float(np.linalg.norm(past_heading)), 1e-6)
    displacement_norm = np.maximum(np.linalg.norm(displacement, axis=1), 1e-6)
    direction = (displacement @ past_heading) / (displacement_norm * heading_norm)
    path_length = np.linalg.norm(np.diff(candidates_px, axis=1), axis=-1).sum(axis=1)
    direct_length = np.linalg.norm(candidates_px[:, -1] - past_px[-1], axis=-1)
    geodesic_ratio = direct_length / np.maximum(path_length, 1e-6)
    raw = {
        "mean_skeleton_distance": distance.mean(axis=1),
        "endpoint_skeleton_distance": distance[:, -1],
        "max_skeleton_deviation": distance.max(axis=1),
        "reachable_endpoint": reachable,
        "path_connectivity": connectivity.astype(np.float32),
        "direction_consistency": direction.astype(np.float32),
        "geodesic_path_ratio": geodesic_ratio.astype(np.float32),
        "mean_walkability": probability.mean(axis=1),
    }
    def row_z(value):
        value = np.asarray(value, np.float64); return (value - value.mean()) / max(float(value.std()), 1e-6)
    raw["topology_consistency"] = (
        -0.30 * row_z(raw["mean_skeleton_distance"])
        -0.20 * row_z(raw["endpoint_skeleton_distance"])
        -0.10 * row_z(raw["max_skeleton_deviation"])
        +0.15 * row_z(raw["reachable_endpoint"])
        +0.10 * row_z(raw["path_connectivity"])
        +0.10 * row_z(raw["direction_consistency"])
        +0.05 * row_z(raw["mean_walkability"])
    ).astype(np.float32)
    return raw


def row_spearman(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    a, b = rankdata(values, axis=1), rankdata(target, axis=1)
    a -= a.mean(axis=1, keepdims=True); b -= b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    result = np.full(len(a), np.nan, np.float64); valid = denominator > 1e-12
    result[valid] = (a[valid] * b[valid]).sum(axis=1) / denominator[valid]
    return result
