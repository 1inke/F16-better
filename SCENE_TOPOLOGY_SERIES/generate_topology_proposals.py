"""Deterministic branch proposals along a cached walkability skeleton."""

from __future__ import annotations

import heapq
import math
import numpy as np


NEIGHBOURS = ((-1, -1, math.sqrt(2)), (-1, 0, 1), (-1, 1, math.sqrt(2)),
              (0, -1, 1), (0, 1, 1), (1, -1, math.sqrt(2)),
              (1, 0, 1), (1, 1, math.sqrt(2)))


def _to_grid(point, image_wh, shape):
    return np.asarray([
        point[0] * (shape[1] - 1) / max(float(image_wh[0] - 1), 1),
        point[1] * (shape[0] - 1) / max(float(image_wh[1] - 1), 1),
    ], np.float32)


def _to_pixels(points, image_wh, shape):
    points = np.asarray(points, np.float32)
    output = points.copy()
    output[:, 0] *= max(float(image_wh[0] - 1), 1) / (shape[1] - 1)
    output[:, 1] *= max(float(image_wh[1] - 1), 1) / (shape[0] - 1)
    return output


def _dijkstra(skeleton, probability, start, limit):
    height, width = skeleton.shape
    distance = {start: 0.0}; parent = {start: None}; queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node] or cost > limit:
            continue
        y, x = node
        for dy, dx, step in NEIGHBOURS:
            neighbour = (y + dy, x + dx)
            if not (0 <= neighbour[0] < height and 0 <= neighbour[1] < width) or not skeleton[neighbour]:
                continue
            weighted = step * (1.0 + 0.15 * (1.0 - float(probability[neighbour])))
            candidate = cost + weighted
            if candidate <= limit and candidate < distance.get(neighbour, float("inf")):
                distance[neighbour] = candidate; parent[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    return distance, parent


def _path(parent, endpoint):
    nodes = []
    while endpoint is not None:
        nodes.append((endpoint[1], endpoint[0])); endpoint = parent[endpoint]
    return np.asarray(nodes[::-1], np.float32)


def _resample(path, steps, target_step):
    if len(path) < 2:
        return np.repeat(path[:1], steps, axis=0)
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    targets = np.minimum(np.arange(1, steps + 1) * target_step, cumulative[-1])
    return np.stack((np.interp(targets, cumulative, path[:, 0]), np.interp(targets, cumulative, path[:, 1])), axis=-1)


def generate_proposals(bank: dict[str, np.ndarray], scene_id: int, past_px: np.ndarray,
                       horizon: int = 12, branches: int = 4, safety_factor: float = 1.5):
    skeleton = bank["skeleton"][scene_id].astype(bool)
    probability = bank["probability"][scene_id].astype(np.float32)
    nearest_y, nearest_x = bank["nearest_y"][scene_id], bank["nearest_x"][scene_id]
    image_wh = bank["image_wh"][scene_id]; shape = skeleton.shape
    if not skeleton.any():
        return np.empty((0, horizon, 2), np.float32), {
            "reachable_nodes": 0, "branches": 0,
            "nearest_skeleton_distance_grid": None, "lmax_grid": None,
        }
    start_float = _to_grid(past_px[-1], image_wh, shape)
    sx = int(np.clip(round(float(start_float[0])), 0, shape[1] - 1)); sy = int(np.clip(round(float(start_float[1])), 0, shape[0] - 1))
    start = (int(nearest_y[sy, sx]), int(nearest_x[sy, sx]))
    observed_steps = np.linalg.norm(np.diff(past_px, axis=0), axis=1)
    speed_px = float(np.median(observed_steps[-3:])) if len(observed_steps) else 1.0
    scale = 0.5 * ((shape[1] - 1) / max(float(image_wh[0] - 1), 1) + (shape[0] - 1) / max(float(image_wh[1] - 1), 1))
    speed_grid = max(speed_px * scale, 0.6)
    limit = max(speed_grid * horizon * safety_factor, 5.0)
    distance, parent = _dijkstra(skeleton, probability, start, limit)
    if len(distance) < 2:
        return np.empty((0, horizon, 2), np.float32), {"reachable_nodes": len(distance), "branches": 0}
    heading = _to_grid(past_px[-1], image_wh, shape) - _to_grid(past_px[max(0, len(past_px) - 4)], image_wh, shape)
    heading /= max(float(np.linalg.norm(heading)), 1e-6)
    maximum = max(distance.values()); frontier = [node for node, value in distance.items() if value >= 0.70 * maximum]
    ranked = []
    for node in frontier:
        vector = np.asarray([node[1] - start[1], node[0] - start[0]], np.float32)
        norm = max(float(np.linalg.norm(vector)), 1e-6); direction = vector / norm
        heading_score = float(direction @ heading)
        score = distance[node] * (0.75 + 0.25 * float(probability[node])) * (1.0 + 0.20 * heading_score)
        ranked.append((score, math.atan2(float(direction[1]), float(direction[0])), node))
    ranked.sort(reverse=True)
    selected = []
    for _, angle, node in ranked:
        if all(abs(math.atan2(math.sin(angle - old), math.cos(angle - old))) >= math.radians(25) for old, _ in selected):
            selected.append((angle, node))
        if len(selected) == branches:
            break
    if not selected:
        selected = [(ranked[0][1], ranked[0][2])]
    proposals = []
    for _, endpoint in selected:
        path_px = _to_pixels(_path(parent, endpoint), image_wh, shape)
        sampled = _resample(path_px, horizon, max(speed_px, 0.5))
        proposals.append(sampled)
    return np.asarray(proposals, np.float32), {
        "reachable_nodes": len(distance), "branches": len(proposals),
        "nearest_skeleton_distance_grid": float(np.linalg.norm(start_float - np.asarray([start[1], start[0]]))),
        "lmax_grid": limit,
    }
