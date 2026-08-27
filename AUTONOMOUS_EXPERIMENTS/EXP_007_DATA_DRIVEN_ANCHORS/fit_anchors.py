"""Fit validation-independent local endpoint residual anchors."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy\processed_data_diverse\univ_data_train.npy")
raw = np.load(DATA, mmap_mode="r")
scene = np.asarray(raw[:, 0, 0, 0], dtype=np.int64)
indices = np.flatnonzero(scene != 1)
coordinates = np.asarray(raw[indices, 0, :, -2:], dtype=np.float32)
past = coordinates[:, :8]
future = coordinates[:, 8:20]
step = past[:, -1] - past[:, -2]
speed = np.linalg.norm(step, axis=-1, keepdims=True)
heading = np.zeros_like(step)
heading[:, 0] = 1.0
moving = speed[:, 0] > 0.02
heading[moving] = step[moving] / np.maximum(speed[moving], 1.0e-6)
cv_endpoint = step * 12.0
actual_endpoint = future[:, -1] - past[:, -1]
residual = actual_endpoint - cv_endpoint
local = np.stack(
    (
        residual[:, 0] * heading[:, 0] + residual[:, 1] * heading[:, 1],
        -residual[:, 0] * heading[:, 1] + residual[:, 1] * heading[:, 0],
    ),
    axis=-1,
)
radius = np.linalg.norm(local, axis=-1)
local = local[radius <= np.quantile(radius, 0.995)]

rng = np.random.default_rng(20260821)
k = 19
centers = [local[rng.integers(len(local))]]
for _ in range(1, k):
    distance_sq = np.min(
        np.sum((local[:, None] - np.asarray(centers)[None]) ** 2, axis=-1), axis=1
    )
    probability = distance_sq / np.maximum(distance_sq.sum(), 1.0e-12)
    centers.append(local[rng.choice(len(local), p=probability)])
centers = np.asarray(centers, dtype=np.float32)
for _ in range(100):
    assignment = np.argmin(
        np.sum((local[:, None] - centers[None]) ** 2, axis=-1), axis=1
    )
    updated = np.stack(
        [local[assignment == index].mean(axis=0) if np.any(assignment == index) else centers[index] for index in range(k)]
    )
    if np.max(np.abs(updated - centers)) < 1.0e-5:
        centers = updated
        break
    centers = updated

anchors = np.concatenate((np.zeros((1, 2), dtype=np.float32), centers), axis=0)
np.save(ROOT / "anchors_scene1_train_only.npy", anchors)
np.savez_compressed(
    ROOT / "anchor_fit_metadata.npz", anchors=anchors,
    validation_scene=np.asarray([1]), train_indices=indices,
    samples=np.asarray([len(local)]),
)
print(anchors)

