"""Post-process walkability probabilities into cached topology fields."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import disk, opening, remove_small_holes, remove_small_objects, skeletonize


def _orientation_field(skeleton: np.ndarray, radius: int = 3) -> np.ndarray:
    """Estimate an unsigned local tangent for every skeleton pixel."""
    ys, xs = np.nonzero(skeleton)
    field = np.zeros((*skeleton.shape, 2), dtype=np.float32)
    points = np.stack((xs, ys), axis=1).astype(np.float32)
    if not len(points):
        return field
    lookup = {(int(y), int(x)) for y, x in zip(ys, xs)}
    for y, x in zip(ys, xs):
        neighbours = []
        for yy in range(max(0, y - radius), min(skeleton.shape[0], y + radius + 1)):
            for xx in range(max(0, x - radius), min(skeleton.shape[1], x + radius + 1)):
                if (yy, xx) in lookup and (yy != y or xx != x):
                    neighbours.append((xx, yy))
        if len(neighbours) < 2:
            continue
        centred = np.asarray(neighbours, np.float32) - np.asarray([x, y], np.float32)
        covariance = centred.T @ centred
        values, vectors = np.linalg.eigh(covariance)
        tangent = vectors[:, int(np.argmax(values))]
        if tangent[0] < 0 or (abs(tangent[0]) < 1e-8 and tangent[1] < 0):
            tangent = -tangent
        field[y, x] = tangent
    return field


def probability_to_topology(probability: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    mask = np.asarray(probability >= threshold, dtype=bool)
    mask = opening(mask, disk(1))
    mask = remove_small_holes(mask, area_threshold=64)
    mask = remove_small_objects(mask, min_size=32)
    skeleton = skeletonize(mask)
    # Keep the validation-selected threshold fixed for every scene.  An unseen
    # scene may legitimately produce no usable topology; record that condition
    # instead of silently introducing a per-test-scene fallback threshold.
    if not skeleton.any():
        height, width = skeleton.shape
        distance = np.full(skeleton.shape, np.hypot(height, width), np.float32)
        nearest = np.full((2, height, width), -1, np.int16)
        components = np.zeros(skeleton.shape, np.int16)
    else:
        distance, nearest = ndimage.distance_transform_edt(~skeleton, return_indices=True)
        components, _ = ndimage.label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    degree = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    degree = np.where(skeleton, degree - 1, 0).astype(np.uint8)
    return {
        "mask": mask.astype(np.uint8),
        "skeleton": skeleton.astype(np.uint8),
        "distance_to_skeleton": distance.astype(np.float32),
        "nearest_y": nearest[0].astype(np.int16),
        "nearest_x": nearest[1].astype(np.int16),
        "component": components.astype(np.int16),
        "degree": degree,
        "orientation": _orientation_field(skeleton),
    }
