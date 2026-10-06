"""Core utilities for conditional empirical future quantization (CEQ1).

CEQ1 is deliberately not another repulsion loss.  It estimates a conditional
future distribution from training-only neighbours and uses weighted Lloyd
quantization to allocate 20 trajectories according to probability mass.
Cluster mass is also the candidate score, so geometry and selection share one
distributional object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import Tensor


EPS = 1.0e-8


def _local_xy(points: np.ndarray, origin: np.ndarray, heading: np.ndarray) -> np.ndarray:
    delta = points - origin
    x = delta[..., 0] * heading[..., 0, None] + delta[..., 1] * heading[..., 1, None]
    y = -delta[..., 0] * heading[..., 1, None] + delta[..., 1] * heading[..., 0, None]
    return np.stack((x, y), axis=-1)


def build_observation_features(
    coordinates: np.ndarray,
    num_valid: np.ndarray,
    past_frames: int,
    future_frames: int,
    max_social: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observation-only features and focal future in a heading frame.

    ``coordinates`` has shape [sample, agent, time, xy].  Features contain the
    complete focal history, short/long motion terms, and the four nearest valid
    neighbours' current relative position and relative velocity.  Future data
    is returned separately and never enters feature construction.
    """
    coordinates = np.asarray(coordinates, dtype=np.float32)
    num_valid = np.asarray(num_valid, dtype=np.int64)
    if coordinates.ndim != 4 or coordinates.shape[-1] != 2:
        raise ValueError(f"coordinates must be [N,A,T,2], got {coordinates.shape}")
    if coordinates.shape[2] < past_frames + future_frames:
        raise ValueError("trajectory array is shorter than past+future contract")

    past = coordinates[:, :, :past_frames]
    focal = past[:, 0]
    origin = focal[:, -1]
    lag = min(3, past_frames - 1)
    direction = focal[:, -1] - focal[:, -1 - lag]
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    fallback = focal[:, -1] - focal[:, -2]
    fallback_norm = np.linalg.norm(fallback, axis=-1, keepdims=True)
    direction = np.where(norm > 1.0e-4, direction, fallback)
    norm = np.where(norm > 1.0e-4, norm, fallback_norm)
    default = np.zeros_like(direction)
    default[:, 0] = 1.0
    heading = np.where(norm > 1.0e-4, direction / np.maximum(norm, EPS), default)

    local_focal = _local_xy(focal, origin[:, None, :], heading)
    short = local_focal[:, -1] - local_focal[:, -2]
    long = (local_focal[:, -1] - local_focal[:, -1 - lag]) / float(lag)
    acceleration = short - long
    focal_features = np.concatenate(
        (local_focal.reshape(len(local_focal), -1), short, long, acceleration), axis=1
    )

    agents = past.shape[1]
    last = past[:, :, -1]
    previous = past[:, :, -2]
    relative_last = last - origin[:, None]
    relative_velocity = (last - previous) - (focal[:, -1] - focal[:, -2])[:, None]
    relative_last_local = _local_xy(
        relative_last + origin[:, None], origin[:, None], heading
    )
    relative_velocity_local = _local_xy(
        relative_velocity + origin[:, None], origin[:, None], heading
    )
    distance = np.linalg.norm(relative_last, axis=-1)
    agent_ids = np.arange(agents)[None]
    valid = (agent_ids > 0) & (agent_ids < num_valid[:, None])
    sortable = np.where(valid, distance, np.inf)
    nearest = np.argsort(sortable, axis=1)[:, :max_social]
    rows = np.arange(len(coordinates))[:, None]
    chosen_pos = relative_last_local[rows, nearest]
    chosen_vel = relative_velocity_local[rows, nearest]
    chosen_distance = distance[rows, nearest, None]
    chosen_valid = np.isfinite(sortable[rows, nearest])[:, :, None]
    social = np.concatenate((chosen_pos, chosen_vel, chosen_distance), axis=-1)
    social = np.where(chosen_valid, social, 0.0).reshape(len(coordinates), -1)

    future = coordinates[:, 0, past_frames : past_frames + future_frames]
    local_future = _local_xy(future, origin[:, None], heading)
    features = np.concatenate((focal_features, social), axis=1).astype(np.float32)
    if not np.isfinite(features).all() or not np.isfinite(local_future).all():
        raise RuntimeError("non-finite CEQ feature or future")
    return features, local_future.astype(np.float32)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardizer":
        mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(scale, 1.0e-3)
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.scale).astype(np.float32)


def weighted_lloyd(
    trajectories: Tensor,
    weights: Tensor,
    components: int = 20,
    iterations: int = 8,
) -> tuple[Tensor, Tensor]:
    """Batched weighted k-means in full-trajectory space.

    Initial centres are the highest conditional-similarity neighbours.  There
    is no pairwise-distance reward and no fixed family quota.
    """
    if trajectories.ndim != 4 or trajectories.shape[-1] != 2:
        raise ValueError(str(tuple(trajectories.shape)))
    batch, neighbours, steps, _ = trajectories.shape
    if neighbours < components:
        raise ValueError("neighbours must be >= components")
    flat = trajectories.reshape(batch, neighbours, steps * 2)
    order = torch.argsort(weights, dim=1, descending=True)
    initial = order[:, :components]
    centres = flat.gather(1, initial[..., None].expand(-1, -1, flat.shape[-1])).clone()
    for _ in range(iterations):
        distance = torch.cdist(flat, centres).square()
        assignment = distance.argmin(dim=-1)
        membership = torch.nn.functional.one_hot(
            assignment, num_classes=components
        ).to(flat.dtype)
        weighted = membership * weights[..., None]
        mass = weighted.sum(dim=1)
        updated = torch.einsum("bnc,bnd->bcd", weighted, flat) / mass.clamp_min(EPS)[..., None]
        centres = torch.where((mass > EPS)[..., None], updated, centres)
    distance = torch.cdist(flat, centres).square()
    assignment = distance.argmin(dim=-1)
    membership = torch.nn.functional.one_hot(
        assignment, num_classes=components
    ).to(flat.dtype)
    mass = (membership * weights[..., None]).sum(dim=1)
    mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(EPS)
    return centres.reshape(batch, components, steps, 2), mass


class ConditionalFutureQuantizer:
    def __init__(
        self,
        bank_features: np.ndarray,
        bank_futures: np.ndarray,
        bank_ids: np.ndarray,
        device: torch.device,
    ) -> None:
        self.device = device
        self.features = torch.as_tensor(bank_features, device=device)
        self.futures = torch.as_tensor(bank_futures, device=device)
        self.ids = torch.as_tensor(bank_ids.astype(np.int64), device=device)

    def generate(
        self,
        query_features: np.ndarray,
        query_ids: np.ndarray,
        neighbours: int,
        temperature: float,
        components: int = 20,
        iterations: int = 8,
        batch_size: int = 64,
        exclusion_gap: int = 12,
    ) -> tuple[np.ndarray, np.ndarray]:
        if neighbours < components or neighbours > len(self.features):
            raise ValueError("invalid neighbour count")
        outputs, logits = [], []
        for start in range(0, len(query_features), batch_size):
            stop = min(start + batch_size, len(query_features))
            query = torch.as_tensor(query_features[start:stop], device=self.device)
            query_id = torch.as_tensor(query_ids[start:stop], device=self.device)
            distance = (
                query.square().sum(dim=1, keepdim=True)
                + self.features.square().sum(dim=1)[None]
                - 2.0 * query @ self.features.T
            ).clamp_min_(0.0)
            if exclusion_gap >= 0:
                overlap = (query_id[:, None] - self.ids[None]).abs() <= exclusion_gap
                distance = distance.masked_fill(overlap, torch.inf)
            top_distance, top_index = torch.topk(
                distance, k=neighbours, dim=1, largest=False, sorted=True
            )
            if not torch.isfinite(top_distance).all():
                raise RuntimeError("too few eligible training neighbours")
            radial = torch.sqrt(top_distance + EPS)
            adaptive_scale = radial.median(dim=1, keepdim=True).values.clamp_min(1.0e-3)
            weights = torch.softmax(-radial / (adaptive_scale * temperature), dim=1)
            neighbour_futures = self.futures[top_index]
            candidate, mass = weighted_lloyd(
                neighbour_futures, weights, components=components, iterations=iterations
            )
            outputs.append(candidate.cpu().numpy())
            logits.append(torch.log(mass.clamp_min(EPS)).cpu().numpy())
        return np.concatenate(outputs).astype(np.float32), np.concatenate(logits).astype(np.float32)


def raw_metrics(
    sample_ids: np.ndarray,
    candidates: np.ndarray,
    target: np.ndarray,
    logits: np.ndarray,
) -> dict[str, np.ndarray]:
    error = np.linalg.norm(candidates - target[:, None], axis=-1)
    return {
        "sample_id": np.asarray(sample_ids, dtype=np.int64),
        "candidate": np.asarray(candidates, dtype=np.float32),
        "ade": error.mean(axis=-1).astype(np.float32),
        "fde": error[..., -1].astype(np.float32),
        "logits": np.asarray(logits, dtype=np.float32),
    }


def calibration_objective(derived: dict[str, np.ndarray]) -> float:
    j = derived["minade"] + 0.5 * derived["minfde"]
    return float(j.mean() + 0.25 * derived["selected_ade"].mean())


def candidate_duplicate_rate(candidates: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    endpoints = candidates[:, :, -1]
    left, right = np.triu_indices(endpoints.shape[1], k=1)
    distance = np.linalg.norm(endpoints[:, left] - endpoints[:, right], axis=-1)
    return (distance < threshold).mean(axis=1)


def observation_motion_ratio(
    coordinates: np.ndarray, past_frames: int
) -> np.ndarray:
    focal = np.asarray(coordinates[:, 0, :past_frames], dtype=np.float32)
    steps = np.diff(focal, axis=1)
    lag = min(3, steps.shape[1])
    short = steps[:, -1]
    long = steps[:, -lag:].mean(axis=1)
    acceleration = np.linalg.norm(short - long, axis=-1)
    speed = np.median(np.linalg.norm(steps[:, -lag:], axis=-1), axis=1)
    return acceleration / np.maximum(speed, 0.02)


def split_fit_calibration(
    train_ids: np.ndarray, seed: int, calibration_count: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(train_ids, dtype=np.int64))
    count = min(calibration_count, max(1, len(order) // 10))
    calibration = np.sort(order[:count])
    fit = np.sort(order[count:])
    return fit, calibration


def parameter_grid(
    neighbours: Iterable[int] = (64, 128, 256),
    temperatures: Iterable[float] = (0.5, 1.0, 2.0),
) -> list[dict[str, float | int]]:
    return [
        {"neighbours": int(k), "temperature": float(t)}
        for k in neighbours for t in temperatures
    ]
