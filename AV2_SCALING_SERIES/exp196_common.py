"""Shared model, dataset, metrics, and bootstrap utilities for EXP-196."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "AV2_TEMPORAL_BRIDGE_SERIES"))

from exp195_common import build_model, model_forward, normalize_to_focal_heading  # noqa: E402,F401

from exp196_constants import FUTURE_HZ, FUTURE_STEPS, VARIANTS  # noqa: E402


class ScalingArrayDataset(Dataset):
    def __init__(self, data_path: Path, num_path: Path, name: str):
        self.data = np.load(data_path, mmap_mode="r")
        self.num = np.load(num_path, mmap_mode="r")
        self.spec = VARIANTS[name]
        expected = (
            int(self.spec["max_agents"]),
            int(self.spec["history_steps"]) + FUTURE_STEPS,
        )
        if tuple(self.data.shape[1:3]) != expected:
            raise ValueError(f"{data_path}: shape {self.data.shape}; expected (*,{expected[0]},{expected[1]},3)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = np.asarray(self.data[index, :, :, -2:], dtype=np.float32)
        history = int(self.spec["history_steps"])
        return (
            torch.from_numpy(row[:, :history].copy()),
            torch.from_numpy(row[:, history:].copy()),
            torch.tensor(min(int(self.num[index]), int(self.spec["max_agents"])), dtype=torch.int64),
            torch.tensor(index, dtype=torch.int64),
        )


def load_metadata(path: Path, length: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "samples" in payload:
        payload = payload["samples"]
    if len(payload) != length:
        raise ValueError(f"Metadata length {len(payload)} != dataset length {length}: {path}")
    return payload


def prefix_ade(prediction: np.ndarray, target: np.ndarray, horizon: float) -> np.ndarray:
    times = np.arange(1, FUTURE_STEPS + 1, dtype=np.float32) / FUTURE_HZ
    indices = np.flatnonzero(times <= horizon + 1e-7)
    pred_parts = [prediction[..., index, :] for index in indices]
    target_parts = [target[..., index, :] for index in indices]
    if not np.isclose(times[indices[-1]], horizon):
        right = int(np.searchsorted(times, horizon))
        left_time = 0.0 if right == 0 else float(times[right - 1])
        left_pred = np.zeros_like(prediction[..., 0, :]) if right == 0 else prediction[..., right - 1, :]
        left_target = np.zeros_like(target[..., 0, :]) if right == 0 else target[..., right - 1, :]
        alpha = (horizon - left_time) / (float(times[right]) - left_time)
        pred_parts.append(left_pred + alpha * (prediction[..., right, :] - left_pred))
        target_parts.append(left_target + alpha * (target[..., right, :] - left_target))
    pred = np.stack(pred_parts, axis=-2)
    truth = np.stack(target_parts, axis=-2)
    return np.linalg.norm(pred - truth, axis=-1).mean(axis=-1)


def per_target_metrics(prediction: np.ndarray, target: np.ndarray, logits: np.ndarray) -> dict[str, np.ndarray]:
    errors = np.linalg.norm(prediction - target[:, None], axis=-1)
    ade = errors.mean(axis=-1)
    fde = errors[..., -1]
    rows = np.arange(len(ade))
    selected = logits.argmax(axis=-1)
    oracle = ade.argmin(axis=-1)
    top6 = np.argsort(-logits, axis=-1)[:, :6]
    top6_ade = np.take_along_axis(ade, top6, axis=1)
    top6_fde = np.take_along_axis(fde, top6, axis=1)
    result = {
        "minADE20": ade.min(axis=1),
        "minFDE20": fde.min(axis=1),
        "selected_ADE": ade[rows, selected],
        "selected_FDE": fde[rows, selected],
        "mean_regret": ade[rows, selected] - ade.min(axis=1),
        "hit_best_ade": (selected == oracle).astype(np.float32),
        "minADE6": top6_ade.min(axis=1),
        "minFDE6": top6_fde.min(axis=1),
        "MR6_at_2m": (top6_fde.min(axis=1) > 2.0).astype(np.float32),
    }
    for horizon in (1.0, 2.0, 3.0, 4.8):
        by_mode = prefix_ade(prediction, target[:, None], horizon)
        result[f"minADE20_at_{horizon:g}s"] = by_mode.min(axis=1)
        result[f"selected_ADE_at_{horizon:g}s"] = by_mode[rows, selected]
    return result


def summarize(per_target: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, float]:
    if indices is None:
        indices = np.arange(len(next(iter(per_target.values()))))
    return {key: float(np.nanmean(values[indices])) for key, values in per_target.items()}


def scenario_bootstrap_difference(
    candidate: np.ndarray,
    baseline: np.ndarray,
    scenario_ids: np.ndarray,
    samples: int,
    seed: int = 2026,
) -> dict[str, float]:
    """Paired scenario bootstrap of relative improvement (positive is better)."""
    unique = np.unique(scenario_ids)
    grouped = {
        scene: np.flatnonzero(scenario_ids == scene)
        for scene in unique
    }
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([grouped[item] for item in chosen])
        base_mean = float(np.mean(baseline[selected]))
        values[index] = (base_mean - float(np.mean(candidate[selected]))) / max(base_mean, 1.0e-12)
    return {
        "mean_improvement": float((np.mean(baseline) - np.mean(candidate)) / max(float(np.mean(baseline)), 1.0e-12)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "bootstrap_samples": int(samples),
        "scenario_count": int(len(unique)),
    }


def temporal_weight_mass(history_steps: int, history_hz: float) -> dict[str, float]:
    gamma = 0.25 * 2.5 / float(history_hz)
    logits = gamma * np.arange(history_steps, dtype=np.float64)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    result = {}
    for seconds in (1.0, 2.0, 3.0, 5.0):
        count = min(history_steps, int(round(seconds * history_hz)))
        result[f"recent_{seconds:g}s_mass"] = float(weights[-count:].sum())
    result["oldest_2s_mass"] = float(weights[: min(history_steps, int(round(2.0 * history_hz)))].sum())
    return result
