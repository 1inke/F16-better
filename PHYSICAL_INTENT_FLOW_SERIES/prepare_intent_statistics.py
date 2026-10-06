"""Fit train-only physical-intent standardization for PIF1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASTRA_DEFAULT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
CONFIG = REPO / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
C00_RUN = (
    REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
    / "results" / "eth" / "v2" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def controls_from_coordinates(coordinates: np.ndarray, obs: int, speed_floor: float) -> np.ndarray:
    past = coordinates[:, :obs]
    future = coordinates[:, obs:obs + 12]
    short = past[:, -1] - past[:, -2]
    long = (past[:, -1] - past[:, -4]) / 3.0
    short_speed = np.linalg.norm(short, axis=-1, keepdims=True)
    long_speed = np.linalg.norm(long, axis=-1, keepdims=True)
    heading = np.where(short_speed > speed_floor, short, long)
    heading_norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    forward = heading / np.maximum(heading_norm, 1.0e-8)
    forward[heading_norm[:, 0] <= 1.0e-8] = np.array([1.0, 0.0])
    left = np.stack((-forward[:, 1], forward[:, 0]), axis=-1)
    speed = np.maximum(short_speed, long_speed).clip(min=speed_floor)
    anchors = future[:, [3, 7, 11]] - past[:, -1, None]
    previous = np.concatenate((np.zeros_like(anchors[:, :1]), anchors[:, :-1]), axis=1)
    segments = anchors - previous
    local_x = np.sum(segments * forward[:, None], axis=-1)
    local_y = np.sum(segments * left[:, None], axis=-1)
    lengths = np.maximum(np.linalg.norm(segments, axis=-1), 1.0e-6)
    log_ratio = np.log(lengths / np.maximum(speed * 4.0, 1.0e-6))
    angles = np.arctan2(local_y, local_x)
    turns = np.concatenate((angles[:, :1], wrap(angles[:, 1:] - angles[:, :-1])), axis=1)
    return np.stack(
        (log_ratio[:, 0], turns[:, 0], log_ratio[:, 1], turns[:, 1], log_ratio[:, 2], turns[:, 2]),
        axis=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astra-root", type=Path, default=ASTRA_DEFAULT)
    parser.add_argument("--output", type=Path, default=HERE / "calibration" / "intent_stats.json")
    args = parser.parse_args()
    data = args.astra_root / "datasets" / "eth_ucy" / "processed_data_diverse" / "eth_data_train.npy"
    split = C00_RUN / "split_indices.npz"
    for path in (data, split, CONFIG):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    obs = int(config["DATA"]["FREQUENCY"] * config["PREDICTION"]["OBS_TIME"])
    scale = float(config["PREDICTION"]["TRAJ_SCALE"])
    source = np.load(data, mmap_mode="r")
    train_ids = np.load(split)["train"].astype(np.int64)
    coordinates = np.asarray(source[train_ids, 0, :, -2:], dtype=np.float64) / scale
    past = coordinates[:, :obs]
    short_speed = np.linalg.norm(past[:, -1] - past[:, -2], axis=-1)
    long_speed = np.linalg.norm((past[:, -1] - past[:, -4]) / 3.0, axis=-1)
    raw_speed = np.maximum(short_speed, long_speed)
    positive = raw_speed[raw_speed > 1.0e-6]
    speed_floor = max(float(np.quantile(positive, 0.05)), 0.02) if len(positive) else 0.02
    controls = controls_from_coordinates(coordinates, obs, speed_floor)
    finite = np.isfinite(controls).all(axis=1)
    if finite.mean() < 0.999:
        raise RuntimeError(f"Non-finite control rate is too high: {1.0 - finite.mean():.6f}")
    controls = controls[finite]
    std = controls.std(axis=0)
    if np.any(std < 1.0e-3):
        raise RuntimeError(f"Degenerate control scale: {std}")
    payload = {
        "protocol": "pif1_train_only_scene_tail_split_physical_controls_v1",
        "train_sample_count": int(len(train_ids)),
        "finite_sample_count": int(len(controls)),
        "validation_samples_used": 0,
        "test_samples_used": 0,
        "anchor_indices_zero_based": [3, 7, 11],
        "control_order": ["log_len_ratio_1", "heading_1", "log_len_ratio_2", "turn_2", "log_len_ratio_3", "turn_3"],
        "control_mean": controls.mean(axis=0).tolist(),
        "control_std": std.tolist(),
        "control_q01": np.quantile(controls, 0.01, axis=0).tolist(),
        "control_q99": np.quantile(controls, 0.99, axis=0).tolist(),
        "speed_floor": speed_floor,
        "data_path": str(data),
        "data_sha256": sha256(data),
        "split_path": str(split),
        "split_sha256": sha256(split),
        "config_path": str(CONFIG),
        "config_sha256": sha256(CONFIG),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

