"""Evaluate CV, zero-shot, and trained EXP-195A models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from exp195_common import (
    CONFIGS, DEFAULT_ETH_CHECKPOINT, FUTURE_HZ, ROOT, BridgeArrayDataset,
    build_model, model_forward, normalize_to_focal_heading,
)
from materialize_configs import materialize


RESULT_DIR = ROOT / "EXP_195A_SHAPE_PRESERVING_BRIDGE"


def checkpoint_for(name: str) -> Path:
    return (
        ROOT / "results" / name / "f16_temporal_social_cross"
        / "seed_1" / "val_scene_-1" / "best_selected.pth"
    )


def prefix_metric(prediction: np.ndarray, target: np.ndarray, horizon: float):
    times = np.arange(1, prediction.shape[-2] + 1, dtype=np.float32) / FUTURE_HZ
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


def summarize_predictions(prediction: np.ndarray, target: np.ndarray, logits: np.ndarray):
    errors = np.linalg.norm(prediction - target[:, None], axis=-1)
    ade = errors.mean(axis=-1)
    fde = errors[..., -1]
    rows = np.arange(len(ade))
    selected = logits.argmax(axis=-1)
    oracle = ade.argmin(axis=-1)
    top6 = np.argsort(-logits, axis=-1)[:, :6]
    top6_ade = np.take_along_axis(ade, top6, axis=1)
    top6_fde = np.take_along_axis(fde, top6, axis=1)
    metrics = {
        "samples": int(len(ade)),
        "minADE20": float(ade.min(axis=1).mean()),
        "minFDE20": float(fde.min(axis=1).mean()),
        "selected_ADE": float(ade[rows, selected].mean()),
        "selected_FDE": float(fde[rows, selected].mean()),
        "mean_regret": float((ade[rows, selected] - ade.min(axis=1)).mean()),
        "hit_best_ade": float((selected == oracle).mean()),
        "minADE6": float(top6_ade.min(axis=1).mean()),
        "minFDE6": float(top6_fde.min(axis=1).mean()),
        "MR6_at_2m": float((top6_fde.min(axis=1) > 2.0).mean()),
        "av2_status": "NON-OFFICIAL AV2 DIAGNOSTIC",
    }
    timestep = []
    for horizon in (1.0, 2.0, 3.0, 4.8):
        per_mode = prefix_metric(prediction, target[:, None], horizon)
        timestep.append({
            "horizon_sec": horizon,
            "minADE20": float(per_mode.min(axis=1).mean()),
            "selected_ADE": float(per_mode[rows, selected].mean()),
            "interpolation": bool(not np.isclose(horizon * FUTURE_HZ, round(horizon * FUTURE_HZ))),
        })
    return metrics, timestep


def evaluate_model(config_path: Path, checkpoint: Path, data_path: Path, num_path: Path,
                   batch_size: int, device: torch.device):
    name = next(key for key, spec in CONFIGS.items() if spec["history_steps"] + 12 == np.load(data_path, mmap_mode="r").shape[2])
    dataset = BridgeArrayDataset(data_path, num_path, CONFIGS[name]["history_steps"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model, _ = build_model(config_path, checkpoint, device)
    model.eval()
    predictions, targets, logits = [], [], []
    with torch.inference_mode():
        for past, future, num_valid in loader:
            past, future = past.to(device), future.to(device)
            past, future = normalize_to_focal_heading(past, future)
            output, score = model_forward(model, past, num_valid.to(device))
            predictions.append(output.cpu().numpy())
            targets.append(future[:, 0].cpu().numpy())
            logits.append(score.cpu().numpy())
    return summarize_predictions(
        np.concatenate(predictions), np.concatenate(targets), np.concatenate(logits)
    )


def evaluate_cv(data_path: Path, num_path: Path):
    dataset = BridgeArrayDataset(data_path, num_path, CONFIGS["a1"]["history_steps"])
    predictions, targets = [], []
    for past, future, _ in DataLoader(dataset, batch_size=512, shuffle=False):
        velocity = (past[:, 0, -1] - past[:, 0, -2]) * CONFIGS["a1"]["history_hz"]
        times = torch.arange(1, 13, dtype=past.dtype)[None, :, None] / FUTURE_HZ
        predictions.append((past[:, 0, -1, None] + velocity[:, None] * times).numpy())
        targets.append(future[:, 0].numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    error = np.linalg.norm(prediction - target, axis=-1)
    return {
        "samples": len(error), "ADE": float(error.mean()),
        "FDE": float(error[:, -1].mean()),
    }


def decision(rows: dict[str, dict]):
    labels = []
    cv, a1 = rows.get("cv"), rows.get("a1")
    if cv and a1:
        if a1["minADE20"] < cv["ADE"] and a1["minFDE20"] < cv["FDE"]:
            labels.append("AV2_BRIDGE_FEASIBLE")
        else:
            labels.append("AV2_BRIDGE_FAILED")
    if all(key in rows for key in ("a1", "a2", "a3")):
        base = rows["a1"]
        gains = [
            max((base["minADE20"] - rows[key]["minADE20"]) / base["minADE20"],
                (base["minFDE20"] - rows[key]["minFDE20"]) / base["minFDE20"])
            for key in ("a2", "a3")
        ]
        worst_selected = max(
            (rows[key]["selected_ADE"] - base["selected_ADE"]) / base["selected_ADE"]
            for key in ("a2", "a3")
        )
        if max(gains) >= 0.02 and worst_selected <= 0.02:
            labels += ["HIGH_RES_HISTORY_USEFUL", "PROCEED_TO_LONG_HISTORY"]
        elif min(gains) < -0.05:
            labels.append("HIGH_RES_HISTORY_HARMFUL")
        else:
            labels.append("HIGH_RES_HISTORY_SATURATED")
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    configs = materialize()
    prepared = ROOT / "prepared"
    required = [prepared / "val_a1_data.npy", prepared / "val_a1_num.npy"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Run prepare_av2_ped_bridge.py first:\n" + "\n".join(missing))
    rows, timestep_rows = {}, []
    rows["cv"] = evaluate_cv(prepared / "val_a1_data.npy", prepared / "val_a1_num.npy")
    candidates = [("zs", "a1", DEFAULT_ETH_CHECKPOINT)] + [
        (name, name, checkpoint_for(name)) for name in CONFIGS
    ]
    for label, config_name, checkpoint in candidates:
        if not checkpoint.is_file():
            print(f"[skip] missing checkpoint for {label}: {checkpoint}")
            continue
        metrics, by_time = evaluate_model(
            configs[config_name], checkpoint,
            prepared / f"val_{config_name}_data.npy",
            prepared / f"val_{config_name}_num.npy",
            args.batch_size, device,
        )
        rows[label] = metrics
        timestep_rows.extend({"config": label, **item} for item in by_time)
        print(f"[evaluated] {label}: minADE20={metrics['minADE20']:.4f}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows.values() for key in row} | {"config"})
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"config": key, **value} for key, value in rows.items())
    with (args.output_dir / "timestep_error.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "horizon_sec", "minADE20", "selected_ADE", "interpolation"])
        writer.writeheader(); writer.writerows(timestep_rows)
    labels = decision(rows)
    summary = {"metrics": rows, "decision_labels": labels}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
