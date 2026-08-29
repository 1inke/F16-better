"""Offline paired interaction-gain diagnostic for B0 versus RB1.

This script never trains a model. It loads the existing seed-1 checkpoints,
reuses the exact saved validation indices, exports per-sample errors, and
analyses whether RB1 improves as observed response/interaction grows.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
SOURCE_ROOT = (
    WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" /
    "EXP_146B_ANCHORED_MODE_TEMPLATE" / "results"
)
RESULTS = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "interaction_gain_diagnostic"
SPLITS = ("general", "challenge")
MODEL_NAMES = ("baseline", "rb1_response_aware")
METRICS = ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde")


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_dir(split: str, model_name: str, seed: int) -> Path:
    return (
        RESULTS / split / model_name / "f16_temporal_social_cross" /
        f"seed_{seed}" / "val_scene_-1"
    )


class IndexedTrajectoryDataset(Dataset):
    def __init__(
        self, data_path: Path, num_path: Path, indices: np.ndarray,
        past_frames: int, future_frames: int, scale: float, max_agents: int,
    ) -> None:
        self.data = np.load(data_path, mmap_mode="r")
        self.num = np.load(num_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.past_frames = int(past_frames)
        self.future_frames = int(future_frames)
        self.scale = float(scale)
        self.max_agents = int(max_agents)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        coordinates = np.array(
            self.data[index, : self.max_agents, :, -2:],
            dtype=np.float32, copy=True,
        ) / self.scale
        past = torch.from_numpy(coordinates[:, : self.past_frames])
        future = torch.from_numpy(
            coordinates[:, self.past_frames:self.past_frames + self.future_frames]
        )
        num_valid = min(int(self.num[index]), self.max_agents)
        scene_id = int(self.data[index, 0, 0, 0])
        return past, future, num_valid, index, scene_id


def prepare_environment(device: torch.device):
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA

    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.MODEL.DECODER.NUM_MODES = 20
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    return cfg, FrontEncoderASTRA


def build_model(split: str, model_name: str, seed: int, cfg, model_class, device):
    source_checkpoint = (
        SOURCE_ROOT / split / "f16_temporal_social_cross" / f"seed_{seed}" /
        "val_scene_-1" / "best_selected.pth"
    )
    checkpoint = run_dir(split, model_name, seed) / "best_selected.pth"
    if not source_checkpoint.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {source_checkpoint} / {checkpoint}")

    os.environ["RAIB_SOURCE_CHECKPOINT"] = str(source_checkpoint)
    model = model_class(cfg, "f16_temporal_social_cross")
    if model_name == "rb1_response_aware":
        encoder = load_module(
            f"raib_encoder_{split}", ROOT / "response_aware_models.py"
        )
        encoder.replace_encoder(model, cfg, "rb1_response_aware_bias")
    decoder = load_module(
        f"raib_decoder_{split}_{model_name}", ROOT / "current_mode_anchor_adapter.py"
    )
    decoder.replace_decoder(model, cfg, "current_mode_anchor_template")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def infer(model, loader, device) -> dict[str, np.ndarray]:
    from train_front_variant import normalize_to_focal_heading

    collected: dict[str, list[np.ndarray]] = {
        key: [] for key in ("index", "scene_id", *METRICS)
    }
    for past, future, num_valid, indices, scene_ids in loader:
        past = past.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        num_valid = num_valid.to(device, non_blocking=True).int()
        past, future = normalize_to_focal_heading(past, future)
        center = past.mean(dim=-2, keepdim=True)
        prediction = model(
            past - center, None, None, mode="testing",
            num_valid=num_valid, graph_loc=past,
        )[2] + center.unsqueeze(-3)
        candidates = prediction[:, 0]
        target = future[:, 0]
        errors = torch.linalg.vector_norm(candidates - target[:, None], dim=-1)
        ade = errors.mean(dim=-1)
        fde = errors[..., -1]
        row = torch.arange(len(ade), device=device)
        best_ade = ade.argmin(dim=-1)
        top1 = model.last_mode_logits[:, 0].argmax(dim=-1)
        values = {
            "index": indices.numpy(),
            "scene_id": scene_ids.numpy(),
            "min_ade_20": ade.min(dim=-1).values.cpu().numpy(),
            "min_fde_20": fde.min(dim=-1).values.cpu().numpy(),
            "same_path_fde": fde[row, best_ade].cpu().numpy(),
            "top1_ade": ade[row, top1].cpu().numpy(),
            "top1_fde": fde[row, top1].cpu().numpy(),
        }
        for key, value in values.items():
            collected[key].append(value)
    return {key: np.concatenate(parts) for key, parts in collected.items()}


def geometry_features(
    data_path: Path, num_path: Path, indices: np.ndarray,
    past_frames: int, scale: float, max_agents: int, frequency: float,
) -> dict[str, np.ndarray]:
    raw = np.load(data_path, mmap_mode="r")
    raw_num = np.load(num_path, mmap_mode="r")
    positions = np.asarray(
        raw[indices, :max_agents, :past_frames, -2:], dtype=np.float32
    ) / float(scale)
    valid_counts = np.minimum(np.asarray(raw_num[indices], dtype=np.int64), max_agents)
    samples, agents, steps, _ = positions.shape
    velocity = np.zeros_like(positions)
    velocity[:, :, 1:] = np.diff(positions, axis=2) * float(frequency)
    velocity[:, :, 0] = velocity[:, :, 1]
    velocity_change = np.zeros_like(velocity)
    velocity_change[:, :, 1:] = np.diff(velocity, axis=2)

    focal_position = positions[:, 0]
    focal_velocity = velocity[:, 0]
    focal_change = velocity_change[:, 0]
    response_magnitude = np.linalg.norm(focal_change, axis=-1).mean(axis=-1)

    if agents <= 1:
        zeros = np.zeros(samples, dtype=np.float32)
        infs = np.full(samples, np.inf, dtype=np.float32)
        return {
            "response_magnitude": response_magnitude,
            "pair_response": zeros, "lateral_response": zeros,
            "gated_pair_response": zeros, "potential_score": zeros,
            "nearest_distance": infs, "max_closing_speed": zeros,
            "minimum_ttc": infs, "neighbor_count": zeros,
        }

    neighbor_position = positions[:, 1:]
    neighbor_velocity = velocity[:, 1:]
    relative = neighbor_position - focal_position[:, None]
    relative_velocity = neighbor_velocity - focal_velocity[:, None]
    distance = np.linalg.norm(relative, axis=-1)
    direction = relative / np.maximum(distance[..., None], 1.0e-4)
    perpendicular = np.stack((-direction[..., 1], direction[..., 0]), axis=-1)
    target_change = focal_change[:, None]
    longitudinal = -(target_change * direction).sum(axis=-1)
    lateral = (target_change * perpendicular).sum(axis=-1)
    dot = (relative * relative_velocity).sum(axis=-1)
    relative_speed_sq = (relative_velocity ** 2).sum(axis=-1)
    closing = -dot / np.maximum(distance, 1.0e-3)
    raw_ttc = -dot / np.maximum(relative_speed_sq, 1.0e-4)
    opportunity = np.exp(-distance / 3.0) / (1.0 + np.exp(-np.clip(closing / 0.5, -30, 30)))

    neighbor_ids = np.arange(1, agents)[None, :]
    valid_neighbor = neighbor_ids < valid_counts[:, None]
    valid_pair_time = valid_neighbor[:, :, None]
    masked_distance = np.where(valid_pair_time, distance, np.inf)
    masked_closing = np.where(valid_pair_time, closing, -np.inf)
    valid_ttc = valid_pair_time & (raw_ttc > 0.0) & np.isfinite(raw_ttc)
    masked_ttc = np.where(valid_ttc, raw_ttc, np.inf)
    pair_strength = np.abs(longitudinal) + np.abs(lateral)
    per_pair_response = np.where(valid_pair_time, pair_strength, 0.0).mean(axis=-1)
    per_pair_lateral = np.where(valid_pair_time, np.abs(lateral), 0.0).mean(axis=-1)
    per_pair_gated = np.where(
        valid_pair_time, opportunity * pair_strength, 0.0
    ).mean(axis=-1)

    has_neighbor = valid_counts > 1
    pair_response = np.where(has_neighbor, per_pair_response.max(axis=1), 0.0)
    lateral_response = np.where(has_neighbor, per_pair_lateral.max(axis=1), 0.0)
    gated_pair_response = np.where(has_neighbor, per_pair_gated.max(axis=1), 0.0)
    potential_score = np.where(
        has_neighbor,
        np.where(valid_pair_time, opportunity, 0.0).max(axis=(1, 2)),
        0.0,
    )
    nearest_distance = masked_distance.min(axis=(1, 2))
    max_closing = masked_closing.max(axis=(1, 2))
    max_closing = np.where(np.isfinite(max_closing), np.maximum(max_closing, 0.0), 0.0)
    minimum_ttc = masked_ttc.min(axis=(1, 2))
    return {
        "response_magnitude": response_magnitude.astype(np.float32),
        "pair_response": pair_response.astype(np.float32),
        "lateral_response": lateral_response.astype(np.float32),
        "gated_pair_response": gated_pair_response.astype(np.float32),
        "potential_score": potential_score.astype(np.float32),
        "nearest_distance": nearest_distance.astype(np.float32),
        "max_closing_speed": max_closing.astype(np.float32),
        "minimum_ttc": minimum_ttc.astype(np.float32),
        "neighbor_count": np.maximum(valid_counts - 1, 0).astype(np.float32),
    }


def relative_change(baseline: float, rb1: float) -> float:
    return 100.0 * (rb1 / baseline - 1.0) if abs(baseline) > 1.0e-12 else float("nan")


def summarize_mask(mask: np.ndarray, paired: dict[str, np.ndarray], label: str) -> dict:
    count = int(mask.sum())
    row = {"group": label, "samples": count}
    if count == 0:
        for metric in METRICS:
            row[f"b0_{metric}"] = None
            row[f"rb1_{metric}"] = None
            row[f"change_pct_{metric}"] = None
            row[f"win_rate_{metric}"] = None
        return row
    for metric in METRICS:
        b0 = float(paired[f"b0_{metric}"][mask].mean())
        rb1 = float(paired[f"rb1_{metric}"][mask].mean())
        delta = paired[f"rb1_{metric}"][mask] - paired[f"b0_{metric}"][mask]
        row[f"b0_{metric}"] = b0
        row[f"rb1_{metric}"] = rb1
        row[f"change_pct_{metric}"] = relative_change(b0, rb1)
        row[f"mean_delta_{metric}"] = float(delta.mean())
        row[f"win_rate_{metric}"] = float((delta < 0.0).mean())
    return row


def quantile_groups(values: np.ndarray, groups: int = 10) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=np.int64)
    for rank, index in enumerate(order):
        labels[index] = min(groups - 1, rank * groups // max(len(values), 1))
    return labels


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyse_one_split(
    split: str, paired: dict[str, np.ndarray], features: dict[str, np.ndarray],
    train_features: dict[str, np.ndarray], output: Path,
) -> dict:
    split_output = output / split
    split_output.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    for i in range(len(paired["index"])):
        row = {"index": int(paired["index"][i]), "scene_id": int(paired["scene_id"][i])}
        for key, values in features.items():
            row[key] = float(values[i])
        for metric in METRICS:
            row[f"b0_{metric}"] = float(paired[f"b0_{metric}"][i])
            row[f"rb1_{metric}"] = float(paired[f"rb1_{metric}"][i])
            row[f"delta_{metric}"] = float(
                paired[f"rb1_{metric}"][i] - paired[f"b0_{metric}"][i]
            )
        sample_rows.append(row)
    write_rows(split_output / "paired_samples.csv", sample_rows)

    decile_tables = {}
    for feature_name in (
        "response_magnitude", "pair_response", "lateral_response",
        "gated_pair_response", "potential_score",
    ):
        labels = quantile_groups(features[feature_name], 10)
        rows = []
        for decile in range(10):
            members = labels == decile
            row = summarize_mask(members, paired, f"D{decile + 1}")
            row["feature"] = feature_name
            row["range_min"] = (
                float(features[feature_name][members].min()) if members.any() else None
            )
            row["range_max"] = (
                float(features[feature_name][members].max()) if members.any() else None
            )
            rows.append(row)
        decile_tables[feature_name] = rows
        write_rows(split_output / f"deciles_{feature_name}.csv", rows)

    nearest_masks = [
        ("<1.5m", features["nearest_distance"] < 1.5),
        ("1.5-3.0m", (features["nearest_distance"] >= 1.5) & (features["nearest_distance"] < 3.0)),
        (">=3.0m/no-neighbor", features["nearest_distance"] >= 3.0),
    ]
    closing_q = np.quantile(train_features["max_closing_speed"], [1 / 3, 2 / 3])
    closing_masks = [
        ("low", features["max_closing_speed"] <= closing_q[0]),
        ("medium", (features["max_closing_speed"] > closing_q[0]) & (features["max_closing_speed"] <= closing_q[1])),
        ("high", features["max_closing_speed"] > closing_q[1]),
    ]
    ttc_masks = [
        ("low_<2s", features["minimum_ttc"] < 2.0),
        ("medium_2-5s", (features["minimum_ttc"] >= 2.0) & (features["minimum_ttc"] < 5.0)),
        ("high/no-conflict_>=5s", features["minimum_ttc"] >= 5.0),
    ]
    neighbor_masks = [
        ("0-1", features["neighbor_count"] <= 1),
        ("2-3", (features["neighbor_count"] >= 2) & (features["neighbor_count"] <= 3)),
        ("4+", features["neighbor_count"] >= 4),
    ]
    potential_tables = {}
    for name, masks in (
        ("nearest_distance", nearest_masks), ("closing_speed", closing_masks),
        ("ttc", ttc_masks), ("neighbor_count", neighbor_masks),
    ):
        rows = [summarize_mask(mask, paired, label) for label, mask in masks]
        potential_tables[name] = rows
        write_rows(split_output / f"potential_{name}.csv", rows)

    response_threshold = float(np.quantile(train_features["response_magnitude"], 0.75))
    potential_threshold = float(np.quantile(train_features["potential_score"], 0.75))
    high_response = features["response_magnitude"] >= response_threshold
    high_potential = features["potential_score"] >= potential_threshold
    two_dimensional = []
    for p_label, p_mask in (("low_potential", ~high_potential), ("high_potential", high_potential)):
        for r_label, r_mask in (("low_response", ~high_response), ("high_response", high_response)):
            row = summarize_mask(p_mask & r_mask, paired, f"{p_label}+{r_label}")
            row["potential_threshold_train_q75"] = potential_threshold
            row["response_threshold_train_q75"] = response_threshold
            two_dimensional.append(row)
    write_rows(split_output / "potential_x_response.csv", two_dimensional)

    cumulative = []
    order = np.argsort(features["response_magnitude"], kind="stable")[::-1]
    for percent in range(10, 101, 10):
        count = max(1, int(math.ceil(len(order) * percent / 100.0)))
        members = np.zeros(len(order), dtype=bool)
        members[order[:count]] = True
        row = summarize_mask(members, paired, f"top_{percent}%")
        row["response_cutoff"] = float(features["response_magnitude"][order[count - 1]])
        cumulative.append(row)
    write_rows(split_output / "cumulative_top_response.csv", cumulative)

    overall = summarize_mask(np.ones(len(paired["index"]), dtype=bool), paired, "all")
    result = {
        "split": split,
        "overall": overall,
        "train_thresholds": {
            "response_q75": response_threshold,
            "potential_q75": potential_threshold,
            "closing_q33_q67": closing_q.tolist(),
        },
        "deciles": decile_tables,
        "potential_tables": potential_tables,
        "potential_x_response": two_dimensional,
        "cumulative_top_response": cumulative,
    }
    (split_output / "summary.json").write_text(
        json.dumps(json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def plot_results(results: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(results), 2, figsize=(11, 4.2 * len(results)), squeeze=False)
    for row, result in enumerate(results):
        deciles = result["deciles"]["response_magnitude"]
        x = np.arange(1, 11)
        for metric, label in (("min_ade_20", "minADE20"), ("min_fde_20", "minFDE20")):
            axes[row, 0].plot(
                x, [item[f"change_pct_{metric}"] for item in deciles], marker="o", label=label
            )
        axes[row, 0].axhline(0.0, color="black", linewidth=0.8)
        axes[row, 0].set_title(f"{result['split']}: response deciles")
        axes[row, 0].set_xlabel("Response decile (low to high)")
        axes[row, 0].set_ylabel("RB1 error change vs B0 (%)")
        axes[row, 0].legend()
        cumulative = result["cumulative_top_response"]
        xp = np.arange(10, 101, 10)
        for metric, label in (("min_ade_20", "minADE20"), ("min_fde_20", "minFDE20")):
            axes[row, 1].plot(
                xp, [item[f"change_pct_{metric}"] for item in cumulative], marker="o", label=label
            )
        axes[row, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[row, 1].set_title(f"{result['split']}: cumulative highest response")
        axes[row, 1].set_xlabel("Included top-response samples (%)")
        axes[row, 1].set_ylabel("RB1 error change vs B0 (%)")
        axes[row, 1].legend()
    figure.tight_layout()
    figure.savefig(output / "interaction_gain_curves.png", dpi=180)
    plt.close(figure)


def verify_aggregate(split: str, model_name: str, values: dict[str, np.ndarray], seed: int) -> dict:
    metrics_path = run_dir(split, model_name, seed) / "metrics.json"
    expected = json.loads(metrics_path.read_text(encoding="utf-8"))["evaluation_best_selected"]
    reference = {
        "min_ade_20": expected["oracle"]["min_ade_20"],
        "min_fde_20": expected["oracle"]["min_fde_20"],
        "same_path_fde": expected["oracle"]["same_path_fde"],
        "top1_ade": expected["original"]["ade"],
        "top1_fde": expected["original"]["fde"],
    }
    checks = {}
    for metric in METRICS:
        observed = float(values[metric].mean())
        difference = observed - float(reference[metric])
        checks[metric] = {
            "expected": float(reference[metric]), "observed": observed,
            "absolute_difference": difference,
            "passed": abs(difference) <= 2.0e-5,
        }
    if not all(item["passed"] for item in checks.values()):
        raise RuntimeError(f"Aggregate reproduction failed for {split}/{model_name}: {checks}")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--splits", nargs="+", choices=("all", *SPLITS), default=["all"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Development smoke check only; disables aggregate reproduction check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    splits = list(SPLITS) if "all" in args.splits else args.splits
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg, model_class = prepare_environment(device)
    all_results, reproduction = [], {}

    for split in splits:
        baseline_run = run_dir(split, "baseline", args.seed)
        rb1_run = run_dir(split, "rb1_response_aware", args.seed)
        base_indices = np.load(baseline_run / "split_indices.npz")
        rb1_indices = np.load(rb1_run / "split_indices.npz")
        train_indices = np.asarray(base_indices["train"], dtype=np.int64)
        val_indices = np.asarray(base_indices["validation"], dtype=np.int64)
        if not np.array_equal(train_indices, rb1_indices["train"]) or not np.array_equal(
            val_indices, rb1_indices["validation"]
        ):
            raise RuntimeError(f"B0/RB1 split mismatch for {split}")
        if args.max_samples is not None:
            val_indices = val_indices[:args.max_samples]

        split_info = json.loads((baseline_run / "split.json").read_text(encoding="utf-8"))
        data_path = Path(split_info["data_path"])
        num_path = Path(split_info["num_path"])
        dataset = IndexedTrajectoryDataset(
            data_path, num_path, val_indices,
            past_frames=int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME),
            future_frames=int(math.ceil(cfg.DATA.FREQUENCY * cfg.PREDICTION.PRED_TIME)),
            scale=float(cfg.PREDICTION.TRAJ_SCALE), max_agents=int(cfg.DATA.MAX_AGENTS),
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
            pin_memory=device.type == "cuda",
        )
        predictions = {}
        reproduction[split] = {}
        for model_name in MODEL_NAMES:
            print(f"[infer] split={split} model={model_name} samples={len(dataset)}")
            model = build_model(split, model_name, args.seed, cfg, model_class, device)
            predictions[model_name] = infer(model, loader, device)
            if args.max_samples is None:
                reproduction[split][model_name] = verify_aggregate(
                    split, model_name, predictions[model_name], args.seed
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not np.array_equal(predictions["baseline"]["index"], predictions["rb1_response_aware"]["index"]):
            raise RuntimeError(f"Prediction row mismatch for {split}")
        paired = {
            "index": predictions["baseline"]["index"],
            "scene_id": predictions["baseline"]["scene_id"],
        }
        for metric in METRICS:
            paired[f"b0_{metric}"] = predictions["baseline"][metric]
            paired[f"rb1_{metric}"] = predictions["rb1_response_aware"][metric]

        geometry_args = dict(
            data_path=data_path, num_path=num_path,
            past_frames=int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME),
            scale=float(cfg.PREDICTION.TRAJ_SCALE), max_agents=int(cfg.DATA.MAX_AGENTS),
            frequency=float(cfg.DATA.FREQUENCY),
        )
        features = geometry_features(indices=val_indices, **geometry_args)
        train_features = geometry_features(indices=train_indices, **geometry_args)
        result = analyse_one_split(split, paired, features, train_features, output)
        all_results.append(result)

    plot_results(all_results, output)
    combined = {
        "seed": args.seed, "splits": splits, "training_performed": False,
        "strict_pairing": True, "aggregate_reproduction": reproduction,
        "results": all_results,
    }
    (output / "combined_summary.json").write_text(
        json.dumps(json_ready(combined), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {output}")
    print(f"[summary] {output / 'combined_summary.json'}")
    print(f"[curves] {output / 'interaction_gain_curves.png'}")


if __name__ == "__main__":
    main()
