"""Paired candidate/anchor geometry diagnostic for trained B0 versus IC1."""

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
OUTPUT = ROOT / "geometry_diagnostic"
SPLITS = ("general", "challenge")
MODELS = ("baseline", "ic1_social_mode_anchor_bridge")
METRICS = ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde")
ANCHOR_STEPS = (4, 8, 12)


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


class IndexedDataset(Dataset):
    def __init__(self, data_path, num_path, indices, cfg) -> None:
        self.data = np.load(data_path, mmap_mode="r")
        self.num = np.load(num_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.past_frames = int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME)
        self.future_frames = int(math.ceil(cfg.DATA.FREQUENCY * cfg.PREDICTION.PRED_TIME))
        self.scale = float(cfg.PREDICTION.TRAJ_SCALE)
        self.max_agents = int(cfg.DATA.MAX_AGENTS)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        coordinates = np.array(
            self.data[index, :self.max_agents, :, -2:], dtype=np.float32, copy=True
        ) / self.scale
        return (
            torch.from_numpy(coordinates[:, :self.past_frames]),
            torch.from_numpy(coordinates[:, self.past_frames:self.past_frames + self.future_frames]),
            min(int(self.num[index]), self.max_agents), index,
            int(self.data[index, 0, 0, 0]),
        )


def prepare(device):
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


def build_model(split, model_name, seed, cfg, model_class, device):
    source = (
        SOURCE_ROOT / split / "f16_temporal_social_cross" / f"seed_{seed}" /
        "val_scene_-1" / "best_selected.pth"
    )
    checkpoint = run_dir(split, model_name, seed) / "best_selected.pth"
    if not source.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {source} / {checkpoint}")
    os.environ["ICMAB_SOURCE_CHECKPOINT"] = str(source)
    model = model_class(cfg, "f16_temporal_social_cross")
    if model_name == "baseline":
        module = load_module(f"icmab_diag_b0_{split}", ROOT / "current_baseline_adapter.py")
        module.replace_decoder(model, cfg, "current_mode_anchor_template")
    else:
        module = load_module(f"icmab_diag_ic1_{split}", ROOT / "icmab_models.py")
        module.replace_decoder(model, cfg, "ic1_social_mode_anchor_bridge")
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    return model.to(device).eval()


@torch.inference_mode()
def infer(model, loader, device, is_ic1):
    from train_front_variant import normalize_to_focal_heading

    keys = (
        "index", "scene_id", "prediction", "anchors", "target", "min_ade_20",
        "min_fde_20", "same_path_fde", "top1_ade", "top1_fde", "winner",
    )
    parts = {key: [] for key in keys}
    bridge_parts = {key: [] for key in ("social_norm", "bridge_norm", "gate_std", "mode_delta_std")}
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
        anchors = model.decoder.last_corrected_anchors[:, 0]
        errors = torch.linalg.vector_norm(candidates - target[:, None], dim=-1)
        ade = errors.mean(dim=-1)
        fde = errors[..., -1]
        rows = torch.arange(len(ade), device=device)
        winner = ade.argmin(dim=-1)
        top1 = model.last_mode_logits[:, 0].argmax(dim=-1)
        values = {
            "index": indices.numpy(), "scene_id": scene_ids.numpy(),
            "prediction": candidates.cpu().numpy(), "anchors": anchors.cpu().numpy(),
            "target": target.cpu().numpy(),
            "min_ade_20": ade.min(dim=-1).values.cpu().numpy(),
            "min_fde_20": fde.min(dim=-1).values.cpu().numpy(),
            "same_path_fde": fde[rows, winner].cpu().numpy(),
            "top1_ade": ade[rows, top1].cpu().numpy(),
            "top1_fde": fde[rows, top1].cpu().numpy(),
            "winner": winner.cpu().numpy(),
        }
        for key, value in values.items():
            parts[key].append(value)
        if is_ic1:
            token = model.decoder.last_interaction_token[:, 0]
            bridge = model.decoder.last_bridge_output[:, 0]
            gates = model.decoder.last_mode_gates[:, 0]
            mode_delta = model.decoder.last_conditioned_mode_delta[:, 0].norm(dim=-1)
            diagnostics = {
                "social_norm": token.norm(dim=-1).cpu().numpy(),
                "bridge_norm": bridge.norm(dim=-1).cpu().numpy(),
                "gate_std": gates.std(dim=-1).cpu().numpy(),
                "mode_delta_std": mode_delta.std(dim=-1).cpu().numpy(),
            }
            for key, value in diagnostics.items():
                bridge_parts[key].append(value)
    result = {key: np.concatenate(value) for key, value in parts.items()}
    if is_ic1:
        result.update({key: np.concatenate(value) for key, value in bridge_parts.items()})
    return result


def describe(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return {
        "mean": float(finite.mean()), "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)), "p95": float(np.quantile(finite, 0.95)),
    }


def relative_change(b0, ic1):
    return 100.0 * (float(ic1) / float(b0) - 1.0)


def metric_summary(values, mask=None):
    if mask is None:
        mask = np.ones(len(values["index"]), dtype=bool)
    result = {"samples": int(mask.sum())}
    for metric in METRICS:
        result[metric] = float(values[metric][mask].mean())
    return result


def pairwise_diversity(prediction, anchors):
    first, second = np.triu_indices(prediction.shape[1], k=1)
    endpoint = np.linalg.norm(
        prediction[:, first, -1] - prediction[:, second, -1], axis=-1
    ).mean(axis=-1)
    trajectory = np.linalg.norm(
        prediction[:, first] - prediction[:, second], axis=-1
    ).mean(axis=(-1, -2))
    anchor = np.linalg.norm(
        anchors[:, first] - anchors[:, second], axis=-1
    ).mean(axis=(-1, -2))
    return {"endpoint": endpoint, "anchor": anchor, "trajectory": trajectory}


def potential_score(data_path, num_path, indices, cfg):
    raw = np.load(data_path, mmap_mode="r")
    raw_num = np.load(num_path, mmap_mode="r")
    past_frames = int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME)
    positions = np.asarray(
        raw[indices, :int(cfg.DATA.MAX_AGENTS), :past_frames, -2:], dtype=np.float32
    ) / float(cfg.PREDICTION.TRAJ_SCALE)
    counts = np.minimum(np.asarray(raw_num[indices], dtype=np.int64), int(cfg.DATA.MAX_AGENTS))
    velocity = np.zeros_like(positions)
    velocity[:, :, 1:] = np.diff(positions, axis=2) * float(cfg.DATA.FREQUENCY)
    velocity[:, :, 0] = velocity[:, :, 1]
    relative = positions[:, 1:] - positions[:, :1]
    relative_velocity = velocity[:, 1:] - velocity[:, :1]
    distance = np.linalg.norm(relative, axis=-1)
    closing = -(relative * relative_velocity).sum(axis=-1) / np.maximum(distance, 1.0e-3)
    opportunity = np.exp(-distance / 3.0) / (
        1.0 + np.exp(-np.clip(closing / 0.5, -30.0, 30.0))
    )
    valid = np.arange(1, positions.shape[1])[None, :, None] < counts[:, None, None]
    return np.where(valid, opportunity, 0.0).max(axis=(1, 2)).astype(np.float32)


def write_rows(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify(split, model_name, values, seed):
    raw = json.loads(
        (run_dir(split, model_name, seed) / "metrics.json").read_text(encoding="utf-8")
    )["evaluation_best_selected"]
    expected = {
        "min_ade_20": raw["oracle"]["min_ade_20"],
        "min_fde_20": raw["oracle"]["min_fde_20"],
        "same_path_fde": raw["oracle"]["same_path_fde"],
        "top1_ade": raw["original"]["ade"], "top1_fde": raw["original"]["fde"],
    }
    checks = {}
    for metric in METRICS:
        observed = float(values[metric].mean())
        checks[metric] = {
            "expected": float(expected[metric]), "observed": observed,
            "absolute_difference": observed - float(expected[metric]),
        }
        if abs(checks[metric]["absolute_difference"]) > 2.0e-5:
            raise RuntimeError(f"Metric reproduction failed: {split}/{model_name}/{metric}")
    return checks


def analyse(split, b0, ic1, potential, train_potential, split_output):
    anchor_shift = np.linalg.norm(ic1["anchors"] - b0["anchors"], axis=-1)
    per_mode_shift = anchor_shift.mean(axis=-1)
    mode_differential = per_mode_shift.std(axis=-1)
    diversity_b0 = pairwise_diversity(b0["prediction"], b0["anchors"])
    diversity_ic1 = pairwise_diversity(ic1["prediction"], ic1["anchors"])
    target_anchors = b0["target"][:, np.asarray(ANCHOR_STEPS) - 1]
    coverage = {}
    for name, values in (("baseline", b0), ("ic1", ic1)):
        anchor_error = np.linalg.norm(
            values["anchors"] - target_anchors[:, None], axis=-1
        )
        coverage[name] = {
            f"step_{step}": anchor_error[:, :, anchor].min(axis=1)
            for anchor, step in enumerate(ANCHOR_STEPS)
        }
        coverage[name]["endpoint"] = np.linalg.norm(
            values["prediction"][:, :, -1] - values["target"][:, None, -1], axis=-1
        ).min(axis=1)
    high_threshold = float(np.quantile(train_potential, 0.75))
    high_mask = potential >= high_threshold
    b0_high = metric_summary(b0, high_mask)
    ic1_high = metric_summary(ic1, high_mask)
    high_changes = {
        metric: relative_change(b0_high[metric], ic1_high[metric]) for metric in METRICS
    }
    summary = {
        "split": split, "samples": int(len(b0["index"])),
        "anchor_shift": {
            f"step_{step}": describe(anchor_shift[:, :, anchor])
            for anchor, step in enumerate(ANCHOR_STEPS)
        },
        "mode_differential_shift": describe(mode_differential),
        "oracle_winner_change_rate": float((b0["winner"] != ic1["winner"]).mean()),
        "candidate_diversity": {}, "coverage": {},
        "bridge_activation": {
            key: describe(ic1[key]) for key in (
                "social_norm", "bridge_norm", "gate_std", "mode_delta_std"
            )
        },
        "high_interaction": {
            "threshold_train_q75": high_threshold,
            "samples": int(high_mask.sum()), "baseline": b0_high, "ic1": ic1_high,
            "change_pct": high_changes,
        },
    }
    for key in diversity_b0:
        b0_mean = float(diversity_b0[key].mean())
        ic1_mean = float(diversity_ic1[key].mean())
        summary["candidate_diversity"][key] = {
            "baseline": describe(diversity_b0[key]), "ic1": describe(diversity_ic1[key]),
            "mean_change_pct": relative_change(b0_mean, ic1_mean),
        }
    for key in coverage["baseline"]:
        b0_mean = float(coverage["baseline"][key].mean())
        ic1_mean = float(coverage["ic1"][key].mean())
        summary["coverage"][key] = {
            "baseline": describe(coverage["baseline"][key]),
            "ic1": describe(coverage["ic1"][key]),
            "mean_change_pct": relative_change(b0_mean, ic1_mean),
        }

    rows = []
    for index in range(len(b0["index"])):
        row = {
            "index": int(b0["index"][index]), "scene_id": int(b0["scene_id"][index]),
            "potential_score": float(potential[index]), "high_interaction": bool(high_mask[index]),
            "winner_changed": bool(b0["winner"][index] != ic1["winner"][index]),
            "mode_differential_shift": float(mode_differential[index]),
            "social_norm": float(ic1["social_norm"][index]),
            "bridge_norm": float(ic1["bridge_norm"][index]),
            "gate_std": float(ic1["gate_std"][index]),
            "mode_delta_std": float(ic1["mode_delta_std"][index]),
        }
        for anchor, step in enumerate(ANCHOR_STEPS):
            row[f"anchor_shift_step_{step}"] = float(anchor_shift[index, :, anchor].mean())
            row[f"b0_coverage_step_{step}"] = float(coverage["baseline"][f"step_{step}"][index])
            row[f"ic1_coverage_step_{step}"] = float(coverage["ic1"][f"step_{step}"][index])
        for key in diversity_b0:
            row[f"b0_diversity_{key}"] = float(diversity_b0[key][index])
            row[f"ic1_diversity_{key}"] = float(diversity_ic1[key][index])
        for metric in METRICS:
            row[f"b0_{metric}"] = float(b0[metric][index])
            row[f"ic1_{metric}"] = float(ic1[metric][index])
            row[f"delta_{metric}"] = float(ic1[metric][index] - b0[metric][index])
        rows.append(row)
    write_rows(split_output / "paired_geometry_samples.csv", rows)
    (split_output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def plot_summaries(summaries, output):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(summaries))
    labels = [item["split"] for item in summaries]
    width = 0.25
    for anchor, step in enumerate(ANCHOR_STEPS):
        axes[0].bar(
            x + (anchor - 1) * width,
            [item["anchor_shift"][f"step_{step}"]["mean"] for item in summaries],
            width, label=f"step {step}",
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean anchor shift")
    axes[0].set_title("IC1 versus B0 anchor geometry")
    axes[0].legend()
    coverage_keys = ("step_4", "step_8", "step_12", "endpoint")
    for summary in summaries:
        axes[1].plot(
            np.arange(len(coverage_keys)),
            [summary["coverage"][key]["mean_change_pct"] for key in coverage_keys],
            marker="o", label=summary["split"],
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(coverage_keys)), coverage_keys)
    axes[1].set_ylabel("IC1 coverage error change (%)")
    axes[1].set_title("GT anchor/endpoint coverage")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "geometry_and_coverage.png", dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--splits", nargs="+", choices=("all", *SPLITS), default=["all"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    splits = list(SPLITS) if "all" in args.splits else args.splits
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg, model_class = prepare(device)
    summaries, reproduction = [], {}
    for split in splits:
        b0_run = run_dir(split, "baseline", args.seed)
        ic1_run = run_dir(split, "ic1_social_mode_anchor_bridge", args.seed)
        b0_indices = np.load(b0_run / "split_indices.npz")
        ic1_indices = np.load(ic1_run / "split_indices.npz")
        train_indices = np.asarray(b0_indices["train"], dtype=np.int64)
        val_indices = np.asarray(b0_indices["validation"], dtype=np.int64)
        if not np.array_equal(train_indices, ic1_indices["train"]) or not np.array_equal(
            val_indices, ic1_indices["validation"]
        ):
            raise RuntimeError(f"B0/IC1 split mismatch for {split}")
        if args.max_samples is not None:
            val_indices = val_indices[:args.max_samples]
        split_info = json.loads((b0_run / "split.json").read_text(encoding="utf-8"))
        data_path, num_path = Path(split_info["data_path"]), Path(split_info["num_path"])
        dataset = IndexedDataset(data_path, num_path, val_indices, cfg)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
            pin_memory=device.type == "cuda",
        )
        values = {}
        reproduction[split] = {}
        for model_name in MODELS:
            print(f"[infer] {split}/{model_name} samples={len(dataset)}", flush=True)
            model = build_model(split, model_name, args.seed, cfg, model_class, device)
            values[model_name] = infer(
                model, loader, device, model_name == "ic1_social_mode_anchor_bridge"
            )
            if args.max_samples is None:
                reproduction[split][model_name] = verify(
                    split, model_name, values[model_name], args.seed
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not np.array_equal(values[MODELS[0]]["index"], values[MODELS[1]]["index"]):
            raise RuntimeError(f"paired index mismatch for {split}")
        potential = potential_score(data_path, num_path, val_indices, cfg)
        train_potential = potential_score(data_path, num_path, train_indices, cfg)
        split_output = output / split
        split_output.mkdir(parents=True, exist_ok=True)
        summaries.append(analyse(
            split, values[MODELS[0]], values[MODELS[1]],
            potential, train_potential, split_output,
        ))
    combined = {
        "training_performed": False, "strict_pairing": True,
        "aggregate_reproduction": reproduction, "results": summaries,
    }
    (output / "combined_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_summaries(summaries, output)
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
