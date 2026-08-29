"""Counterfactual Social Necessity Audit: identical F16 checkpoint, one edge intervention."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
IR1 = WORKSPACE / "INTERACTION_RESIDUAL_SUPERVISION_SERIES"
SOURCE_ROOT = WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" / "EXP_146B_ANCHORED_MODE_TEMPLATE" / "results"
BASELINE_ROOT = IR1 / "results"
OUTPUT = ROOT / "results"
SPLITS = ("general", "challenge")
METRICS = ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde", "oracle_cost")
LAMBDA_FDE = 0.5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_dir(split: str, seed: int) -> Path:
    return BASELINE_ROOT / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def source_checkpoint(split: str, seed: int) -> Path:
    return SOURCE_ROOT / split / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"


def prepare(device: torch.device):
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT), str(ROOT)]
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


def build_model(split, seed, cfg, model_class, device):
    source = source_checkpoint(split, seed)
    checkpoint = run_dir(split, seed) / "best_selected.pth"
    if not source.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Missing source/checkpoint: {source} / {checkpoint}")
    os.environ["IRS_SOURCE_CHECKPOINT"] = str(source)
    model = model_class(cfg, "f16_temporal_social_cross")
    adapter = load_module(f"csna_adapter_{split}", IR1 / "current_baseline_adapter.py")
    adapter.replace_decoder(model, cfg, "current_mode_anchor_template")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def infer(model, loader, device, self_only=False):
    from train_front_variant import normalize_to_focal_heading
    from csna_mask import focal_self_only_intervention

    keys = ("index", "scene_id", "start_frame", "prediction", "target", *METRICS, "ade_winner", "cost_winner", "top1")
    parts = {key: [] for key in keys}
    context = focal_self_only_intervention(model) if self_only else _nullcontext(model)
    with context:
        for past, future, num_valid, indices, scene_ids in loader:
            past = past.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            num_valid = num_valid.to(device, non_blocking=True).int()
            past, future = normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            prediction = model(
                past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past,
            )[2] + center.unsqueeze(-3)
            candidates, target = prediction[:, 0], future[:, 0]
            errors = torch.linalg.vector_norm(candidates - target[:, None], dim=-1)
            ade, fde = errors.mean(dim=-1), errors[..., -1]
            cost = ade + LAMBDA_FDE * fde
            row = torch.arange(len(ade), device=device)
            ade_winner, cost_winner = ade.argmin(-1), cost.argmin(-1)
            top1 = model.last_mode_logits[:, 0].argmax(-1)
            start_frame = past[:, 0, 0, 0].new_zeros(len(past))  # overwritten from dataset metadata below
            values = {
                "index": indices.numpy(), "scene_id": scene_ids.numpy(),
                "start_frame": start_frame.cpu().numpy(),
                "prediction": candidates.cpu().numpy(), "target": target.cpu().numpy(),
                "min_ade_20": ade.min(-1).values.cpu().numpy(),
                "min_fde_20": fde.min(-1).values.cpu().numpy(),
                "same_path_fde": fde[row, ade_winner].cpu().numpy(),
                "top1_ade": ade[row, top1].cpu().numpy(),
                "top1_fde": fde[row, top1].cpu().numpy(),
                "oracle_cost": cost.min(-1).values.cpu().numpy(),
                "ade_winner": ade_winner.cpu().numpy(), "cost_winner": cost_winner.cpu().numpy(),
                "top1": top1.cpu().numpy(),
            }
            for key, value in values.items():
                parts[key].append(value)
    return {key: np.concatenate(value) for key, value in parts.items()}


class _nullcontext:
    def __init__(self, value): self.value = value
    def __enter__(self): return self.value
    def __exit__(self, *args): return False


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)), "p90": float(np.quantile(values, 0.90)),
    }


def block_ids(scene, frame, block_frames):
    return np.asarray([f"{int(s)}:{int(f) // block_frames}" for s, f in zip(scene, frame)])


def cluster_bootstrap(values, clusters, repetitions, rng):
    values = np.asarray(values, dtype=np.float64)
    unique = np.unique(clusters)
    members = [np.flatnonzero(clusters == item) for item in unique]
    estimates = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        chosen = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[item] for item in chosen])
        estimates[iteration] = values[indices].mean()
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def paired_stat(delta, clusters, repetitions, rng):
    delta = np.asarray(delta, dtype=np.float64)
    indicator = (delta > 0.0).astype(np.float64)
    return {
        "mean_paired_difference": float(delta.mean()),
        "median_paired_difference": float(np.median(delta)),
        "mean_95_ci_block_bootstrap": cluster_bootstrap(delta, clusters, repetitions, rng),
        "social_help_win_rate": float(indicator.mean()),
        "win_rate_95_ci_block_bootstrap": cluster_bootstrap(indicator, clusters, repetitions, rng),
    }


def metric_pair(full, self_only, mask):
    result = {"samples": int(mask.sum())}
    for metric in METRICS:
        f, s = float(full[metric][mask].mean()), float(self_only[metric][mask].mean())
        result[metric] = {
            "full": f, "self_only": s,
            "full_advantage_pct": 100.0 * (s / f - 1.0),
        }
    return result


def rank_deciles(potential):
    order = np.argsort(potential, kind="stable")
    bins = np.empty(len(order), dtype=np.int64)
    bins[order] = np.minimum(9, np.arange(len(order)) * 10 // max(1, len(order)))
    return bins


def spearman(x, y):
    from scipy.stats import spearmanr
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyse(split, full, self_only, potential, train_potential, scene, frame, args, output):
    n = len(full["index"])
    all_mask = np.ones(n, dtype=bool)
    high_threshold = float(np.quantile(train_potential, 0.75))
    high = potential >= high_threshold
    clusters = block_ids(scene, frame, args.block_frames)
    rng = np.random.default_rng(args.bootstrap_seed)
    deltas = {metric: self_only[metric] - full[metric] for metric in METRICS}
    decile_id = rank_deciles(potential)
    deciles = []
    for decile in range(10):
        mask = decile_id == decile
        entry = {
            "decile": decile + 1, "samples": int(mask.sum()),
            "potential_min": float(potential[mask].min()), "potential_max": float(potential[mask].max()),
        }
        for metric in ("min_ade_20", "min_fde_20", "oracle_cost"):
            entry[f"delta_{metric}"] = float(deltas[metric][mask].mean())
            entry[f"full_advantage_{metric}_pct"] = 100.0 * (
                self_only[metric][mask].mean() / full[metric][mask].mean() - 1.0
            )
        deciles.append(entry)
    high_stats = metric_pair(full, self_only, high)
    high_cost_stats = paired_stat(deltas["oracle_cost"][high], clusters[high], args.bootstrap, rng)
    rho_ade = spearman(np.arange(1, 11), [x["delta_min_ade_20"] for x in deciles])
    rho_fde = spearman(np.arange(1, 11), [x["delta_min_fde_20"] for x in deciles])
    high_ade = high_stats["min_ade_20"]["full_advantage_pct"]
    high_fde = high_stats["min_fde_20"]["full_advantage_pct"]
    metric_pass = (high_ade >= 2.0 and high_fde >= 0.0) or (high_fde >= 2.0 and high_ade >= 0.0)
    trend_pass = max(rho_ade or -1.0, rho_fde or -1.0) >= 0.5
    win_pass = high_cost_stats["win_rate_95_ci_block_bootstrap"][0] > 0.5
    decision = "enter_csrs" if metric_pass and trend_pass and win_pass else "do_not_enter_csrs"
    summary = {
        "split": split, "samples": n, "lambda_fde": LAMBDA_FDE,
        "intervention": "only focal receive-row in F16 temporal-social allowed mask becomes self-loop",
        "bootstrap": {"method": "scene_x_contiguous_time_block", "block_frames": args.block_frames,
                      "blocks": int(len(np.unique(clusters))), "repetitions": args.bootstrap},
        "overall": metric_pair(full, self_only, all_mask),
        "overall_paired": {m: paired_stat(deltas[m], clusters, args.bootstrap, rng) for m in METRICS},
        "high_interaction": {"threshold_train_q75": high_threshold, "metrics": high_stats,
                             "oracle_cost_paired": high_cost_stats},
        "interaction_deciles": deciles,
        "decile_trend_spearman": {"min_ade_20": rho_ade, "min_fde_20": rho_fde},
        "prediction_effect": {
            "mean_all_mode_trajectory_shift": float(np.linalg.norm(self_only["prediction"] - full["prediction"], axis=-1).mean()),
            "mean_all_mode_endpoint_shift": float(np.linalg.norm(self_only["prediction"][:, :, -1] - full["prediction"][:, :, -1], axis=-1).mean()),
            "ade_winner_change_rate": float((self_only["ade_winner"] != full["ade_winner"]).mean()),
            "top1_change_rate": float((self_only["top1"] != full["top1"]).mean()),
        },
        "gate_decision": {"decision": decision, "high_metric_pass": metric_pass,
                          "high_win_ci_pass": win_pass, "decile_trend_pass": trend_pass},
    }
    rows = []
    for i in range(n):
        row = {"index": int(full["index"][i]), "scene_id": int(scene[i]), "start_frame": int(frame[i]),
               "time_block": str(clusters[i]), "interaction_potential": float(potential[i]),
               "interaction_decile": int(decile_id[i] + 1), "high_interaction": bool(high[i])}
        for metric in METRICS:
            row[f"full_{metric}"] = float(full[metric][i]); row[f"self_{metric}"] = float(self_only[metric][i])
            row[f"social_gain_{metric}"] = float(deltas[metric][i])
        rows.append(row)
    write_csv(output / "paired_samples.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def verify_reproduction(split, seed, full):
    raw = json.loads((run_dir(split, seed) / "metrics.json").read_text(encoding="utf-8"))["evaluation_best_selected"]
    expected = {"min_ade_20": raw["oracle"]["min_ade_20"], "min_fde_20": raw["oracle"]["min_fde_20"],
                "same_path_fde": raw["oracle"]["same_path_fde"], "top1_ade": raw["original"]["ade"],
                "top1_fde": raw["original"]["fde"]}
    checks = {}
    for metric, target in expected.items():
        observed = float(full[metric].mean()); checks[metric] = {"expected": float(target), "observed": observed,
                                                                "absolute_difference": observed - float(target)}
        if abs(observed - float(target)) > 2e-5:
            raise RuntimeError(f"Full-social reproduction failed: {split}/{metric}")
    return checks


def plot(summaries, output):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for item in summaries:
        x = [d["decile"] for d in item["interaction_deciles"]]
        axes[0].plot(x, [d["full_advantage_min_ade_20_pct"] for d in item["interaction_deciles"]], marker="o", label=item["split"])
        axes[1].plot(x, [d["full_advantage_min_fde_20_pct"] for d in item["interaction_deciles"]], marker="o", label=item["split"])
    for axis, name in zip(axes, ("minADE", "minFDE")):
        axis.axhline(0, color="black", linewidth=.8); axis.set_xlabel("Interaction-potential decile")
        axis.set_ylabel("Full-social advantage (%)"); axis.set_title(f"Social necessity: {name}"); axis.legend()
    figure.tight_layout(); figure.savefig(output / "interaction_decile_social_gain.png", dpi=180); plt.close(figure)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0"); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--splits", nargs="+", choices=("all", *SPLITS), default=["all"])
    p.add_argument("--bootstrap", type=int, default=2000); p.add_argument("--bootstrap-seed", type=int, default=20260828)
    p.add_argument("--block-frames", type=int, default=20); p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=Path, default=OUTPUT)
    return p.parse_args()


def main():
    args = parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    splits = list(SPLITS) if "all" in args.splits else args.splits
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    cfg, model_class = prepare(device)
    geometry = load_module("csna_geometry_helpers", WORKSPACE / "INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES" / "run_geometry_diagnostic.py")
    summaries, reproduction = [], {}
    for split in splits:
        run = run_dir(split, args.seed); indices_file = np.load(run / "split_indices.npz")
        train_indices, val_indices = np.asarray(indices_file["train"], dtype=np.int64), np.asarray(indices_file["validation"], dtype=np.int64)
        if args.max_samples is not None: val_indices = val_indices[:args.max_samples]
        info = json.loads((run / "split.json").read_text(encoding="utf-8")); data_path, num_path = Path(info["data_path"]), Path(info["num_path"])
        dataset = geometry.IndexedDataset(data_path, num_path, val_indices, cfg)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        print(f"[load] {split} checkpoint={run/'best_selected.pth'} samples={len(dataset)}", flush=True)
        model = build_model(split, args.seed, cfg, model_class, device)
        print(f"[infer] {split}/full-social", flush=True); full = infer(model, loader, device, self_only=False)
        print(f"[infer] {split}/self-only", flush=True); self_only = infer(model, loader, device, self_only=True)
        if not np.array_equal(full["index"], self_only["index"]): raise RuntimeError("Pairing failure")
        if args.max_samples is None: reproduction[split] = verify_reproduction(split, args.seed, full)
        raw = np.load(data_path, mmap_mode="r")
        scene = np.asarray(raw[val_indices, 0, 0, 0], dtype=np.int64)
        frame = np.asarray(raw[val_indices, 0, 0, 1], dtype=np.int64)
        potential = geometry.potential_score(data_path, num_path, val_indices, cfg)
        train_potential = geometry.potential_score(data_path, num_path, train_indices, cfg)
        split_output = output / split; split_output.mkdir(parents=True, exist_ok=True)
        summaries.append(analyse(split, full, self_only, potential, train_potential, scene, frame, args, split_output))
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    combined = {"training_performed": False, "strict_pairing": True, "full_social_reproduction": reproduction,
                "results": summaries, "overall_decision": "enter_csrs" if summaries and all(s["gate_decision"]["decision"] == "enter_csrs" for s in summaries) else "do_not_enter_csrs"}
    (output / "combined_summary.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(summaries, output)
    print(f"[done] decision={combined['overall_decision']} output={output}", flush=True)


if __name__ == "__main__": main()

