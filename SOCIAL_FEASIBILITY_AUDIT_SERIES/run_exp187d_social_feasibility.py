"""EXP-187D: zero-training social feasibility audit on final candidates."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASTRA_ROOT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = ASTRA_ROOT / "datasets" / "eth_ucy" / "processed_data_diverse"
CFG_PATH = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
TRAINER = ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
MODEL_FILE = ROOT / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
ADAPTER = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
EXP187A = ROOT / "FINAL_END_TO_END_ENCODER_ABLATION_SERIES" / "EXP_187A_END_TO_END_ENCODER_ABLATION" / "results"
EXP173 = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results"
DEFAULT_OUT = HERE / "EXP_187D_SOCIAL_FEASIBILITY"

FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
MODELS = {
    "e0": {"variant": "f0_baseline", "root": EXP187A, "label": "e0"},
    "e1": {"variant": "f3_factorized", "root": EXP187A, "label": "e1"},
    "e2": {"variant": "f4_relation_tokens", "root": EXP173, "label": "e2"},
    "f16": {"variant": "f16_temporal_social_cross", "root": EXP173, "label": "e3"},
}
COLLISION_THRESHOLD = 0.20
NEAR_THRESHOLD = 0.50
CACHE_PROTOCOL = "focal_prediction_vs_gt_neighbors_v2"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(device: torch.device):
    os.chdir(ASTRA_ROOT)
    sys.path[:0] = [str(ASTRA_ROOT), str(ASTRA_ROOT / "scripts"), str(ROOT / "FRONT_ENCODER_SERIES")]
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20

    cfg = load_locked_q20(CFG_PATH)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    return cfg


def checkpoint_paths(model_name: str, fold: str, seed: int) -> tuple[Path, Path]:
    meta = MODELS[model_name]
    base = meta["root"] / fold
    h3 = base / "h3" / meta["label"] / meta["variant"] / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"
    final = base / "v2" / meta["label"] / meta["variant"] / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"
    return h3, final


def build_model(model_name: str, fold: str, seed: int, device: torch.device, models, adapter):
    meta = MODELS[model_name]
    h3, final = checkpoint_paths(model_name, fold, seed)
    if not h3.is_file() or not final.is_file():
        raise FileNotFoundError(h3 if not h3.is_file() else final)
    cfg = configure(device)
    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3)
    os.environ["EXP173_DECODER_INIT_SEED"] = str(seed)
    model = models.FrontEncoderASTRA(cfg, meta["variant"])
    adapter.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
    model.load_state_dict(torch.load(final, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval(), cfg


def empty_accumulator() -> dict:
    return {
        "samples": 0, "eligible_scenes": 0, "colliding_scenes": 0, "near_scenes": 0,
        "valid_agents": 0, "pairs": 0, "collision_pairs": 0, "near_pairs": 0,
        "pair_timesteps": 0, "collision_pairtimes": 0, "near_pairtimes": 0,
        "min_pair_distance_sum": 0.0, "min_pair_distance_count": 0,
    }


def update_social(
    acc: dict,
    focal_trajectory: torch.Tensor,
    neighbour_trajectories: torch.Tensor,
    num_valid: torch.Tensor,
) -> None:
    """Accumulate focal-to-valid-neighbour statistics.

    Only the focal output is supervised by the locked training protocol. The
    neighbour paths therefore come from ground truth for both prediction and
    reference measurements.
    """
    batch, steps = focal_trajectory.shape[:2]
    neighbours = neighbour_trajectories.shape[1]
    if neighbours == 0:
        acc["samples"] += batch
        acc["valid_agents"] += int(num_valid.sum().item())
        return
    distance = torch.linalg.vector_norm(
        focal_trajectory[:, None] - neighbour_trajectories, dim=-1
    )
    neighbour_index = torch.arange(neighbours, device=focal_trajectory.device)
    pair_mask = neighbour_index[None] < (num_valid - 1).clamp_min(0)[:, None]
    pair_distance = distance[pair_mask]
    pair_count = int(pair_distance.shape[0])
    eligible = num_valid >= 2
    acc["samples"] += batch
    acc["eligible_scenes"] += int(eligible.sum().item())
    acc["valid_agents"] += int(num_valid.sum().item())
    acc["pairs"] += pair_count
    acc["pair_timesteps"] += pair_count * steps
    if pair_count == 0:
        return
    pair_min = pair_distance.min(dim=-1).values
    pair_collision = pair_min < COLLISION_THRESHOLD
    pair_near = pair_min < NEAR_THRESHOLD
    acc["collision_pairs"] += int(pair_collision.sum().item())
    acc["near_pairs"] += int(pair_near.sum().item())
    acc["collision_pairtimes"] += int((pair_distance < COLLISION_THRESHOLD).sum().item())
    acc["near_pairtimes"] += int((pair_distance < NEAR_THRESHOLD).sum().item())
    acc["min_pair_distance_sum"] += float(pair_min.sum().item())
    acc["min_pair_distance_count"] += pair_count

    # Scene rates ask whether the focal path conflicts with any valid neighbour.
    scene_collision = torch.zeros(batch, dtype=torch.bool, device=focal_trajectory.device)
    scene_near = torch.zeros_like(scene_collision)
    for b in range(batch):
        local = distance[b, pair_mask[b]]
        if local.numel():
            local_min = local.min(dim=-1).values
            scene_collision[b] = (local_min < COLLISION_THRESHOLD).any()
            scene_near[b] = (local_min < NEAR_THRESHOLD).any()
    acc["colliding_scenes"] += int((scene_collision & eligible).sum().item())
    acc["near_scenes"] += int((scene_near & eligible).sum().item())


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def finalize_social(acc: dict, prefix: str) -> dict:
    return {
        f"{prefix}_eligible_scenes": acc["eligible_scenes"],
        f"{prefix}_pair_count": acc["pairs"],
        f"{prefix}_scene_collision_rate": safe_ratio(acc["colliding_scenes"], acc["eligible_scenes"]),
        f"{prefix}_scene_near_collision_rate": safe_ratio(acc["near_scenes"], acc["eligible_scenes"]),
        f"{prefix}_pair_collision_rate": safe_ratio(acc["collision_pairs"], acc["pairs"]),
        f"{prefix}_pair_near_collision_rate": safe_ratio(acc["near_pairs"], acc["pairs"]),
        f"{prefix}_time_under_collision_threshold": safe_ratio(acc["collision_pairtimes"], acc["pair_timesteps"]),
        f"{prefix}_time_under_near_threshold": safe_ratio(acc["near_pairtimes"], acc["pair_timesteps"]),
        f"{prefix}_mean_min_pairwise_distance": safe_ratio(acc["min_pair_distance_sum"], acc["min_pair_distance_count"]),
    }


def collect(model, loader, trainer, device: torch.device) -> dict:
    pred_acc, gt_acc = empty_accumulator(), empty_accumulator()
    focal_ade_sum = focal_fde_sum = oracle_ade_sum = oracle_fde_sum = 0.0
    focal_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            raw = output[2] if isinstance(output, (tuple, list)) else output
            prediction = raw + center.unsqueeze(-3)
            logits = model.last_mode_logits
            if prediction.ndim != 5 or logits.ndim != 3:
                raise RuntimeError(f"Unexpected prediction/logit shape: {prediction.shape}, {logits.shape}")
            selected_index = logits.argmax(dim=-1)
            gather_index = selected_index[:, :, None, None, None].expand(-1, -1, 1, prediction.shape[-2], 2)
            selected = prediction.gather(2, gather_index).squeeze(2)
            update_social(pred_acc, selected[:, 0], future[:, 1:], num_valid)
            update_social(gt_acc, future[:, 0], future[:, 1:], num_valid)

            focal_distance = torch.linalg.vector_norm(prediction[:, 0] - future[:, 0, None], dim=-1)
            rows = torch.arange(prediction.shape[0], device=device)
            focal_selected = selected_index[:, 0]
            focal_ade_sum += float(focal_distance.mean(-1)[rows, focal_selected].sum().item())
            focal_fde_sum += float(focal_distance[..., -1][rows, focal_selected].sum().item())
            oracle_ade_sum += float(focal_distance.mean(-1).min(-1).values.sum().item())
            oracle_fde_sum += float(focal_distance[..., -1].min(-1).values.sum().item())
            focal_count += prediction.shape[0]
    result = {
        "samples": focal_count,
        "top1_ADE": safe_ratio(focal_ade_sum, focal_count),
        "top1_FDE": safe_ratio(focal_fde_sum, focal_count),
        "minADE20": safe_ratio(oracle_ade_sum, focal_count),
        "minFDE20": safe_ratio(oracle_fde_sum, focal_count),
        "elapsed_seconds": time.perf_counter() - started,
        **finalize_social(pred_acc, "pred"),
        **finalize_social(gt_acc, "gt"),
    }
    result["excess_pair_collision_vs_gt"] = result["pred_pair_collision_rate"] - result["gt_pair_collision_rate"]
    result["excess_pair_near_collision_vs_gt"] = result["pred_pair_near_collision_rate"] - result["gt_pair_near_collision_rate"]
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # During a resumable multi-fold run no macro row exists until every
        # requested fold for at least one model has completed.
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], output_dir: Path, folds: tuple[str, ...], model_names: tuple[str, ...], seed: int) -> dict:
    social_metrics = (
        "pred_scene_collision_rate", "pred_scene_near_collision_rate",
        "pred_pair_collision_rate", "pred_pair_near_collision_rate",
        "pred_time_under_collision_threshold", "pred_time_under_near_threshold",
        "pred_mean_min_pairwise_distance", "excess_pair_collision_vs_gt",
        "excess_pair_near_collision_vs_gt",
    )
    accuracy_metrics = ("minADE20", "minFDE20", "top1_ADE", "top1_FDE")
    macro = []
    for name in model_names:
        group = [row for row in rows if row["model"] == name]
        if len(group) != len(folds):
            continue
        record = {"model": name, "variant": MODELS[name]["variant"], "folds": len(group), "seed": seed}
        for key in (*accuracy_metrics, *social_metrics):
            record[key] = float(np.mean([float(row[key]) for row in group]))
        macro.append(record)
    write_csv(output_dir / "per_fold.csv", rows)
    write_csv(output_dir / "collision_metrics.csv", macro)

    decision = "AUDIT_COMPLETE"
    if all(name in {row["model"] for row in macro} for name in ("e0", "f16")):
        lookup = {row["model"]: row for row in macro}
        e0, f16 = lookup["e0"], lookup["f16"]
        accuracy_better = f16["minADE20"] < e0["minADE20"] and f16["minFDE20"] < e0["minFDE20"]
        social_better = (
            f16["pred_pair_collision_rate"] < e0["pred_pair_collision_rate"]
            and f16["pred_pair_near_collision_rate"] < e0["pred_pair_near_collision_rate"]
        )
        if accuracy_better and social_better:
            decision = "ACCURACY_AND_SOCIAL_FEASIBILITY_SUPPORTED"
        elif accuracy_better:
            decision = "ACCURACY_SUPPORTED_SOCIAL_FEASIBILITY_NOT_SUPPORTED"
        else:
            decision = "SOCIAL_CLAIM_NOT_ESTABLISHED"
    payload = {
        "experiment": "EXP-187D Social Feasibility Audit", "complete": len(rows) == len(folds) * len(model_names),
        "protocol": {"folds": list(folds), "models": list(model_names), "seed": seed, "collision_threshold_m": COLLISION_THRESHOLD,
                     "near_collision_threshold_m": NEAR_THRESHOLD, "selection": "native focal-agent Top1",
                     "social_reference": "ground-truth valid-neighbour futures", "metric_protocol": CACHE_PROTOCOL,
                     "training": "none",
                     "aggregation": "five-fold equal-weight macro"},
        "runs": rows, "macro": macro, "decision": decision,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# EXP-187D Social Feasibility Audit", "", f"Decision: **{decision}**", "",
             "| Model | minADE20 | minFDE20 | Pair collision | Pair near collision | Mean min clearance |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in macro:
        lines.append(f"| {row['model'].upper()} | {row['minADE20']:.6f} | {row['minFDE20']:.6f} | {100*row['pred_pair_collision_rate']:.3f}% | {100*row['pred_pair_near_collision_rate']:.3f}% | {row['pred_mean_min_pairwise_distance']:.4f} m |")
    (output_dir / "RESULT_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-187D zero-training social feasibility audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--models", nargs="+", choices=("all", *MODELS), default=["all"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    model_names = tuple(MODELS) if "all" in args.models else tuple(args.models)
    output_dir = args.output_dir.resolve()
    required = [TRAINER, MODEL_FILE, ADAPTER, CFG_PATH]
    for fold in folds:
        required.extend((DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy"))
        for name in model_names:
            required.extend(checkpoint_paths(name, fold, args.seed))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    if args.dry_run:
        print(json.dumps({"status": "preflight_ok", "profiles": len(folds) * len(model_names), "folds": folds, "models": model_names}, indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # The dynamically loaded trainer/model modules import the original ASTRA
    # packages at module-import time, before build_model() calls configure().
    os.chdir(ASTRA_ROOT)
    sys.path[:0] = [str(ASTRA_ROOT), str(ASTRA_ROOT / "scripts"), str(ROOT / "FRONT_ENCODER_SERIES")]
    import sitecustomize  # noqa: F401
    trainer = load_module("exp187d_trainer", TRAINER)
    models = load_module("exp187d_models", MODEL_FILE)
    adapter = load_module("exp187d_adapter", ADAPTER)
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in model_names:
        for fold in folds:
            cache = profiles_dir / f"{name}_{fold}_seed{args.seed}.json"
            cached = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else None
            if cached is not None and cached.get("metric_protocol") == CACHE_PROTOCOL and not args.force:
                row = cached
                print(f"[skip] {name}/{fold} samples={row['samples']}", flush=True)
            else:
                print(f"[audit] {name}/{fold}/seed{args.seed}", flush=True)
                model, cfg = build_model(name, fold, args.seed, device, models, adapter)
                dataset = trainer.TrajectoryDataset(DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy", cfg)
                if args.max_samples > 0:
                    dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                row = {"metric_protocol": CACHE_PROTOCOL, "model": name, "variant": MODELS[name]["variant"], "fold": fold, "seed": args.seed,
                       **collect(model, loader, trainer, device)}
                cache.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[done] {name}/{fold} collision={100*row['pred_pair_collision_rate']:.3f}% near={100*row['pred_pair_near_collision_rate']:.3f}%", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            rows.append(row)
            aggregate(rows, output_dir, folds, model_names, args.seed)
    payload = aggregate(rows, output_dir, folds, model_names, args.seed)
    print(json.dumps({"complete": payload["complete"], "decision": payload["decision"]}, ensure_ascii=False), flush=True)
    print(f"[complete] {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
