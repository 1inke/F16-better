"""EXP-172A: deterministic geometry-only K40-to-K20 distillability audit."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ASTRA_PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
if str(Path(sys.executable).resolve()).casefold() != str(ASTRA_PYTHON.resolve()).casefold():
    raise SystemExit(subprocess.run([str(ASTRA_PYTHON), *sys.argv]).returncode)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
CONFIG = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
V2_MODULE = WORKSPACE / "ANCHOR_CORRECTION_VALIDATION_SERIES" / "anchor_correction_validation_models.py"
FREE_MODULE = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
V2_SOURCE = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "EXP_156B_FOLD_LOCAL_CURRENT_DECODER" / "results"
FREE_SOURCE = WORKSPACE / "FOLD_STAGE_ATTRIBUTION_SERIES" / "EXP_157B_MATCHED_K20" / "results"
OUTPUT = ROOT / "EXP_172A_K40_K20_DISTILLABILITY"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")

sys.path.insert(0, str(ROOT))
from deterministic_kmedoids import deterministic_kmedoids  # noqa: E402
from set_distance import mean_off_diagonal, pairwise_trajectory_distance  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def v2_dir(fold: str, stage: str, seed: int) -> Path:
    return V2_SOURCE / fold / stage / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def free_dir(fold: str, seed: int) -> Path:
    return FREE_SOURCE / fold / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def build_model(fold, seed, kind, cfg, model_class, device):
    if kind == "v2":
        os.environ["EXP143_SOURCE_CHECKPOINT"] = str(v2_dir(fold, "h3", seed) / "best_selected.pth")
        module, experiment = load_module(f"exp172a_v2_{fold}", V2_MODULE), "v2_wide_endpoint_correction"
        checkpoint = v2_dir(fold, "v2", seed) / "best_selected.pth"
    else:
        module, experiment = load_module(f"exp172a_free_{fold}", FREE_MODULE), "kinematic_k20"
        checkpoint = free_dir(fold, seed) / "best_selected.pth"
    model = model_class(cfg, "f16_temporal_social_cross")
    module.replace_decoder(model, cfg, experiment)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def candidate_metrics(candidate: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    point = torch.linalg.vector_norm(candidate - target[:, None], dim=-1)
    ade = point.mean(dim=-1)
    fde = point[..., -1]
    min_ade, ade_winner = ade.min(dim=1)
    min_fde = fde.min(dim=1).values
    same_fde = fde.gather(1, ade_winner[:, None]).squeeze(1)
    return {"minADE": min_ade, "minFDE": min_fde, "sameFDE": same_fde, "J": min_ade + 0.5 * min_fde}


def observation_features(position: np.ndarray, num_valid: np.ndarray) -> dict[str, np.ndarray]:
    focal = position[:, 0]
    velocity = np.diff(focal, axis=1)
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration = np.diff(velocity, axis=1)
    last = position[:, :, -1]
    distance = np.linalg.norm(last - last[:, :1], axis=-1)
    valid = np.arange(position.shape[1])[None] < num_valid[:, None]
    valid[:, 0] = False
    return {
        "median_speed": np.median(speed, axis=1),
        "mean_acceleration": np.linalg.norm(acceleration, axis=-1).mean(axis=1),
        "neighbor_count_r4": ((distance < 4.0) & valid).sum(axis=1).astype(np.float32),
    }


def source_edges(fold: str, cfg) -> dict[str, np.ndarray]:
    raw = np.load(DATA / f"{fold}_data_train.npy", mmap_mode="r")
    num = np.minimum(np.asarray(np.load(DATA / f"{fold}_num_train.npy", mmap_mode="r")), int(cfg.DATA.MAX_AGENTS))
    position = np.asarray(raw[:, : int(cfg.DATA.MAX_AGENTS), :8, -2:], dtype=np.float32) / float(cfg.PREDICTION.TRAJ_SCALE)
    feat = observation_features(position, num)
    return {
        "speed": np.quantile(feat["median_speed"], [0.25, 0.5, 0.75]),
        "acceleration": np.quantile(feat["mean_acceleration"], [0.25, 0.5, 0.75]),
        "density": np.quantile(feat["neighbor_count_r4"], [1 / 3, 2 / 3]),
    }


def aggregate_metrics(values: dict[str, list[np.ndarray]]) -> dict[str, float]:
    return {key: float(np.concatenate(parts).mean()) for key, parts in values.items()}


def safe_retention(base: float, union: float, core: float) -> float | None:
    denominator = base - union
    return None if abs(denominator) < 1e-10 else (base - core) / denominator


def audit_fold(fold, seed, cfg, model_class, dataset_class, normalize, device, batch_size, max_samples):
    dataset = dataset_class(DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy", cfg)
    if max_samples:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    v2_model = build_model(fold, seed, "v2", cfg, model_class, device)
    free_model = build_model(fold, seed, "free", cfg, model_class, device)
    metric_parts = {name: {key: [] for key in ("minADE", "minFDE", "sameFDE", "J")} for name in ("v2", "union", "core")}
    per_sample: list[dict] = []
    compositions, teacher_diversity, core_diversity, cover_distance = [], [], [], []
    offset = 0
    with torch.inference_mode():
        for past, future, num, _, _, _ in loader:
            past, future, num = past.to(device), future.to(device), num.to(device).int()
            past, future = normalize(past, future)
            center = past.mean(dim=-2, keepdim=True)
            args = (past - center, None, None)
            kwargs = {"mode": "testing", "num_valid": num, "graph_loc": past}
            structured = v2_model(*args, **kwargs)[2][:, 0] + center[:, 0].unsqueeze(1)
            free = free_model(*args, **kwargs)[2][:, 0] + center[:, 0].unsqueeze(1)
            target = future[:, 0]
            teacher = torch.cat((structured, free), dim=1)
            distance = pairwise_trajectory_distance(teacher)
            selected = deterministic_kmedoids(distance, k=20, max_swaps=40)
            core = teacher.gather(1, selected[:, :, None, None].expand(-1, -1, teacher.shape[2], 2))
            metrics = {"v2": candidate_metrics(structured, target), "union": candidate_metrics(teacher, target), "core": candidate_metrics(core, target)}
            for name in metrics:
                for key, tensor in metrics[name].items():
                    metric_parts[name][key].append(tensor.cpu().numpy())
            structured_count = (selected < 20).sum(dim=1)
            selected_distance = distance.gather(1, selected[:, :, None].expand(-1, -1, 40)).gather(2, selected[:, None, :].expand(-1, 20, -1))
            cover = distance.gather(2, selected[:, None, :].expand(-1, 40, -1)).min(dim=2).values.mean(dim=1)
            td = mean_off_diagonal(distance)
            cd = mean_off_diagonal(selected_distance)
            arrays = {name: {key: value.cpu().numpy() for key, value in group.items()} for name, group in metrics.items()}
            sc, td_np, cd_np, cover_np = structured_count.cpu().numpy(), td.cpu().numpy(), cd.cpu().numpy(), cover.cpu().numpy()
            compositions.append(sc); teacher_diversity.append(td_np); core_diversity.append(cd_np); cover_distance.append(cover_np)
            for index in range(len(target)):
                row = {"fold": fold, "seed": seed, "sample_id": offset + index, "structured_count": int(sc[index]), "free_count": int(20 - sc[index]), "teacher_pairwise_distance": float(td_np[index]), "coreset_pairwise_distance": float(cd_np[index]), "teacher_to_coreset_distance": float(cover_np[index])}
                for name in metrics:
                    for key in metrics[name]:
                        row[f"{name}_{key}"] = float(arrays[name][key][index])
                per_sample.append(row)
            offset += len(target)
    aggregate = {name: aggregate_metrics(group) for name, group in metric_parts.items()}
    counts = np.concatenate(compositions)
    td_all, cd_all, cover_all = np.concatenate(teacher_diversity), np.concatenate(core_diversity), np.concatenate(cover_distance)
    row = {"fold": fold, "seed": seed, "samples": len(per_sample)}
    for name, group in aggregate.items():
        for key, value in group.items():
            row[f"{name}_{key}"] = value
    for key in ("minADE", "minFDE", "sameFDE", "J"):
        row[f"delta_core_vs_v2_{key}_pct"] = 100.0 * (aggregate["v2"][key] / aggregate["core"][key] - 1.0)
        row[f"union_gain_{key}"] = aggregate["v2"][key] - aggregate["union"][key]
        row[f"retention_{key}"] = safe_retention(aggregate["v2"][key], aggregate["union"][key], aggregate["core"][key])
    row.update({"structured_count_mean": float(counts.mean()), "structured_count_p10": float(np.quantile(counts, 0.10)), "structured_count_p50": float(np.quantile(counts, 0.50)), "structured_count_p90": float(np.quantile(counts, 0.90)), "free_count_mean": float(20.0 - counts.mean()), "teacher_pairwise_distance": float(td_all.mean()), "coreset_pairwise_distance": float(cd_all.mean()), "teacher_to_coreset_distance": float(cover_all.mean())})

    raw = np.load(DATA / f"{fold}_data_test.npy", mmap_mode="r")
    n = len(per_sample)
    num_raw = np.minimum(np.asarray(np.load(DATA / f"{fold}_num_test.npy", mmap_mode="r"))[:n], int(cfg.DATA.MAX_AGENTS))
    pos = np.asarray(raw[:n, : int(cfg.DATA.MAX_AGENTS), :8, -2:], dtype=np.float32) / float(cfg.PREDICTION.TRAJ_SCALE)
    features, edges = observation_features(pos, num_raw), source_edges(fold, cfg)
    bins = {"speed": np.digitize(features["median_speed"], edges["speed"], right=True) + 1, "acceleration": np.digitize(features["mean_acceleration"], edges["acceleration"], right=True) + 1, "density": np.digitize(features["neighbor_count_r4"], edges["density"], right=True) + 1}
    composition_rows = []
    for variable, values in (("ALL", np.ones(n, dtype=int)), *bins.items()):
        levels = [1] if variable == "ALL" else sorted(np.unique(values).tolist())
        for level in levels:
            mask = np.ones(n, dtype=bool) if variable == "ALL" else values == level
            subset = counts[mask]
            composition_rows.append({"fold": fold, "variable": variable, "bin": "ALL" if variable == "ALL" else int(level), "samples": int(mask.sum()), "structured_count_mean": float(subset.mean()), "structured_count_p10": float(np.quantile(subset, 0.10)), "structured_count_p50": float(np.quantile(subset, 0.50)), "structured_count_p90": float(np.quantile(subset, 0.90)), "free_count_mean": float(20.0 - subset.mean())})
    del v2_model, free_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return per_sample, row, composition_rows


def make_decision(rows: list[dict]) -> dict:
    macro = lambda key: float(np.mean([row[key] for row in rows]))
    base = {key: macro(f"v2_{key}") for key in ("minADE", "minFDE", "sameFDE", "J")}
    union = {key: macro(f"union_{key}") for key in base}
    core = {key: macro(f"core_{key}") for key in base}
    retention = {key: safe_retention(base[key], union[key], core[key]) for key in base}
    improvement = {key: 100.0 * (base[key] / core[key] - 1.0) for key in base}
    improved_folds = sum(row["core_J"] < row["v2_J"] for row in rows)
    worst_regression = min(row["delta_core_vs_v2_J_pct"] for row in rows)
    passed = improvement["J"] >= 2.0 and retention["J"] is not None and retention["J"] >= 0.60 and improved_folds >= 4 and worst_regression >= -2.0 and retention["minADE"] is not None and retention["minADE"] >= 0.50 and retention["minFDE"] is not None and retention["minFDE"] >= 0.50
    return {"passed": passed, "next_experiment": "EXP-172B" if passed else "STOP_SET_DISTILLATION", "macro": {"v2": base, "union_k40": union, "coreset_k20": core}, "macro_improvement_core_vs_v2_pct": improvement, "macro_union_gain_retention": retention, "folds_core_better_J": improved_folds, "worst_fold_J_improvement_pct": worst_regression, "criteria": {"J_improvement_ge_2pct": improvement["J"] >= 2.0, "J_retention_ge_0p60": retention["J"] is not None and retention["J"] >= 0.60, "ADE_retention_ge_0p50": retention["minADE"] is not None and retention["minADE"] >= 0.50, "FDE_retention_ge_0p50": retention["minFDE"] is not None and retention["minFDE"] >= 0.50, "at_least_4_folds_better": improved_folds >= 4, "worst_fold_regression_le_2pct": worst_regression >= -2.0}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    required = [CONFIG, TRAINER, V2_MODULE, FREE_MODULE]
    for fold in folds:
        required += [v2_dir(fold, "h3", args.seed) / "best_selected.pth", v2_dir(fold, "v2", args.seed) / "best_selected.pth", free_dir(fold, args.seed) / "best_selected.pth", DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing:\n" + "\n".join(missing))
    if args.dry_run:
        print(json.dumps({"experiment": "EXP-172A", "teacher": "official V2 K20 + matched free K20", "folds": folds, "seed": args.seed, "max_samples": args.max_samples, "output": str(args.output)}, indent=2))
        return

    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(WORKSPACE / "FRONT_ENCODER_SERIES")]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from train_front_variant import TrajectoryDataset, normalize_to_focal_heading

    device = torch.device(args.device)
    cfg = load_locked_q20(CONFIG)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0]
    all_samples, fold_rows, composition_rows = [], [], []
    for fold in folds:
        print(f"[EXP-172A] {fold}", flush=True)
        samples, row, composition = audit_fold(fold, args.seed, cfg, FrontEncoderASTRA, TrajectoryDataset, normalize_to_focal_heading, device, args.batch_size, args.max_samples)
        all_samples.extend(samples); fold_rows.append(row); composition_rows.extend(composition)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    complete = set(folds) == set(FOLDS) and args.max_samples == 0
    decision = make_decision(fold_rows) if complete else None
    suffix = "" if complete else "_partial"
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / f"per_sample_summary{suffix}.csv", all_samples)
    write_csv(args.output / f"per_fold_summary{suffix}.csv", fold_rows)
    write_csv(args.output / f"all_runs{suffix}.csv", fold_rows)
    write_csv(args.output / f"family_composition{suffix}.csv", composition_rows)
    write_csv(args.output / f"representability_summary{suffix}.csv", fold_rows)
    payload = {"experiment": "EXP-172A K40-to-K20 Set Distillability Audit", "teacher_source": {"structured": "EXP-164-compatible official V2 K20", "free": "EXP-157B matched-protocol kinematic K20", "same_fold_seed_sample": True, "ground_truth_used_for_coreset": False}, "complete": complete, "per_fold": fold_rows, "decision": decision}
    (args.output / f"summary{suffix}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
