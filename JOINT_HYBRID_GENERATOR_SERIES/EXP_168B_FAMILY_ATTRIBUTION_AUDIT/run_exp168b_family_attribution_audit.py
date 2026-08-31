"""EXP-168B: no-training family attribution audit for EXP-168A.

The audit evaluates the exact official outer-test folds and separates the
structured modes [0:10], free modes [10:20], and their K20 union.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# Users often invoke this file with the Windows system Python, which does not
# contain the project's PyTorch installation.  Relaunch transparently in the
# locked ASTRA environment before importing numpy/torch.
ASTRA_PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
current_python = Path(sys.executable).resolve()
locked_python = ASTRA_PYTHON.resolve() if ASTRA_PYTHON.is_file() else ASTRA_PYTHON
if str(current_python).casefold() != str(locked_python).casefold():
    if not ASTRA_PYTHON.is_file():
        raise FileNotFoundError(f"The locked ASTRA Python was not found: {ASTRA_PYTHON}")
    print(f"[bootstrap] relaunching with locked environment: {ASTRA_PYTHON}", flush=True)
    completed = subprocess.run([str(ASTRA_PYTHON), *sys.argv], check=False)
    raise SystemExit(completed.returncode)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
SERIES = HERE.parent
WORKSPACE = SERIES.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
CONFIG = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = SERIES / "hybrid_generator_models.py"
SOURCE = SERIES / "EXP_168A_HYBRID_K20_10_10" / "results"
DEFAULT_OUT = HERE
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
FAMILIES = ("structured", "free", "union")
OBS = 8


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_path(fold: str, seed: int) -> Path:
    return SOURCE / fold / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"


def observation_features(positions: np.ndarray, num_valid: np.ndarray) -> dict[str, np.ndarray]:
    focal = positions[:, 0]
    velocity = np.diff(focal, axis=1)
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration = np.diff(velocity, axis=1)
    acc_norm = np.linalg.norm(acceleration, axis=-1)
    v0, v1 = velocity[:, :-1], velocity[:, 1:]
    denom = np.linalg.norm(v0, axis=-1) * np.linalg.norm(v1, axis=-1)
    cosine = np.sum(v0 * v1, axis=-1) / np.maximum(denom, 1.0e-8)
    turns = np.arccos(np.clip(cosine, -1.0, 1.0)); turns[denom < 1.0e-8] = 0.0
    start, end = focal[:, 0], focal[:, -1]
    chord = end - start
    chord_norm = np.linalg.norm(chord, axis=-1)
    rel = focal - start[:, None]
    cross = np.abs(rel[..., 0] * chord[:, None, 1] - rel[..., 1] * chord[:, None, 0])
    nonlinearity = np.mean(cross / np.maximum(chord_norm[:, None], 1.0e-6), axis=1)
    last = positions[:, :, -1]
    distance = np.linalg.norm(last - last[:, :1], axis=-1)
    valid = np.arange(positions.shape[1])[None] < num_valid[:, None]
    neighbor = valid.copy(); neighbor[:, 0] = False
    masked = np.where(neighbor, distance, np.nan)
    nearest = np.full(len(positions), np.nan, dtype=np.float64)
    mean_neighbor = np.full(len(positions), np.nan, dtype=np.float64)
    has_neighbor = neighbor.any(axis=1)
    nearest[has_neighbor] = np.nanmin(masked[has_neighbor], axis=1)
    mean_neighbor[has_neighbor] = np.nanmean(masked[has_neighbor], axis=1)
    return {
        "mean_step_speed": speed.mean(axis=1),
        "median_step_speed": np.median(speed, axis=1),
        "last_step_speed": speed[:, -1],
        "obs_displacement": chord_norm,
        "mean_acceleration": acc_norm.mean(axis=1),
        "mean_turn_angle": turns.mean(axis=1),
        "observation_nonlinearity": nonlinearity,
        "valid_agent_count": num_valid.astype(np.float64),
        "nearest_neighbor_distance": nearest,
        "neighbor_count_r2": ((distance < 2.0) & neighbor).sum(axis=1).astype(np.float64),
        "neighbor_count_r4": ((distance < 4.0) & neighbor).sum(axis=1).astype(np.float64),
        "neighbor_count_r6": ((distance < 6.0) & neighbor).sum(axis=1).astype(np.float64),
        "mean_neighbor_distance": mean_neighbor,
    }


def source_boundaries(fold: str, scale: float, max_agents: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(DATA / f"{fold}_data_train.npy", mmap_mode="r")
    num = np.asarray(np.load(DATA / f"{fold}_num_train.npy", mmap_mode="r"))
    positions = np.asarray(raw[:, :max_agents, :OBS, -2:], dtype=np.float32) / scale
    features = observation_features(positions, np.minimum(num, max_agents))
    speed = np.quantile(features["median_step_speed"], (0.25, 0.50, 0.75))
    density = np.quantile(features["neighbor_count_r4"], (1.0 / 3.0, 2.0 / 3.0))
    return speed, density


def candidate_metrics(candidates: np.ndarray, target: np.ndarray, past_last: np.ndarray) -> dict[str, np.ndarray]:
    error = np.linalg.norm(candidates - target[:, None], axis=-1)
    ade = error.mean(axis=-1)
    fde = error[..., -1]
    ade_winner = ade.argmin(axis=1)
    endpoint = candidates[:, :, -1] - past_last[:, None]
    endpoint_radius = np.linalg.norm(endpoint, axis=-1)
    flat = candidates.reshape(candidates.shape[0], candidates.shape[1], -1)
    path_pairwise = np.linalg.norm(flat[:, :, None] - flat[:, None, :], axis=-1) / math.sqrt(candidates.shape[-2])
    upper = np.triu_indices(candidates.shape[1], k=1)
    pairs = path_pairwise[:, upper[0], upper[1]]
    return {
        "minADE": ade.min(axis=1),
        "minFDE": fde.min(axis=1),
        "same_path_FDE": fde[np.arange(len(fde)), ade_winner],
        "J": ade.min(axis=1) + 0.5 * fde.min(axis=1),
        "ade_winner_mode": ade_winner.astype(np.int32),
        "fde_winner_mode": fde.argmin(axis=1).astype(np.int32),
        "endpoint_radius_q95": np.quantile(endpoint_radius, 0.95, axis=1),
        "endpoint_radius_max": endpoint_radius.max(axis=1),
        "lateral_std": endpoint[..., 1].std(axis=1),
        "longitudinal_std": endpoint[..., 0].std(axis=1),
        "pairwise_path_distance": pairs.mean(axis=1),
        "duplicate_path_010": (pairs < 0.10).mean(axis=1),
    }


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def build_model(fold: str, seed: int, cfg, FrontEncoderASTRA, device: torch.device):
    checkpoint = checkpoint_path(fold, seed)
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    module = load_module(f"exp168b_decoder_{fold}_{seed}", DECODER)
    module.replace_decoder(model, cfg, "hybrid_k20_10_10")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def collect_fold(fold: str, seed: int, cfg, FrontEncoderASTRA, TrajectoryDataset,
                 normalize_to_focal_heading, device: torch.device, batch_size: int,
                 max_samples: int) -> tuple[list[dict], list[dict], list[dict]]:
    dataset = TrajectoryDataset(DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy", cfg)
    if max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(fold, seed, cfg, FrontEncoderASTRA, device)
    speed_edges, density_edges = source_boundaries(fold, float(cfg.PREDICTION.TRAJ_SCALE), int(cfg.DATA.MAX_AGENTS))
    all_past, all_target, all_num, all_candidates = [], [], [], []
    with torch.inference_mode():
        for past, future, num_valid, _, _, _ in loader:
            past = past.to(device); future = future.to(device); num_valid = num_valid.to(device).int()
            past, future = normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            prediction = model(
                past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past,
            )[2] + center.unsqueeze(-3)
            all_past.append(past.cpu().numpy())
            all_target.append(future[:, 0].cpu().numpy())
            all_num.append(num_valid.cpu().numpy())
            all_candidates.append(prediction[:, 0].cpu().numpy())
    past = np.concatenate(all_past); target = np.concatenate(all_target)
    num_valid = np.concatenate(all_num); candidates = np.concatenate(all_candidates)
    observed = observation_features(past, num_valid)
    speed_bin = np.digitize(observed["median_step_speed"], speed_edges, right=True) + 1
    density_index = np.digitize(observed["neighbor_count_r4"], density_edges, right=True)
    density_name = np.asarray(("low", "medium", "high"), dtype=object)[density_index]
    family_candidates = {
        "structured": candidates[:, :10], "free": candidates[:, 10:], "union": candidates,
    }
    metrics = {
        family: candidate_metrics(values, target, past[:, 0, -1])
        for family, values in family_candidates.items()
    }
    union_error = np.linalg.norm(candidates - target[:, None], axis=-1)
    union_ade = union_error.mean(axis=-1)
    union_fde = union_error[..., -1]
    ade_family = np.where(union_ade.argmin(axis=1) < 10, "structured", "free")
    fde_family = np.where(union_fde.argmin(axis=1) < 10, "structured", "free")
    per_sample = []
    for index in range(len(target)):
        row = {
            "fold": fold, "seed": seed, "sample_id": index,
            "speed_bin": int(speed_bin[index]), "density_bin": str(density_name[index]),
            "union_ade_winner_family": str(ade_family[index]),
            "union_fde_winner_family": str(fde_family[index]),
            "structured_J_minus_union": float(metrics["structured"]["J"][index] - metrics["union"]["J"][index]),
            "free_J_minus_union": float(metrics["free"]["J"][index] - metrics["union"]["J"][index]),
        }
        for key, values in observed.items(): row[key] = float(values[index])
        for family in FAMILIES:
            for key, values in metrics[family].items():
                row[f"{family}_{key}"] = float(values[index])
        per_sample.append(row)
    fold_rows = []
    for family in FAMILIES:
        row = {"fold": fold, "seed": seed, "family": family, "modes": 20 if family == "union" else 10, "samples": len(target)}
        for key in ("minADE", "minFDE", "same_path_FDE", "J", "endpoint_radius_q95", "endpoint_radius_max",
                    "lateral_std", "longitudinal_std", "pairwise_path_distance", "duplicate_path_010"):
            row[key] = finite_mean(metrics[family][key])
        row["ade_winner_fraction"] = float((ade_family == family).mean()) if family != "union" else 1.0
        row["fde_winner_fraction"] = float((fde_family == family).mean()) if family != "union" else 1.0
        fold_rows.append(row)
    binned_rows = []
    for speed in range(1, 5):
        for density in ("low", "medium", "high"):
            mask = (speed_bin == speed) & (density_name == density)
            if not mask.any(): continue
            for family in FAMILIES:
                row = {"fold": fold, "seed": seed, "speed_bin": speed, "density_bin": density,
                       "family": family, "samples": int(mask.sum())}
                for key in ("minADE", "minFDE", "same_path_FDE", "J", "endpoint_radius_q95", "lateral_std",
                            "pairwise_path_distance", "duplicate_path_010"):
                    row[key] = finite_mean(metrics[family][key][mask])
                if family != "union":
                    row["ade_winner_fraction"] = float((ade_family[mask] == family).mean())
                    row["fde_winner_fraction"] = float((fde_family[mask] == family).mean())
                else:
                    row["ade_winner_fraction"] = 1.0; row["fde_winner_fraction"] = 1.0
                binned_rows.append(row)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return per_sample, fold_rows, binned_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    required = [CONFIG, TRAINER, DECODER]
    required += [DATA / f"{fold}_data_test.npy" for fold in folds]
    required += [DATA / f"{fold}_num_test.npy" for fold in folds]
    required += [DATA / f"{fold}_data_train.npy" for fold in folds]
    required += [DATA / f"{fold}_num_train.npy" for fold in folds]
    required += [checkpoint_path(fold, args.seed) for fold in folds]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    if args.dry_run:
        print(json.dumps({"folds": folds, "seed": args.seed, "max_samples": args.max_samples,
                          "source": str(SOURCE), "output": str(args.output_dir)}, indent=2))
        print("[done] dry-run passed"); return
    # The audit must never activate the training-only atexit logger.
    os.environ.pop("EXP168_DIAGNOSTICS_PATH", None)
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(WORKSPACE / "FRONT_ENCODER_SERIES")]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from train_front_variant import TrajectoryDataset, normalize_to_focal_heading
    device = torch.device(args.device)
    cfg = load_locked_q20(CONFIG)
    cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device); cfg.device_list = [device.index or 0]
    all_sample: list[dict] = []; all_fold: list[dict] = []; all_binned: list[dict] = []
    for fold in folds:
        print(f"[audit] {fold}", flush=True)
        sample, fold_rows, binned = collect_fold(
            fold, args.seed, cfg, FrontEncoderASTRA, TrajectoryDataset,
            normalize_to_focal_heading, device, args.batch_size, args.max_samples,
        )
        all_sample += sample; all_fold += fold_rows; all_binned += binned
        print(f"[done] {fold}: {len(sample)} samples", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.max_samples > 0 else "full"
    write_csv(args.output_dir / f"per_sample_diagnostics_{tag}.csv", all_sample)
    write_csv(args.output_dir / f"per_fold_summary_{tag}.csv", all_fold)
    write_csv(args.output_dir / f"binned_family_summary_{tag}.csv", all_binned)
    complete = set(folds) == set(FOLDS) and args.max_samples == 0
    family_macro = []
    for family in FAMILIES:
        group = [row for row in all_fold if row["family"] == family]
        if group:
            family_macro.append({"family": family, **{
                key: sum(float(row[key]) for row in group) / len(group)
                for key in ("minADE", "minFDE", "same_path_FDE", "J", "ade_winner_fraction", "fde_winner_fraction")
            }})
    payload = {
        "experiment": "EXP-168B Hybrid Family Attribution Audit", "complete": complete,
        "trained_new_model": False, "folds": list(folds), "seed": args.seed,
        "mode_partition": {"structured": [0, 9], "free": [10, 19]},
        "family_macro": family_macro, "per_fold": all_fold,
        "protocol": {"outer_test_diagnostic_only": True, "speed_density_bins_from_outer_train": True,
                     "no_model_or_hyperparameter_selection": True},
    }
    (args.output_dir / f"summary_{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"complete": complete, "family_macro": family_macro}, indent=2), flush=True)
    print(f"[done] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
