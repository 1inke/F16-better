"""EXP-168C: inner-validation family-need predictability audit.

No outer-test sample is used. The fixed EXP-168A checkpoints are evaluated on
their saved 5% validation subsets, then observation-only classifiers predict
whether the structured or free family contains the best training-quality mode.
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


ASTRA_PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
current_python = Path(sys.executable).resolve()
locked_python = ASTRA_PYTHON.resolve() if ASTRA_PYTHON.is_file() else ASTRA_PYTHON
if str(current_python).casefold() != str(locked_python).casefold():
    if not ASTRA_PYTHON.is_file(): raise FileNotFoundError(ASTRA_PYTHON)
    print(f"[bootstrap] relaunching with locked environment: {ASTRA_PYTHON}", flush=True)
    completed = subprocess.run([str(ASTRA_PYTHON), *sys.argv], check=False)
    raise SystemExit(completed.returncode)

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
SPEED_FEATURES = ("mean_step_speed", "median_step_speed", "last_step_speed", "obs_displacement")
FULL_FEATURES = SPEED_FEATURES + (
    "mean_acceleration", "mean_turn_angle", "observation_nonlinearity",
    "valid_agent_count", "nearest_neighbor_distance", "neighbor_count_r2",
    "neighbor_count_r4", "neighbor_count_r6", "mean_neighbor_distance",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def run_dir(fold: str, seed: int) -> Path:
    return SOURCE / fold / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def observation_features(positions: np.ndarray, num_valid: np.ndarray) -> dict[str, np.ndarray]:
    focal = positions[:, 0]
    velocity = np.diff(focal, axis=1)
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration = np.diff(velocity, axis=1)
    v0, v1 = velocity[:, :-1], velocity[:, 1:]
    denom = np.linalg.norm(v0, axis=-1) * np.linalg.norm(v1, axis=-1)
    cosine = np.sum(v0 * v1, axis=-1) / np.maximum(denom, 1.0e-8)
    turns = np.arccos(np.clip(cosine, -1.0, 1.0)); turns[denom < 1.0e-8] = 0.0
    start, end = focal[:, 0], focal[:, -1]
    chord = end - start; chord_norm = np.linalg.norm(chord, axis=-1)
    relative = focal - start[:, None]
    cross = np.abs(relative[..., 0] * chord[:, None, 1] - relative[..., 1] * chord[:, None, 0])
    last = positions[:, :, -1]
    distance = np.linalg.norm(last - last[:, :1], axis=-1)
    valid = np.arange(positions.shape[1])[None] < num_valid[:, None]
    neighbor = valid.copy(); neighbor[:, 0] = False
    masked = np.where(neighbor, distance, np.nan)
    nearest = np.full(len(positions), np.nan); mean_neighbor = np.full(len(positions), np.nan)
    has_neighbor = neighbor.any(axis=1)
    nearest[has_neighbor] = np.nanmin(masked[has_neighbor], axis=1)
    mean_neighbor[has_neighbor] = np.nanmean(masked[has_neighbor], axis=1)
    return {
        "mean_step_speed": speed.mean(axis=1), "median_step_speed": np.median(speed, axis=1),
        "last_step_speed": speed[:, -1], "obs_displacement": chord_norm,
        "mean_acceleration": np.linalg.norm(acceleration, axis=-1).mean(axis=1),
        "mean_turn_angle": turns.mean(axis=1),
        "observation_nonlinearity": np.mean(cross / np.maximum(chord_norm[:, None], 1.0e-6), axis=1),
        "valid_agent_count": num_valid.astype(np.float64), "nearest_neighbor_distance": nearest,
        "neighbor_count_r2": ((distance < 2.0) & neighbor).sum(axis=1).astype(float),
        "neighbor_count_r4": ((distance < 4.0) & neighbor).sum(axis=1).astype(float),
        "neighbor_count_r6": ((distance < 6.0) & neighbor).sum(axis=1).astype(float),
        "mean_neighbor_distance": mean_neighbor,
    }


def build_model(fold: str, seed: int, cfg, FrontEncoderASTRA, device: torch.device):
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    module = load_module(f"exp168c_decoder_{fold}_{seed}", DECODER)
    module.replace_decoder(model, cfg, "hybrid_k20_10_10")
    checkpoint = run_dir(fold, seed) / "best_selected.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def collect_fold(fold: str, seed: int, cfg, FrontEncoderASTRA, TrajectoryDataset,
                 normalize_to_focal_heading, device: torch.device, batch_size: int,
                 max_samples: int) -> list[dict]:
    base = TrajectoryDataset(DATA / f"{fold}_data_train.npy", DATA / f"{fold}_num_train.npy", cfg)
    split = np.load(run_dir(fold, seed) / "split_indices.npz")
    indices = np.asarray(split["validation"], dtype=np.int64)
    if max_samples > 0: indices = indices[:max_samples]
    loader = DataLoader(Subset(base, indices.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(fold, seed, cfg, FrontEncoderASTRA, device)
    rows: list[dict] = []; cursor = 0
    with torch.inference_mode():
        for past, future, num_valid, _, _, _ in loader:
            past = past.to(device); future = future.to(device); num_valid = num_valid.to(device).int()
            past, future = normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            prediction = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)[2]
            target = future - center
            candidates = prediction[:, 0]
            truth = target[:, 0]
            error = torch.linalg.vector_norm(candidates - truth[:, None], dim=-1)
            quality = error.mean(dim=-1) + 0.25 * error[..., -1]
            structured_quality = quality[:, :10].min(dim=-1).values
            free_quality = quality[:, 10:].min(dim=-1).values
            observed = observation_features(past.cpu().numpy(), num_valid.cpu().numpy())
            for local in range(len(past)):
                row = {
                    "outer_fold": fold, "seed": seed, "validation_index": int(indices[cursor + local]),
                    "structured_quality": float(structured_quality[local].cpu()),
                    "free_quality": float(free_quality[local].cpu()),
                    "structured_needed": int(structured_quality[local] < free_quality[local]),
                    "quality_margin_free_minus_structured": float((free_quality[local] - structured_quality[local]).cpu()),
                }
                for key, values in observed.items(): row[key] = float(values[local])
                rows.append(row)
            cursor += len(past)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return rows


def make_pipeline(feature_count: int) -> Pipeline:
    numeric = list(range(feature_count))
    transformer = ColumnTransformer([("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
    ]), numeric)])
    return Pipeline([
        ("features", transformer),
        ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=168)),
    ])


def cross_validate(rows: list[dict], features: tuple[str, ...], feature_set: str,
                   splits: int) -> tuple[dict, list[dict], list[dict]]:
    x = np.asarray([[row[key] for key in features] for row in rows], dtype=np.float64)
    y = np.asarray([row["structured_needed"] for row in rows], dtype=np.int64)
    structured = np.asarray([row["structured_quality"] for row in rows], dtype=np.float64)
    free = np.asarray([row["free_quality"] for row in rows], dtype=np.float64)
    if len(np.unique(y)) != 2: raise RuntimeError("Family target contains only one class")
    effective_splits = min(splits, int(np.bincount(y).min()))
    if effective_splits < 2: raise RuntimeError("Insufficient minority samples for stratified CV")
    cv = StratifiedKFold(effective_splits, shuffle=True, random_state=20260831)
    probabilities = np.zeros(len(rows), dtype=np.float64)
    predictions = np.zeros(len(rows), dtype=np.int64)
    constant_predictions = np.zeros(len(rows), dtype=np.int64)
    coefficients = []
    split_ids = np.zeros(len(rows), dtype=np.int32)
    for split_id, (train, test) in enumerate(cv.split(x, y), start=1):
        pipeline = make_pipeline(len(features)); pipeline.fit(x[train], y[train])
        probabilities[test] = pipeline.predict_proba(x[test])[:, 1]
        predictions[test] = (probabilities[test] >= 0.5).astype(np.int64)
        constant_family = int(structured[train].mean() < free[train].mean())
        constant_predictions[test] = constant_family; split_ids[test] = split_id
        classifier = pipeline.named_steps["classifier"]
        for feature, value in zip(features, classifier.coef_[0]):
            coefficients.append({"feature_set": feature_set, "split": split_id, "feature": feature, "coefficient": float(value)})
    selected = np.where(predictions == 1, structured, free)
    constant = np.where(constant_predictions == 1, structured, free)
    oracle = np.minimum(structured, free)
    constant_mean = float(constant.mean()); selected_mean = float(selected.mean()); oracle_mean = float(oracle.mean())
    denominator = max(constant_mean - oracle_mean, 1.0e-12)
    metrics = {
        "feature_set": feature_set, "samples": len(rows),
        "structured_prevalence": float(y.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "constant_quality": constant_mean, "predicted_quality": selected_mean,
        "oracle_family_quality": oracle_mean,
        "quality_gain_vs_constant_pct": 100.0 * (constant_mean - selected_mean) / constant_mean,
        "oracle_regret_closed_pct": 100.0 * (constant_mean - selected_mean) / denominator,
    }
    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append({
            "outer_fold": row["outer_fold"], "validation_index": row["validation_index"],
            "feature_set": feature_set, "cv_split": int(split_ids[index]),
            "target_structured": int(y[index]), "probability_structured": float(probabilities[index]),
            "predicted_structured": int(predictions[index]), "constant_structured": int(constant_predictions[index]),
            "selected_quality": float(selected[index]), "constant_quality": float(constant[index]),
            "oracle_family_quality": float(oracle[index]),
        })
    return metrics, prediction_rows, coefficients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    required = [CONFIG, TRAINER, DECODER]
    for fold in folds:
        required += [DATA / f"{fold}_data_train.npy", DATA / f"{fold}_num_train.npy",
                     run_dir(fold, args.seed) / "best_selected.pth", run_dir(fold, args.seed) / "split_indices.npz"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    if args.dry_run:
        print(json.dumps({"folds": folds, "seed": args.seed, "cv_splits": args.cv_splits,
                          "max_samples": args.max_samples, "output": str(HERE)}, indent=2))
        print("[done] dry-run passed"); return
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
    all_rows: list[dict] = []
    for fold in folds:
        print(f"[collect] {fold}", flush=True)
        rows = collect_fold(fold, args.seed, cfg, FrontEncoderASTRA, TrajectoryDataset,
                            normalize_to_focal_heading, device, args.batch_size, args.max_samples)
        all_rows += rows; print(f"[done] {fold}: {len(rows)} inner-validation samples", flush=True)
    metrics_rows = []; prediction_rows = []; coefficient_rows = []
    for fold in folds:
        group = [row for row in all_rows if row["outer_fold"] == fold]
        for feature_set, features in (("speed", SPEED_FEATURES), ("full", FULL_FEATURES)):
            metrics, predictions, coefficients = cross_validate(group, features, feature_set, args.cv_splits)
            metrics = {"outer_fold": fold, **metrics}
            metrics_rows.append(metrics); prediction_rows += predictions
            coefficient_rows += [{"outer_fold": fold, **row} for row in coefficients]
            print("[cv] " + json.dumps(metrics), flush=True)
    complete = set(folds) == set(FOLDS) and args.max_samples == 0
    full = [row for row in metrics_rows if row["feature_set"] == "full"]
    macro = {
        key: float(np.mean([row[key] for row in full]))
        for key in ("balanced_accuracy", "roc_auc", "quality_gain_vs_constant_pct", "oracle_regret_closed_pct")
    } if full else None
    gate = None
    if complete and macro:
        gate = {
            "macro_balanced_accuracy_ge_0p60": macro["balanced_accuracy"] >= 0.60,
            "all_fold_balanced_accuracy_ge_0p55": all(row["balanced_accuracy"] >= 0.55 for row in full),
            "macro_roc_auc_ge_0p65": macro["roc_auc"] >= 0.65,
            "macro_quality_gain_vs_constant_ge_3pct": macro["quality_gain_vs_constant_pct"] >= 3.0,
        }
        gate["conditional_allocator_allowed"] = all(gate.values())
    tag = "smoke" if args.max_samples > 0 else "full"
    HERE.mkdir(parents=True, exist_ok=True)
    write_csv(HERE / f"validation_family_samples_{tag}.csv", all_rows)
    write_csv(HERE / f"per_fold_predictability_{tag}.csv", metrics_rows)
    write_csv(HERE / f"crossval_predictions_{tag}.csv", prediction_rows)
    write_csv(HERE / f"feature_coefficients_{tag}.csv", coefficient_rows)
    payload = {
        "experiment": "EXP-168C Family Need Predictability Audit", "complete": complete,
        "trained_new_trajectory_model": False, "outer_test_used": False,
        "target": "family containing minimum ADE + 0.25*FDE candidate",
        "metrics": metrics_rows, "full_feature_macro": macro, "gate": gate,
    }
    (HERE / f"summary_{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"complete": complete, "macro": macro, "gate": gate}, indent=2), flush=True)
    print(f"[done] {HERE}", flush=True)


if __name__ == "__main__": main()

