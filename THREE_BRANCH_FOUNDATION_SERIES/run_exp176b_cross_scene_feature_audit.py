"""EXP-176B: zero-training cross-scene candidate score-feature audit.

The audit never changes candidates or their native logits.  It asks whether
interpretable motion/social/decoder features explain candidate-quality residuals
after controlling the original score, consistently across held-out scenes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_exp176a_final_score_audit as base  # noqa: E402


DEFAULT_OUT = HERE / "EXP_176B_CROSS_SCENE_SCORE_FEATURE_AUDIT"
FEATURE_META = {
    "endpoint_speed_ratio": ("motion", True),
    "mean_speed": ("motion", True),
    "mean_speed_ratio": ("motion", True),
    "mean_acceleration": ("motion", True),
    "mean_jerk": ("motion", True),
    "heading_change": ("motion", True),
    "curvature_proxy": ("motion", True),
    "heading_difference": ("motion", True),
    "acceleration_mismatch": ("motion", True),
    "endpoint_displacement": ("motion", True),
    "min_social_distance": ("social", True),
    "mean_social_clearance": ("social", True),
    "collision_fraction": ("social", True),
    "ttc_proxy": ("social", True),
    "original_logit": ("native_score_control", False),
    "original_rank": ("native_score_control", False),
    "correction_norm_t4": ("decoder", True),
    "correction_norm_t8": ("decoder", True),
    "correction_norm_t12": ("decoder", True),
    "proposal_anchor_extent": ("decoder", True),
    "ensemble_deviation": ("decoder", True),
    "mode_identity": ("decoder_identity_diagnostic", False),
}


def angle_wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def candidate_features(
    prediction: torch.Tensor,
    logits: torch.Tensor,
    past: torch.Tensor,
    num_valid: torch.Tensor,
    source_anchors: torch.Tensor,
    corrected_anchors: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Return focal-agent features shaped [batch, modes]."""
    eps = 1e-6
    focal_past = past[:, 0]
    last = focal_past[:, -1]
    obs_velocity = focal_past[:, -1] - focal_past[:, -2]
    obs_speed = torch.linalg.vector_norm(obs_velocity, dim=-1).clamp_min(eps)
    full = torch.cat((last[:, None, None, :].expand(-1, prediction.shape[1], 1, -1), prediction), dim=2)
    velocity = torch.diff(full, dim=2)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    acceleration = torch.diff(velocity, dim=2)
    acceleration_norm = torch.linalg.vector_norm(acceleration, dim=-1)
    jerk = torch.diff(acceleration, dim=2)
    jerk_norm = torch.linalg.vector_norm(jerk, dim=-1)
    heading = torch.atan2(velocity[..., 1], velocity[..., 0])
    heading_delta = angle_wrap(torch.diff(heading, dim=2)).abs()
    observed_heading = torch.atan2(obs_velocity[:, 1], obs_velocity[:, 0])
    displacement = prediction[:, :, -1] - last[:, None]
    endpoint_heading = torch.atan2(displacement[..., 1], displacement[..., 0])

    batch, modes, steps = prediction.shape[:3]
    max_agents = past.shape[1]
    if max_agents > 1:
        other_last = past[:, 1:, -1]
        other_velocity = past[:, 1:, -1] - past[:, 1:, -2]
        timeline = torch.arange(1, steps + 1, dtype=past.dtype, device=past.device)
        other_future = other_last[:, :, None] + timeline[None, None, :, None] * other_velocity[:, :, None]
        distance = torch.linalg.vector_norm(prediction[:, :, None] - other_future[:, None], dim=-1)
        valid_other = torch.arange(max_agents - 1, device=past.device)[None] < (num_valid - 1).clamp_min(0)[:, None]
        distance = distance.masked_fill(~valid_other[:, None, :, None], float("inf"))
        min_time = distance.min(dim=2).values
        no_neighbour = ~valid_other.any(dim=1)
        min_social = min_time.min(dim=-1).values
        mean_clearance = min_time.clamp_max(10.0).mean(dim=-1)
        collision = ((distance < 0.20) & valid_other[:, None, :, None]).sum(dim=(2, 3)).float()
        denom = (valid_other.sum(dim=1).clamp_min(1) * steps).float()[:, None]
        collision_fraction = collision / denom
        near = min_time < 0.50
        time_index = torch.arange(1, steps + 1, device=past.device)[None, None].expand(batch, modes, -1)
        first_near = torch.where(near, time_index, torch.full_like(time_index, steps + 1)).min(dim=-1).values.float() / steps
        min_social[no_neighbour] = 10.0
        mean_clearance[no_neighbour] = 10.0
        first_near[no_neighbour] = float(steps + 1) / steps
    else:
        min_social = torch.full((batch, modes), 10.0, device=past.device)
        mean_clearance = min_social.clone()
        collision_fraction = torch.zeros_like(min_social)
        first_near = torch.full_like(min_social, float(steps + 1) / steps)

    native_order = torch.argsort(logits, dim=1, descending=True)
    native_rank = torch.empty_like(native_order, dtype=torch.float32)
    native_rank.scatter_(1, native_order, torch.arange(modes, device=logits.device, dtype=torch.float32)[None].expand(batch, -1))
    correction = torch.linalg.vector_norm(corrected_anchors - source_anchors, dim=-1)
    source_relative = source_anchors - last[:, None, None]
    proposal_extent = torch.linalg.vector_norm(source_relative, dim=-1).mean(dim=-1)
    ensemble_mean = prediction.mean(dim=1, keepdim=True)
    ensemble_deviation = torch.linalg.vector_norm(prediction - ensemble_mean, dim=-1).mean(dim=-1)

    result = {
        "endpoint_speed_ratio": speed[..., -1] / obs_speed[:, None],
        "mean_speed": speed.mean(dim=-1),
        "mean_speed_ratio": speed.mean(dim=-1) / obs_speed[:, None],
        "mean_acceleration": acceleration_norm.mean(dim=-1),
        "mean_jerk": jerk_norm.mean(dim=-1),
        "heading_change": heading_delta.mean(dim=-1),
        "curvature_proxy": heading_delta.mean(dim=-1) / speed.mean(dim=-1).clamp_min(eps),
        "heading_difference": angle_wrap(endpoint_heading - observed_heading[:, None]).abs(),
        "acceleration_mismatch": torch.linalg.vector_norm(velocity[..., 0, :] - obs_velocity[:, None], dim=-1),
        "endpoint_displacement": torch.linalg.vector_norm(displacement, dim=-1),
        "min_social_distance": min_social,
        "mean_social_clearance": mean_clearance,
        "collision_fraction": collision_fraction,
        "ttc_proxy": first_near,
        "original_logit": logits,
        "original_rank": native_rank / max(1, modes - 1),
        "correction_norm_t4": correction[..., 0],
        "correction_norm_t8": correction[..., 1],
        "correction_norm_t12": correction[..., 2],
        "proposal_anchor_extent": proposal_extent,
        "ensemble_deviation": ensemble_deviation,
        "mode_identity": torch.arange(modes, device=logits.device, dtype=torch.float32)[None].expand(batch, -1) / max(1, modes - 1),
    }
    return {name: value.detach().float().cpu().numpy() for name, value in result.items()}


def collect(model, loader, helper, device: torch.device):
    logits_parts, q_parts = [], []
    feature_parts = {name: [] for name in FEATURE_META}
    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            past, future, num_valid = past.to(device), future.to(device), num_valid.to(device).int()
            past, future = helper.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            raw_prediction = output[2] if isinstance(output, (tuple, list)) else output
            prediction = (raw_prediction + center.unsqueeze(-3))[:, 0]
            logits = model.last_mode_logits[:, 0]
            source = model.decoder.last_anchor_proposals[:, 0] + center[:, 0, None]
            corrected = model.decoder.last_corrected_anchors[:, 0] + center[:, 0, None]
            distance = torch.linalg.vector_norm(prediction - future[:, 0, None], dim=-1)
            q = distance.mean(dim=-1) + 0.25 * distance[..., -1]
            features = candidate_features(prediction, logits, past, num_valid, source, corrected)
            logits_parts.append(logits.float().cpu().numpy())
            q_parts.append(q.float().cpu().numpy())
            for name, value in features.items():
                feature_parts[name].append(value)
    return {
        "logits": np.concatenate(logits_parts),
        "quality": np.concatenate(q_parts),
        "features": {name: np.concatenate(parts) for name, parts in feature_parts.items()},
    }


def standardized_logits(logits: np.ndarray) -> np.ndarray:
    return (logits - logits.mean(axis=1, keepdims=True)) / np.maximum(logits.std(axis=1, keepdims=True), 1e-6)


def fit_score_calibrator(bundle: dict, bins: int = 25):
    score = standardized_logits(bundle["logits"]).reshape(-1)
    quality = bundle["quality"]
    target = (quality - quality.mean(axis=1, keepdims=True)).reshape(-1)
    edges = np.unique(np.quantile(score, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return np.array([-1.0, 1.0]), np.array([target.mean(), target.mean()])
    index = np.clip(np.digitize(score, edges[1:-1]), 0, len(edges) - 2)
    x, y = [], []
    for i in range(len(edges) - 1):
        mask = index == i
        if mask.any():
            x.append(float(score[mask].mean())); y.append(float(target[mask].mean()))
    return np.asarray(x), np.asarray(y)


def predict_score_control(logits: np.ndarray, calibrators: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    score = standardized_logits(logits)
    predictions = [np.interp(score, x, y, left=y[0], right=y[-1]) for x, y in calibrators]
    return np.mean(predictions, axis=0)


def per_sample_spearman(feature: np.ndarray, residual: np.ndarray):
    a = rankdata(feature, axis=1, method="average")
    b = rankdata(residual, axis=1, method="average")
    a -= a.mean(axis=1, keepdims=True); b -= b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    valid = denominator > 1e-12
    correlations = np.full(feature.shape[0], np.nan, dtype=np.float64)
    correlations[valid] = (a[valid] * b[valid]).sum(axis=1) / denominator[valid]
    return correlations, valid


def analyse_seed(seed: int, bundles: dict[str, dict]):
    rows = []
    for fold, bundle in bundles.items():
        source_calibrators = [fit_score_calibrator(other) for name, other in bundles.items() if name != fold]
        predicted = predict_score_control(bundle["logits"], source_calibrators)
        centered_quality = bundle["quality"] - bundle["quality"].mean(axis=1, keepdims=True)
        residual = centered_quality - predicted
        for name, feature in bundle["features"].items():
            corr, valid = per_sample_spearman(feature, residual)
            rows.append({
                "fold": fold, "seed": seed, "feature": name, "group": FEATURE_META[name][0],
                "eligible_for_gate": FEATURE_META[name][1], "samples": int(feature.shape[0]),
                "valid_samples": int(valid.sum()), "valid_fraction": float(valid.mean()),
                "spearman_mean": float(np.nanmean(corr)) if valid.any() else None,
                "spearman_median": float(np.nanmedian(corr)) if valid.any() else None,
                "feature_within_sample_std": float(feature.std(axis=1).mean()),
                "residual_abs_mean": float(np.abs(residual).mean()),
            })
    return rows


def consistency_table(rows: list[dict], folds: list[str], seeds: list[int]):
    result = []
    for name in FEATURE_META:
        feature_rows = [row for row in rows if row["feature"] == name and row["spearman_mean"] is not None]
        fold_values = {}
        for fold in folds:
            values = [row["spearman_mean"] for row in feature_rows if row["fold"] == fold]
            if values:
                fold_values[fold] = float(np.mean(values))
        signs = [np.sign(value) for value in fold_values.values() if abs(value) > 1e-9]
        positive, negative = signs.count(1), signs.count(-1)
        macro = float(np.mean(list(fold_values.values()))) if fold_values else None
        stable_count = max(positive, negative)
        eligible = FEATURE_META[name][1]
        passed = bool(eligible and macro is not None and abs(macro) >= 0.15 and stable_count >= 4 and len(fold_values) >= 4)
        result.append({
            "feature": name, "group": FEATURE_META[name][0], "eligible_for_gate": eligible,
            "macro_spearman": macro, "abs_macro_spearman": None if macro is None else abs(macro),
            "positive_folds": positive, "negative_folds": negative, "stable_sign_folds": stable_count,
            "available_folds": len(fold_values), "passed": passed,
            **{f"{fold}_spearman": fold_values.get(fold) for fold in folds},
        })
    return sorted(result, key=lambda row: -1.0 if row["abs_macro_spearman"] is None else -row["abs_macro_spearman"])


def main():
    parser = argparse.ArgumentParser(description="EXP-176B zero-training cross-scene score-feature audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    folds = base.FOLDS if args.folds == ["all"] else args.folds
    unknown = sorted(set(folds) - set(base.FOLDS))
    if unknown:
        raise ValueError(f"Unknown folds: {unknown}")
    missing = [str(base.ckpt(fold, seed, "v2")) for fold in folds for seed in args.seeds if not base.ckpt(fold, seed, "v2").is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    if args.dry_run:
        print(json.dumps({"status": "preflight_ok", "experiment": "EXP-176B", "training": False, "folds": folds, "seeds": args.seeds, "runs": len(folds) * len(args.seeds), "features": list(FEATURE_META)}, ensure_ascii=False, indent=2))
        return
    if len(folds) < 2:
        raise ValueError("At least two folds are required for cross-fitted original-score control")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    base.configure(device)
    trainer = base.load_module("exp176b_trainer", base.TRAINER)
    models = base.load_module("exp176b_models", base.MODEL_FILE)
    adapter = base.load_module("exp176b_adapter", base.ADAPTER)
    all_rows = []
    for seed in args.seeds:
        bundles = {}
        for fold in folds:
            print(f"[collect] fold={fold} seed={seed}", flush=True)
            model, cfg = base.build(fold, seed, device, trainer, models, adapter)
            dataset = trainer.TrajectoryDataset(base.DATA / f"{fold}_data_test.npy", base.DATA / f"{fold}_num_test.npy", cfg)
            if args.max_samples > 0:
                dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            bundles[fold] = collect(model, loader, trainer, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        all_rows.extend(analyse_seed(seed, bundles))
        del bundles
    consistency = consistency_table(all_rows, folds, args.seeds)
    passed = [row["feature"] for row in consistency if row["passed"]]
    gate = len(passed) >= 2 and len(folds) >= 5
    payload = {
        "experiment": "EXP-176B",
        "training": False,
        "candidate_checkpoint": "EXP173 F16 Joint-V2, frozen",
        "quality": "ADE + 0.25 * FDE",
        "score_control": "outer-fold cross-fitted 25-bin calibration of within-sample standardized native logits to within-sample centered quality",
        "correlation": "mean per-sample Spearman(feature, residual quality) across the 20 candidates",
        "social_proxy_note": "Constant-velocity extrapolation is used only to audit distances to other agents; it is not a decoder reference or candidate generator.",
        "gate_rule": "at least 2 eligible non-score features with |macro Spearman| >= 0.15 and same sign in >=4/5 folds",
        "passed_features": passed,
        "gate_pass": gate,
        "recommendation": "Proceed to EXP-176C bounded residual scorer" if gate else "Do not train EXP-176C; audited features do not meet the cross-scene gate",
        "feature_consistency": consistency,
        "per_fold_seed": all_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cross_scene_feature_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, rows in (("per_fold_seed_feature_correlations.csv", all_rows), ("feature_consistency.csv", consistency)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"gate_pass": gate, "passed_features": passed, "recommendation": payload["recommendation"]}, ensure_ascii=False, indent=2))
    print(f"[complete] {args.output_dir / 'cross_scene_feature_audit.json'}")


if __name__ == "__main__":
    main()
