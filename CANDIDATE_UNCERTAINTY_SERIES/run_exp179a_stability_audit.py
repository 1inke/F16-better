"""EXP-179A: zero-training candidate perturbation-stability audit."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from perturb_observation import PRIMARY_VARIANTS, early_step_drop, make_primary
from stability_metrics import FEATURES, derive, fit_score_calibrator, per_sample_spearman, score_control


SERIES = Path(__file__).resolve().parent
ROOT = SERIES.parent
OUTPUT = SERIES / "EXP_179A_PERTURBATION_STABILITY"
BASE_PATH = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "run_exp176a_final_score_audit.py"
PRIMARY_FEATURE = "trajectory_U"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def heading_frame(absolute_past: torch.Tensor):
    lag = min(3, absolute_past.shape[-2] - 1)
    displacement = absolute_past[:, 0, -1] - absolute_past[:, 0, -1 - lag]
    speed = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    default = torch.zeros_like(displacement); default[:, 0] = 1.0
    heading = torch.where(speed > 0.03, displacement / speed.clamp_min(1e-6), default)
    return heading[:, 0], heading[:, 1]


def local_points_to_world(local: torch.Tensor, absolute_past: torch.Tensor):
    cosine, sine = heading_frame(absolute_past)
    while cosine.ndim < local.ndim - 1:
        cosine = cosine.unsqueeze(-1); sine = sine.unsqueeze(-1)
    x = local[..., 0] * cosine - local[..., 1] * sine
    y = local[..., 0] * sine + local[..., 1] * cosine
    origin = absolute_past[:, 0, -1]
    while origin.ndim < local.ndim:
        origin = origin.unsqueeze(-2)
    return torch.stack((x, y), dim=-1) + origin


def local_vectors_to_world(local: torch.Tensor, absolute_past: torch.Tensor):
    cosine, sine = heading_frame(absolute_past)
    while cosine.ndim < local.ndim - 1:
        cosine = cosine.unsqueeze(-1); sine = sine.unsqueeze(-1)
    return torch.stack((local[..., 0] * cosine - local[..., 1] * sine,
                        local[..., 0] * sine + local[..., 1] * cosine), dim=-1)


@torch.inference_mode()
def forward_variant(model, trainer, absolute_past, absolute_future, num_valid):
    past, _ = trainer.normalize_to_focal_heading(absolute_past, absolute_future)
    center = past.mean(dim=-2, keepdim=True)
    output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
    raw = output[2] if isinstance(output, (tuple, list)) else output
    local_prediction = (raw + center.unsqueeze(-3))[:, 0]
    logits = model.last_mode_logits[:, 0]
    source = model.decoder.last_anchor_proposals[:, 0]
    corrected = model.decoder.last_corrected_anchors[:, 0]
    correction_world = local_vectors_to_world(corrected - source, absolute_past)
    prediction_world = local_points_to_world(local_prediction, absolute_past)
    endpoint = prediction_world[:, :, -1] - absolute_past[:, 0, -1, None]
    direction = torch.atan2(endpoint[..., 1], endpoint[..., 0])
    return (
        prediction_world.float().cpu().numpy(), logits.float().cpu().numpy(),
        correction_world.float().cpu().numpy(), direction.float().cpu().numpy(),
    )


def collect_fold(model, loader, trainer, device, fold_index: int):
    pieces = {"logits": [], "quality": [], **{name: [] for name in FEATURES}}
    sample_offset = 0
    for absolute_past, absolute_future, num_valid, *_ in loader:
        absolute_past = absolute_past.to(device); absolute_future = absolute_future.to(device)
        num_valid = num_valid.to(device).int()
        perturbations = make_primary(absolute_past, num_valid, 179000 + fold_index * 10000 + sample_offset)
        primary_predictions, primary_logits, primary_corrections, primary_directions = [], [], [], []
        for name in PRIMARY_VARIANTS:
            prediction, logits, correction, direction = forward_variant(
                model, trainer, perturbations[name], absolute_future, num_valid
            )
            primary_predictions.append(prediction); primary_logits.append(logits)
            primary_corrections.append(correction); primary_directions.append(direction)
        p2_past = early_step_drop(absolute_past, num_valid)
        p2_prediction, _, _, p2_direction = forward_variant(model, trainer, p2_past, absolute_future, num_valid)
        primary_predictions = np.stack(primary_predictions); primary_logits = np.stack(primary_logits)
        features = derive(
            primary_predictions, primary_logits, np.stack(primary_corrections),
            np.stack(primary_directions), p2_prediction, p2_direction,
        )
        target = absolute_future[:, 0].float().cpu().numpy()
        distance = np.linalg.norm(primary_predictions[0] - target[:, None], axis=-1)
        quality = distance.mean(axis=-1) + 0.25 * distance[..., -1]
        pieces["logits"].append(primary_logits[0]); pieces["quality"].append(quality)
        for name, values in features.items():
            pieces[name].append(values)
        sample_offset += len(absolute_past)
        print(f"[collect] samples={sample_offset}", flush=True)
    return {name: np.concatenate(values) for name, values in pieces.items()}


def fold_analysis(fold: str, bundle, other_bundles):
    calibrators = [fit_score_calibrator(x["logits"], x["quality"]) for x in other_bundles]
    predicted = score_control(bundle["logits"], calibrators)
    residual = bundle["quality"] - bundle["quality"].mean(axis=1, keepdims=True) - predicted
    correlation_rows, quantile_rows = [], []
    for name in FEATURES:
        values = bundle[name]; corr, valid = per_sample_spearman(values, residual)
        order = np.argsort(values, axis=1, kind="mergesort")
        rows = np.arange(len(values))[:, None]
        low_index, high_index = order[:, :5], order[:, -5:]
        q1 = float(bundle["quality"][rows, low_index].mean())
        q4 = float(bundle["quality"][rows, high_index].mean())
        correlation_rows.append({
            "fold": fold, "feature": name, "samples": len(values), "valid_samples": int(valid.sum()),
            "spearman_mean": float(np.nanmean(corr)) if valid.any() else None,
            "spearman_median": float(np.nanmedian(corr)) if valid.any() else None,
            "within_sample_std": float(values.std(axis=1).mean()),
        })
        quantile_rows.append({
            "fold": fold, "feature": name, "q1_quality": q1, "q4_quality": q4,
            "q4_vs_q1_relative_gap": float((q4 - q1) / max(abs(q1), 1e-8)),
        })
    uncertainty = bundle[PRIMARY_FEATURE]; top5 = np.argsort(-bundle["logits"], axis=1)[:, :5]
    top5_q = np.take_along_axis(bundle["quality"], top5, axis=1)
    winner_position = top5_q.argmin(axis=1); row = np.arange(len(top5))
    top5_u = np.take_along_axis(uncertainty, top5, axis=1)
    winner_u = top5_u[row, winner_position]
    nonwinner_u = (top5_u.sum(axis=1) - winner_u) / 4.0
    discrimination = {
        "fold": fold, "samples": len(bundle["quality"]),
        "winner_uncertainty": float(winner_u.mean()), "nonwinner_uncertainty": float(nonwinner_u.mean()),
        "nonwinner_minus_winner": float((nonwinner_u - winner_u).mean()),
        "winner_lower_fraction": float((winner_u < nonwinner_u).mean()),
    }
    return correlation_rows, quantile_rows, discrimination


def summarize(folds, correlations, quantiles, discrimination):
    features = []
    for name in FEATURES:
        rows = [x for x in correlations if x["feature"] == name and x["spearman_mean"] is not None]
        values = [x["spearman_mean"] for x in rows]; positive = sum(x > 0 for x in values); negative = sum(x < 0 for x in values)
        features.append({
            "feature": name, "macro_spearman": float(np.mean(values)) if values else None,
            "abs_macro_spearman": float(abs(np.mean(values))) if values else None,
            "positive_folds": positive, "negative_folds": negative,
            "stable_sign_folds": max(positive, negative),
            "cross_scene_gate": bool(values and abs(np.mean(values)) >= 0.15 and max(positive, negative) >= 4),
        })
    primary = next(x for x in features if x["feature"] == PRIMARY_FEATURE)
    qrows = [x for x in quantiles if x["feature"] == PRIMARY_FEATURE]
    q1 = np.mean([x["q1_quality"] for x in qrows]); q4 = np.mean([x["q4_quality"] for x in qrows])
    quantile_gap = float((q4 - q1) / max(abs(q1), 1e-8))
    quantile_positive_folds = sum(x["q4_vs_q1_relative_gap"] > 0 for x in qrows)
    quantile_gate = quantile_gap >= 0.10 and quantile_positive_folds >= 4
    discrimination_positive = sum(x["nonwinner_minus_winner"] > 0 for x in discrimination)
    discrimination_gap = float(np.mean([x["nonwinner_minus_winner"] for x in discrimination]))
    discrimination_gate = discrimination_gap > 0 and discrimination_positive >= 4
    passed = bool(primary["cross_scene_gate"] and quantile_gate and discrimination_gate and len(folds) == 5)
    gate = {
        "primary_feature": PRIMARY_FEATURE,
        "cross_scene_spearman_gate": primary["cross_scene_gate"],
        "macro_spearman": primary["macro_spearman"], "stable_sign_folds": primary["stable_sign_folds"],
        "q1_quality": float(q1), "q4_quality": float(q4), "q4_vs_q1_relative_gap": quantile_gap,
        "q1_q4_gap_at_least_10_percent": quantile_gate,
        "top5_nonwinner_minus_winner_uncertainty": discrimination_gap,
        "top5_positive_folds": discrimination_positive, "top5_discrimination_gate": discrimination_gate,
        "passed": passed, "decision": "PROCEED_TO_EXP179B" if passed else "STOP_UNCERTAINTY_SCORER",
    }
    return features, gate


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_analysis(path, payload):
    gate = payload["gate"]
    path.write_text(f"""# EXP-179A result analysis

## Decision

**{gate['decision']}** (passed: `{str(gate['passed']).lower()}`).

## Primary stability signal

- Macro Spearman: {gate['macro_spearman']:.4f}
- Stable-sign folds: {gate['stable_sign_folds']}/5
- Q4 vs Q1 true-quality gap: {100 * gate['q4_vs_q1_relative_gap']:.2f}%
- Top5 non-winner minus winner uncertainty: {gate['top5_nonwinner_minus_winner_uncertainty']:.6f}

The five EXP-173 F16 Joint-V2 checkpoints remain frozen. No model is trained in EXP-179A.
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="EXP-179A candidate perturbation-stability audit")
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--folds", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=1); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0); parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(); args.output_dir = args.output_dir.resolve()
    base = load_module("exp179_base", BASE_PATH)
    folds = base.FOLDS if args.folds == ["all"] else args.folds
    unknown = sorted(set(folds) - set(base.FOLDS))
    if unknown: raise ValueError(f"Unknown folds: {unknown}")
    missing = [str(base.ckpt(fold, args.seed, "v2")) for fold in folds if not base.ckpt(fold, args.seed, "v2").is_file()]
    if missing: raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    if args.preflight_only and args.max_samples == 0: args.max_samples = 8
    if not args.preflight_only and len(folds) < 5: raise ValueError("Formal gate requires all five folds")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    base.configure(device)
    trainer = base.load_module("exp179_trainer", base.TRAINER); models = base.load_module("exp179_models", base.MODEL_FILE)
    adapter = base.load_module("exp179_adapter", base.ADAPTER); bundles = {}
    for fold_index, fold in enumerate(folds):
        print(f"[fold] {fold} seed={args.seed}", flush=True)
        model, cfg = base.build(fold, args.seed, device, trainer, models, adapter)
        dataset = trainer.TrajectoryDataset(base.DATA / f"{fold}_data_test.npy", base.DATA / f"{fold}_num_test.npy", cfg)
        if args.max_samples > 0: dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        bundles[fold] = collect_fold(model, loader, trainer, device, fold_index)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    correlations, quantiles, discrimination = [], [], []
    for fold in folds:
        corr, quant, disc = fold_analysis(fold, bundles[fold], [bundles[x] for x in folds if x != fold])
        correlations.extend(corr); quantiles.extend(quant); discrimination.append(disc)
    feature_summary, gate = summarize(folds, correlations, quantiles, discrimination)
    payload = {
        "experiment": "EXP-179A", "training": False, "seed": args.seed, "folds": folds,
        "checkpoint": "EXP-173 final F16 + H3 + Joint-V2, fixed mode identity",
        "primary_perturbations": list(PRIMARY_VARIANTS),
        "p1": "symmetric Gaussian noise, sigma=0.01 * per-sample median observed step",
        "p2": "drop first observation and linearly realign 7 to 8 steps; supplementary only",
        "p3": "last two velocity magnitudes x0.98/x1.02 with headings preserved",
        "score_control": "source-fold cross-fitted 25-bin native-score calibration",
        "quality": "ADE + 0.25 * FDE", "feature_summary": feature_summary,
        "top5_discrimination": discrimination, "gate": gate,
    }
    output = args.output_dir / ("preflight" if args.preflight_only else "results")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "per_fold_seed.csv", correlations); write_csv(output / "uncertainty_quality.csv", feature_summary)
    write_csv(output / "quantile_analysis.csv", quantiles); write_csv(output / "top5_discrimination.csv", discrimination)
    write_analysis(output / "RESULT_ANALYSIS.md", payload)
    print(json.dumps({"status": "preflight_ok" if args.preflight_only else "complete", **gate}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
