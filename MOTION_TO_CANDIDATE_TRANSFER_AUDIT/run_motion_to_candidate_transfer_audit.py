"""Zero-training audit of motion-representation to V2 candidate-geometry transfer."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ASTRA_PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
if str(Path(sys.executable).resolve()).casefold() != str(ASTRA_PYTHON.resolve()).casefold():
    print(f"[bootstrap] relaunching with {ASTRA_PYTHON}", flush=True)
    raise SystemExit(subprocess.run([str(ASTRA_PYTHON), *sys.argv]).returncode)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
CONFIG = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
HELPER_SCRIPT = WORKSPACE / "INDIVIDUAL_MOTION_REPRESENTATION_AUDIT" / "run_motion_representation_audit.py"
BASELINE = WORKSPACE / "OFFICIAL_ETH_UCY_FINAL_3SEED_SERIES" / "EXP_164_FINAL_V2_OFFICIAL_3SEED" / "summary.json"
OUTPUT = ROOT / "MOTION_TO_CANDIDATE_TRANSFER_AUDIT"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
ANCHORS = ((4, 3, 0.20), (8, 7, 0.25), (12, 11, 0.50))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def heading_frame(past: torch.Tensor):
    velocity = past[:, 0, -1] - past[:, 0, -2]
    chord = past[:, 0, -1] - past[:, 0, 0]
    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    chord_norm = torch.linalg.vector_norm(chord, dim=-1, keepdim=True)
    fallback = chord / chord_norm.clamp_min(1e-6)
    default = torch.zeros_like(fallback)
    default[:, 0] = 1.0
    fallback = torch.where(chord_norm > 1e-5, fallback, default)
    heading = torch.where(speed > 1e-5, velocity / speed.clamp_min(1e-6), fallback)
    lateral = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    return heading, lateral


def project(relative: torch.Tensor, heading: torch.Tensor, lateral: torch.Tensor):
    # relative [B,K,2] or [B,2]
    while heading.ndim < relative.ndim:
        heading = heading.unsqueeze(1)
        lateral = lateral.unsqueeze(1)
    return (relative * heading).sum(-1), (relative * lateral).sum(-1)


def point_geometry(prefix, candidates, source, target, last, heading, lateral, correction, cap):
    # candidates/source: [B,K,2], target/last: [B,2]
    corrected_relative = candidates - last[:, None]
    source_relative = source - last[:, None]
    target_relative = target - last
    cand_long, cand_lat = project(corrected_relative, heading, lateral)
    source_long, source_lat = project(source_relative, heading, lateral)
    gt_long, gt_lat = project(target_relative, heading, lateral)
    radius = torch.linalg.vector_norm(corrected_relative, dim=-1)
    source_radius = torch.linalg.vector_norm(source_relative, dim=-1)
    corrected_distance = torch.linalg.vector_norm(candidates - target[:, None], dim=-1)
    source_distance = torch.linalg.vector_norm(source - target[:, None], dim=-1)
    correction_norm = torch.linalg.vector_norm(correction, dim=-1)
    long_min, long_max = cand_long.min(-1).values, cand_long.max(-1).values
    lat_min, lat_max = cand_lat.min(-1).values, cand_lat.max(-1).values
    source_long_min, source_long_max = source_long.min(-1).values, source_long.max(-1).values
    source_lat_min, source_lat_max = source_lat.min(-1).values, source_lat.max(-1).values
    corrected_inside = (gt_long >= long_min) & (gt_long <= long_max) & (gt_lat >= lat_min) & (gt_lat <= lat_max)
    source_inside = (gt_long >= source_long_min) & (gt_long <= source_long_max) & (gt_lat >= source_lat_min) & (gt_lat <= source_lat_max)
    result = {
        f"gt_radius_t{prefix}": torch.linalg.vector_norm(target_relative, dim=-1),
        f"gt_long_t{prefix}": gt_long,
        f"gt_abs_lat_t{prefix}": gt_lat.abs(),
        f"candidate_radius_mean_t{prefix}": radius.mean(-1),
        f"candidate_radius_max_t{prefix}": radius.max(-1).values,
        f"candidate_long_center_t{prefix}": cand_long.mean(-1),
        f"candidate_long_spread_t{prefix}": long_max - long_min,
        f"candidate_lat_spread_t{prefix}": lat_max - lat_min,
        f"source_radius_mean_t{prefix}": source_radius.mean(-1),
        f"source_long_spread_t{prefix}": source_long_max - source_long_min,
        f"source_lat_spread_t{prefix}": source_lat_max - source_lat_min,
        f"corrected_nearest_t{prefix}": corrected_distance.min(-1).values,
        f"source_nearest_t{prefix}": source_distance.min(-1).values,
        f"correction_oracle_gain_t{prefix}": source_distance.min(-1).values - corrected_distance.min(-1).values,
        f"corrected_outside_box_t{prefix}": (~corrected_inside).float(),
        f"source_outside_box_t{prefix}": (~source_inside).float(),
        f"correction_mean_norm_t{prefix}": correction_norm.mean(-1),
        f"correction_max_norm_t{prefix}": correction_norm.max(-1).values,
        f"correction_cap_hit_t{prefix}": (correction_norm >= 0.98 * cap).float().mean(-1),
    }
    return {key: value.cpu().numpy() for key, value in result.items()}


@torch.inference_mode()
def collect(model, loader, device, normalize, sample_ids, helper):
    parts: dict[str, list[np.ndarray]] = {}
    offset = 0
    for past, future, num, _, _, _ in loader:
        batch = len(past)
        past, future = past.to(device), future.to(device)
        num = num.to(device).int()
        past, future = normalize(past, future)
        center = past.mean(dim=-2, keepdim=True)
        prediction = model(past - center, None, None, mode="testing", num_valid=num, graph_loc=past)[2]
        prediction = prediction + center.unsqueeze(-3)
        candidates, target = prediction[:, 0], future[:, 0]
        source = model.decoder.last_anchor_proposals[:, 0]
        corrected = model.decoder.last_corrected_anchors[:, 0]
        coordinate_offset = center[:, 0, 0][:, None, None]
        source = source + coordinate_offset
        corrected = corrected + coordinate_offset
        correction = corrected - source
        reconstructed = torch.stack(
            [candidates[:, :, future_index] for _, future_index, _ in ANCHORS], dim=2
        )
        anchor_consistency_error = float((reconstructed - corrected).abs().max().item())
        if anchor_consistency_error > 2e-5:
            raise RuntimeError(
                f"Corrected-anchor coordinate mismatch: max error={anchor_consistency_error}"
            )
        last = past[:, 0, -1]
        heading, lateral = heading_frame(past)

        error = torch.linalg.vector_norm(candidates - target[:, None], dim=-1)
        ade, fde = error.mean(-1), error[..., -1]
        winner = ade.argmin(-1)
        rows = torch.arange(batch, device=device)
        values = {
            "sample_id": np.asarray(sample_ids[offset: offset + batch], dtype=np.int64),
            "past": past.cpu().numpy(),
            "minADE": ade.min(-1).values.cpu().numpy(),
            "minFDE": fde.min(-1).values.cpu().numpy(),
            "sameFDE": fde[rows, winner].cpu().numpy(),
            "J": (ade.min(-1).values + 0.5 * fde.min(-1).values).cpu().numpy(),
        }
        for anchor_index, (step, future_index, cap) in enumerate(ANCHORS):
            values.update(point_geometry(
                step, corrected[:, :, anchor_index], source[:, :, anchor_index],
                target[:, future_index], last, heading, lateral,
                correction[:, :, anchor_index], cap,
            ))
        for key, value in values.items():
            parts.setdefault(key, []).append(value)
        offset += batch
    result = {key: np.concatenate(value) for key, value in parts.items()}
    result.update(helper.observation_features(result.pop("past")))
    return result


def response_row(fold, scope, predictor, metric, data, helper):
    x = np.asarray(data[predictor], dtype=np.float64)
    y = np.asarray(data[metric], dtype=np.float64)
    variance = np.square(x - x.mean()).sum()
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / variance) if variance > 1e-12 else None
    return {
        "fold": fold, "scope": scope, "predictor": predictor, "geometry_metric": metric,
        "samples": len(x), "raw_slope": slope,
        "pearson": helper.pearson(x, y), "spearman": helper.spearman(x, y),
    }


def condition_summary(fold, scope, data, threshold):
    high_acc = data["mean_acceleration"] >= threshold["mean_acceleration_q75"]
    high_nonlin = data["observation_nonlinearity"] >= threshold["observation_nonlinearity_q75"]
    conditions = {
        "ALL": np.ones(len(high_acc), dtype=bool),
        "high_acc": high_acc, "non_high_acc": ~high_acc,
        "high_nonlinearity": high_nonlin, "non_high_nonlinearity": ~high_nonlin,
    }
    metrics = ["minADE", "minFDE", "J"]
    for step, _, _ in ANCHORS:
        metrics += [
            f"gt_radius_t{step}", f"gt_abs_lat_t{step}",
            f"candidate_radius_mean_t{step}", f"candidate_radius_max_t{step}",
            f"candidate_long_spread_t{step}", f"candidate_lat_spread_t{step}",
            f"corrected_nearest_t{step}", f"source_nearest_t{step}",
            f"correction_oracle_gain_t{step}", f"corrected_outside_box_t{step}",
            f"source_outside_box_t{step}", f"correction_mean_norm_t{step}",
            f"correction_cap_hit_t{step}",
        ]
    rows = []
    for condition, mask in conditions.items():
        row = {"fold": fold, "scope": scope, "condition": condition, "samples": int(mask.sum())}
        for metric in metrics:
            row[metric] = float(data[metric][mask].mean()) if mask.any() else None
        rows.append(row)
    return rows


def fold_summary(fold, data, threshold):
    high = data["mean_acceleration"] >= threshold["mean_acceleration_q75"]
    low = ~high
    row = {"fold": fold, "samples": len(high), "high_acc_samples": int(high.sum())}
    for step, _, _ in ANCHORS:
        for metric in (
            f"corrected_nearest_t{step}", f"corrected_outside_box_t{step}",
            f"correction_oracle_gain_t{step}", f"correction_mean_norm_t{step}",
            f"correction_cap_hit_t{step}", f"candidate_radius_mean_t{step}",
            f"candidate_long_spread_t{step}", f"candidate_lat_spread_t{step}",
            f"gt_radius_t{step}", f"gt_abs_lat_t{step}",
        ):
            high_mean = float(data[metric][high].mean())
            low_mean = float(data[metric][low].mean())
            row[f"high_acc_{metric}"] = high_mean
            row[f"non_high_acc_{metric}"] = low_mean
            row[f"high_vs_non_high_{metric}_pct"] = 100.0 * (high_mean / low_mean - 1.0) if abs(low_mean) > 1e-12 else None
    return row


def verify_reproduction(fold, outer, seed):
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected = next(row for row in payload["rows"] if row["fold"] == fold and int(row["seed"]) == seed)
    mapping = {"minADE": "minADE", "minFDE": "minFDE", "sameFDE": "same_path_FDE"}
    checks = {}
    for observed_name, expected_name in mapping.items():
        observed = float(outer[observed_name].mean())
        target = float(expected[expected_name])
        difference = observed - target
        checks[observed_name] = {"expected": target, "observed": observed, "absolute_difference": difference}
        if abs(difference) > 2e-5:
            raise RuntimeError(f"Official V2 reproduction failed: {fold}/{observed_name}: {difference}")
    return checks


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
    folds = FOLDS if "all" in args.folds else tuple(args.folds)

    sys.path.insert(0, str(HELPER_SCRIPT.parent))
    import run_motion_representation_audit as helper

    required = [CONFIG, HELPER_SCRIPT, BASELINE, helper.V2_MODULE]
    for fold in folds:
        required += [
            helper.run_dir(fold, "h3", args.seed) / "best_selected.pth",
            helper.run_dir(fold, "v2", args.seed) / "best_selected.pth",
            helper.run_dir(fold, "v2", args.seed) / "split_indices.npz",
            DATA / f"{fold}_data_train.npy", DATA / f"{fold}_num_train.npy",
            DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy",
        ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    if args.dry_run:
        print(json.dumps({
            "audit": "motion_to_candidate_transfer", "training": False,
            "folds": folds, "seed": args.seed, "output": str(args.output),
            "anchors": [step for step, _, _ in ANCHORS],
        }, indent=2))
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

    per_sample, conditions, responses, fold_rows, threshold_rows, reproduction = [], [], [], [], [], {}
    geometry_metrics = []
    for step, _, _ in ANCHORS:
        geometry_metrics += [
            f"candidate_radius_mean_t{step}", f"candidate_long_center_t{step}",
            f"candidate_long_spread_t{step}", f"candidate_lat_spread_t{step}",
            f"correction_mean_norm_t{step}", f"corrected_nearest_t{step}",
            f"gt_radius_t{step}", f"gt_abs_lat_t{step}",
        ]

    for fold in folds:
        print(f"[transfer-audit] {fold}", flush=True)
        split = np.load(helper.run_dir(fold, "v2", args.seed) / "split_indices.npz")
        train_indices = np.asarray(split["train"], dtype=np.int64)
        inner_indices = np.asarray(split["validation"], dtype=np.int64)
        threshold = helper.source_thresholds(fold, cfg, train_indices)
        threshold_rows.append({"fold": fold, **threshold})
        model = helper.build_model(fold, args.seed, cfg, FrontEncoderASTRA, device)
        datasets = {
            "inner": (TrajectoryDataset(DATA / f"{fold}_data_train.npy", DATA / f"{fold}_num_train.npy", cfg), inner_indices),
            "outer": (TrajectoryDataset(DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy", cfg), None),
        }
        collected = {}
        for scope in ("inner", "outer"):
            dataset, indices = datasets[scope]
            if indices is None:
                indices = np.arange(len(dataset), dtype=np.int64)
            if args.max_samples:
                indices = indices[: args.max_samples]
            loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)
            print(f"[transfer-audit] {fold}/{scope} n={len(indices)}", flush=True)
            data = collect(model, loader, device, normalize_to_focal_heading, indices, helper)
            collected[scope] = data
            conditions += condition_summary(fold, scope, data, threshold)
            for predictor in ("mean_acceleration", "observation_nonlinearity"):
                for metric in geometry_metrics:
                    responses.append(response_row(fold, scope, predictor, metric, data, helper))
            for i in range(len(indices)):
                row = {"fold": fold, "scope": scope}
                for name, values in data.items():
                    row[name] = int(values[i]) if name == "sample_id" else float(values[i])
                per_sample.append(row)
        outer = collected["outer"]
        fold_rows.append(fold_summary(fold, outer, threshold))
        if not args.max_samples:
            reproduction[fold] = verify_reproduction(fold, outer, args.seed)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    complete = set(folds) == set(FOLDS) and args.max_samples == 0
    suffix = "" if complete else "_partial"
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / f"per_sample_geometry{suffix}.csv", per_sample)
    write_csv(args.output / f"per_fold_summary{suffix}.csv", fold_rows)
    write_csv(args.output / f"condition_summary{suffix}.csv", conditions)
    write_csv(args.output / f"response_slope_summary{suffix}.csv", responses)
    write_csv(args.output / f"source_thresholds{suffix}.csv", threshold_rows)
    payload = {
        "audit": "motion_to_candidate_transfer", "complete": complete, "training": False,
        "model": "EXP-164 official F16+H3+V2 seed1",
        "full_model_reproduction": reproduction, "per_fold": fold_rows,
        "interpretation_guardrail": "This audit diagnoses association and envelope coverage; it does not prove acceleration causally creates the error.",
        "next_step": "Decide whether candidate geometry is under-responsive, mis-scaled across domains, or already adequate before training any module.",
    }
    (args.output / f"summary{suffix}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"complete": complete, "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
