"""Zero-training, same-mode attribution of H3 coarse vs V2 corrected trajectories."""
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
OUTPUT = ROOT / "H3_V2_CORRECTION_ATTRIBUTION_AUDIT"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
ANCHORS = ((4, 3), (8, 7), (12, 11))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gather(values: torch.Tensor, indices: torch.Tensor):
    rows = torch.arange(len(values), device=values.device)
    return values[rows, indices]


def metrics(prefix: str, ade: torch.Tensor, fde: torch.Tensor, logits: torch.Tensor):
    rows = torch.arange(len(ade), device=ade.device)
    ade_winner = ade.argmin(-1)
    fde_winner = fde.argmin(-1)
    joint = ade + 0.5 * fde
    joint_winner = joint.argmin(-1)
    top1 = logits.argmax(-1)
    return {
        f"{prefix}_minADE": ade.min(-1).values,
        f"{prefix}_minFDE": fde.min(-1).values,
        f"{prefix}_evalJ": ade.min(-1).values + 0.5 * fde.min(-1).values,
        f"{prefix}_sameFDE": fde[rows, ade_winner],
        f"{prefix}_minJoint": joint.min(-1).values,
        f"{prefix}_ade_winner": ade_winner,
        f"{prefix}_fde_winner": fde_winner,
        f"{prefix}_joint_winner": joint_winner,
        f"{prefix}_top1_ADE": ade[rows, top1],
        f"{prefix}_top1_FDE": fde[rows, top1],
        f"{prefix}_top1_J": joint[rows, top1],
        f"{prefix}_top1_hits_ade_winner": (top1 == ade_winner).float(),
        f"{prefix}_ade_winner_score_rank": (logits > logits[rows, ade_winner, None]).sum(-1).float() + 1.0,
    }


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
        corrected = prediction[:, 0] + center[:, 0, 0][:, None, None]
        coarse = model.decoder.last_coarse_trajectories[:, 0] + center[:, 0, 0][:, None, None]
        logits = model.last_mode_logits[:, 0]
        target = future[:, 0]
        coordinate_offset = center[:, 0, 0][:, None, None]
        source_anchors = model.decoder.last_anchor_proposals[:, 0] + coordinate_offset
        corrected_anchors = model.decoder.last_corrected_anchors[:, 0] + coordinate_offset
        coarse_at_anchors = torch.stack([coarse[:, :, index] for _, index in ANCHORS], dim=2)
        corrected_at_anchors = torch.stack([corrected[:, :, index] for _, index in ANCHORS], dim=2)
        source_error_check = float((coarse_at_anchors - source_anchors).abs().max().item())
        corrected_error_check = float((corrected_at_anchors - corrected_anchors).abs().max().item())
        if source_error_check > 2e-5 or corrected_error_check > 2e-5:
            raise RuntimeError(
                "Anchor/trajectory coordinate mismatch: "
                f"source={source_error_check}, corrected={corrected_error_check}"
            )

        source_error = torch.linalg.vector_norm(coarse - target[:, None], dim=-1)
        corrected_error = torch.linalg.vector_norm(corrected - target[:, None], dim=-1)
        source_ade, source_fde = source_error.mean(-1), source_error[..., -1]
        corrected_ade, corrected_fde = corrected_error.mean(-1), corrected_error[..., -1]
        source_stats = metrics("source", source_ade, source_fde, logits)
        corrected_stats = metrics("corrected", corrected_ade, corrected_fde, logits)
        source_aw = source_stats["source_ade_winner"].long()
        corrected_aw = corrected_stats["corrected_ade_winner"].long()
        source_fw = source_stats["source_fde_winner"].long()
        corrected_fw = corrected_stats["corrected_fde_winner"].long()

        values: dict[str, torch.Tensor | np.ndarray] = {
            "sample_id": np.asarray(sample_ids[offset: offset + batch], dtype=np.int64),
            "past": past.cpu().numpy(),
            **source_stats,
            **corrected_stats,
            "delta_minADE": source_stats["source_minADE"] - corrected_stats["corrected_minADE"],
            "delta_minFDE": source_stats["source_minFDE"] - corrected_stats["corrected_minFDE"],
            "delta_evalJ": source_stats["source_evalJ"] - corrected_stats["corrected_evalJ"],
            "delta_sameFDE": source_stats["source_sameFDE"] - corrected_stats["corrected_sameFDE"],
            "delta_minJoint": source_stats["source_minJoint"] - corrected_stats["corrected_minJoint"],
            "delta_top1_ADE": source_stats["source_top1_ADE"] - corrected_stats["corrected_top1_ADE"],
            "delta_top1_FDE": source_stats["source_top1_FDE"] - corrected_stats["corrected_top1_FDE"],
            "ade_winner_changed": (source_aw != corrected_aw).float(),
            "fde_winner_changed": (source_fw != corrected_fw).float(),
            "source_ade_winner_same_mode_gain": gather(source_ade, source_aw) - gather(corrected_ade, source_aw),
            "corrected_ade_winner_same_mode_gain": gather(source_ade, corrected_aw) - gather(corrected_ade, corrected_aw),
            "source_fde_winner_same_mode_gain": gather(source_fde, source_fw) - gather(corrected_fde, source_fw),
            "corrected_fde_winner_same_mode_gain": gather(source_fde, corrected_fw) - gather(corrected_fde, corrected_fw),
            "mode_ADE_improved_fraction": (corrected_ade < source_ade).float().mean(-1),
            "mode_FDE_improved_fraction": (corrected_fde < source_fde).float().mean(-1),
            "corrected_ade_winner_rank_change": source_stats["source_ade_winner_score_rank"] - corrected_stats["corrected_ade_winner_score_rank"],
        }
        for step, index in ANCHORS:
            source_point = source_error[..., index]
            corrected_point = corrected_error[..., index]
            source_winner = source_point.argmin(-1)
            corrected_winner = corrected_point.argmin(-1)
            values.update({
                f"source_point_oracle_t{step}": source_point.min(-1).values,
                f"corrected_point_oracle_t{step}": corrected_point.min(-1).values,
                f"delta_point_oracle_t{step}": source_point.min(-1).values - corrected_point.min(-1).values,
                f"point_winner_changed_t{step}": (source_winner != corrected_winner).float(),
                f"source_point_winner_same_mode_gain_t{step}": gather(source_point, source_winner) - gather(corrected_point, source_winner),
                f"corrected_point_winner_same_mode_gain_t{step}": gather(source_point, corrected_winner) - gather(corrected_point, corrected_winner),
                f"mode_point_improved_fraction_t{step}": (corrected_point < source_point).float().mean(-1),
            })

        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            parts.setdefault(key, []).append(value)
        offset += batch
    result = {key: np.concatenate(value) for key, value in parts.items()}
    result.update(helper.observation_features(result.pop("past")))
    return result


SUMMARY_METRICS = (
    "source_minADE", "corrected_minADE", "delta_minADE",
    "source_minFDE", "corrected_minFDE", "delta_minFDE",
    "source_evalJ", "corrected_evalJ", "delta_evalJ",
    "source_sameFDE", "corrected_sameFDE", "delta_sameFDE",
    "source_minJoint", "corrected_minJoint", "delta_minJoint",
    "source_top1_ADE", "corrected_top1_ADE", "delta_top1_ADE",
    "source_top1_FDE", "corrected_top1_FDE", "delta_top1_FDE",
    "source_top1_hits_ade_winner", "corrected_top1_hits_ade_winner",
    "source_ade_winner_score_rank", "corrected_ade_winner_score_rank",
    "ade_winner_changed", "fde_winner_changed",
    "source_ade_winner_same_mode_gain", "corrected_ade_winner_same_mode_gain",
    "source_fde_winner_same_mode_gain", "corrected_fde_winner_same_mode_gain",
    "mode_ADE_improved_fraction", "mode_FDE_improved_fraction",
    "delta_point_oracle_t4", "delta_point_oracle_t8", "delta_point_oracle_t12",
    "point_winner_changed_t4", "point_winner_changed_t8", "point_winner_changed_t12",
    "source_point_winner_same_mode_gain_t4", "source_point_winner_same_mode_gain_t8", "source_point_winner_same_mode_gain_t12",
    "corrected_point_winner_same_mode_gain_t4", "corrected_point_winner_same_mode_gain_t8", "corrected_point_winner_same_mode_gain_t12",
    "mode_point_improved_fraction_t4", "mode_point_improved_fraction_t8", "mode_point_improved_fraction_t12",
)


def condition_summary(fold, scope, data, threshold):
    high_acc = data["mean_acceleration"] >= threshold["mean_acceleration_q75"]
    high_nonlin = data["observation_nonlinearity"] >= threshold["observation_nonlinearity_q75"]
    conditions = {
        "ALL": np.ones(len(high_acc), dtype=bool),
        "high_acc": high_acc, "non_high_acc": ~high_acc,
        "high_nonlinearity": high_nonlin, "non_high_nonlinearity": ~high_nonlin,
    }
    rows = []
    for condition, mask in conditions.items():
        row = {"fold": fold, "scope": scope, "condition": condition, "samples": int(mask.sum())}
        for metric in SUMMARY_METRICS:
            row[metric] = float(data[metric][mask].mean()) if mask.any() else None
        for metric in ("minADE", "minFDE", "evalJ", "sameFDE", "minJoint"):
            source_value, corrected_value = row[f"source_{metric}"], row[f"corrected_{metric}"]
            row[f"corrected_improvement_{metric}_pct"] = 100.0 * (source_value / corrected_value - 1.0) if corrected_value else None
        rows.append(row)
    return rows


def verify_corrected(fold, outer, seed):
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected = next(row for row in payload["rows"] if row["fold"] == fold and int(row["seed"]) == seed)
    mapping = {"corrected_minADE": "minADE", "corrected_minFDE": "minFDE", "corrected_sameFDE": "same_path_FDE"}
    checks = {}
    for observed_name, expected_name in mapping.items():
        observed = float(outer[observed_name].mean())
        target = float(expected[expected_name])
        difference = observed - target
        checks[observed_name] = {"expected": target, "observed": observed, "absolute_difference": difference}
        if abs(difference) > 2e-5:
            raise RuntimeError(f"Official V2 reproduction failed: {fold}/{observed_name}: {difference}")
    return checks


def verify_source_inner(fold, inner, helper, seed):
    metrics_path = helper.run_dir(fold, "h3", seed) / "metrics.json"
    expected = json.loads(metrics_path.read_text(encoding="utf-8"))["evaluation_best_selected"]["oracle"]
    mapping = {"source_minADE": "min_ade_20", "source_minFDE": "min_fde_20", "source_sameFDE": "same_path_fde"}
    checks = {}
    for observed_name, expected_name in mapping.items():
        observed = float(inner[observed_name].mean())
        target = float(expected[expected_name])
        difference = observed - target
        checks[observed_name] = {"expected": target, "observed": observed, "absolute_difference": difference}
        if abs(difference) > 2e-5:
            raise RuntimeError(f"H3 source reproduction failed: {fold}/{observed_name}: {difference}")
    return checks


def decide(condition_rows):
    find = lambda fold, condition: next(row for row in condition_rows if row["fold"] == fold and row["scope"] == "outer" and row["condition"] == condition)
    high = [find(fold, "high_acc") for fold in FOLDS]
    low = [find(fold, "non_high_acc") for fold in FOLDS]
    high_j = [row["corrected_improvement_evalJ_pct"] for row in high]
    low_j = [row["corrected_improvement_evalJ_pct"] for row in low]
    macro_high, macro_low = float(np.mean(high_j)), float(np.mean(low_j))
    if macro_high <= 0 and macro_low > 0:
        case = "high_acc_specific_correction_harm"
    elif macro_high > 0 and macro_low > 0 and macro_high < 0.5 * macro_low:
        case = "high_acc_correction_underbenefit"
    elif macro_high <= 0 and macro_low <= 0:
        case = "correction_not_beneficial"
    else:
        case = "correction_broadly_beneficial"
    return {
        "case": case,
        "high_acc_corrected_improvement_evalJ_pct": dict(zip(FOLDS, high_j)),
        "non_high_acc_corrected_improvement_evalJ_pct": dict(zip(FOLDS, low_j)),
        "macro_high_acc_improvement_evalJ_pct": macro_high,
        "macro_non_high_acc_improvement_evalJ_pct": macro_low,
        "next_step": "Use same-mode and horizon attribution before choosing proposal-only or correction-only conditioning.",
    }


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
            helper.run_dir(fold, "h3", args.seed) / "metrics.json",
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
            "audit": "h3_to_v2_correction_attribution", "training": False,
            "folds": folds, "seed": args.seed, "output": str(args.output),
            "same_mode_comparison": True, "anchor_steps": [step for step, _ in ANCHORS],
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

    per_sample, condition_rows, thresholds, corrected_reproduction, source_reproduction = [], [], [], {}, {}
    for fold in folds:
        print(f"[correction-audit] {fold}", flush=True)
        split = np.load(helper.run_dir(fold, "v2", args.seed) / "split_indices.npz")
        train_indices = np.asarray(split["train"], dtype=np.int64)
        inner_indices = np.asarray(split["validation"], dtype=np.int64)
        threshold = helper.source_thresholds(fold, cfg, train_indices)
        thresholds.append({"fold": fold, **threshold})
        model = helper.build_model(fold, args.seed, cfg, FrontEncoderASTRA, device)
        collected = {}
        for scope in ("inner", "outer"):
            if scope == "inner":
                dataset = TrajectoryDataset(DATA / f"{fold}_data_train.npy", DATA / f"{fold}_num_train.npy", cfg)
                indices = inner_indices
            else:
                dataset = TrajectoryDataset(DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy", cfg)
                indices = np.arange(len(dataset), dtype=np.int64)
            if args.max_samples:
                indices = indices[: args.max_samples]
            loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)
            print(f"[correction-audit] {fold}/{scope} n={len(indices)}", flush=True)
            data = collect(model, loader, device, normalize_to_focal_heading, indices, helper)
            collected[scope] = data
            condition_rows += condition_summary(fold, scope, data, threshold)
            for i in range(len(indices)):
                row = {"fold": fold, "scope": scope}
                for name, values in data.items():
                    row[name] = int(values[i]) if name.endswith("winner") or name == "sample_id" else float(values[i])
                per_sample.append(row)
        if not args.max_samples:
            corrected_reproduction[fold] = verify_corrected(fold, collected["outer"], args.seed)
            source_reproduction[fold] = verify_source_inner(fold, collected["inner"], helper, args.seed)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    complete = set(folds) == set(FOLDS) and args.max_samples == 0
    decision = decide(condition_rows) if complete else None
    suffix = "" if complete else "_partial"
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / f"per_sample_attribution{suffix}.csv", per_sample)
    write_csv(args.output / f"condition_summary{suffix}.csv", condition_rows)
    write_csv(args.output / f"per_fold_summary{suffix}.csv", [row for row in condition_rows if row["scope"] == "outer" and row["condition"] in ("ALL", "high_acc", "non_high_acc")])
    write_csv(args.output / f"source_thresholds{suffix}.csv", thresholds)
    payload = {
        "audit": "h3_to_v2_correction_attribution", "complete": complete, "training": False,
        "model": "EXP-164 official F16+H3+V2 seed1",
        "corrected_v2_reproduction": corrected_reproduction,
        "source_h3_inner_reproduction": source_reproduction,
        "decision": decision, "condition_summary": condition_rows,
    }
    (args.output / f"summary{suffix}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
