"""Joint zero-training audit for EXP-158 candidate complementarity and EXP-160 score mismatch."""
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
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
CONFIG = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
K20_MODULE = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
K20_RESULTS = WORKSPACE / "FOLD_STAGE_ATTRIBUTION_SERIES" / "EXP_157B_MATCHED_K20" / "results"
ANCHOR_RESULTS = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "EXP_156B_FOLD_LOCAL_CURRENT_DECODER" / "results"
MODULES = {
    "h3": WORKSPACE / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py",
    "v2": WORKSPACE / "ANCHOR_CORRECTION_VALIDATION_SERIES" / "anchor_correction_validation_models.py",
    "m2": WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py",
}
DECODER_NAMES = {
    "k20": "kinematic_k20",
    "h3": "h3_anchor_proposal",
    "v2": "v2_wide_endpoint_correction",
    "m2": "m2_anchored_mode_template",
}
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
FAMILIES = ("k20", "h3", "v2", "m2")
STAGES = ("h3", "v2", "m2")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(fold: str, family: str, seed: int) -> Path:
    if family == "k20":
        root = K20_RESULTS / fold / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"
    else:
        root = ANCHOR_RESULTS / fold / family / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"
    return root / "best_selected.pth"


def preflight(folds: tuple[str, ...], seed: int) -> None:
    required = (PROJECT, DATA, CONFIG, TRAINER, K20_MODULE, *MODULES.values())
    missing = [str(path) for path in required if not path.exists()]
    missing += [str(checkpoint(fold, family, seed)) for fold in folds for family in FAMILIES if not checkpoint(fold, family, seed).is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def configure(device: torch.device):
    os.chdir(PROJECT)
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(WORKSPACE / "FRONT_ENCODER_SERIES")]
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20

    cfg = load_locked_q20(CONFIG)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    return cfg


def build_model(fold: str, family: str, seed: int, cfg, model_class, device: torch.device):
    source = checkpoint(fold, family, seed)
    model = model_class(cfg, "f16_temporal_social_cross")
    if family == "k20":
        module = load_module(f"exp158_k20_{fold}_{seed}", K20_MODULE)
    else:
        if family == "v2":
            os.environ["EXP143_SOURCE_CHECKPOINT"] = str(checkpoint(fold, "h3", seed))
        elif family == "m2":
            os.environ["EXP146_SOURCE_CHECKPOINT"] = str(checkpoint(fold, "v2", seed))
        module = load_module(f"exp158_{family}_{fold}_{seed}", MODULES[family])
    module.replace_decoder(model, cfg, DECODER_NAMES[family])
    model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def collect(model, loader, helper, device: torch.device, keep_context: bool):
    candidates, logits = [], []
    past_parts, future_parts, num_parts = [], [], []
    with torch.inference_mode():
        for past, future, num_valid, _, _, _ in loader:
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = helper.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            prediction = model(
                past - center, None, None, mode="testing",
                num_valid=num_valid, graph_loc=past,
            )[2] + center.unsqueeze(-3)
            candidates.append(prediction[:, 0].float().cpu().numpy())
            logits.append(model.last_mode_logits[:, 0].float().cpu().numpy())
            if keep_context:
                past_parts.append(past.float().cpu().numpy())
                future_parts.append(future[:, 0].float().cpu().numpy())
                num_parts.append(num_valid.cpu().numpy())
    result = {
        "candidates": np.concatenate(candidates),
        "logits": np.concatenate(logits),
    }
    if keep_context:
        result.update({
            "past": np.concatenate(past_parts),
            "future": np.concatenate(future_parts),
            "num_valid": np.concatenate(num_parts),
        })
    return result


def errors(candidates: np.ndarray, future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(candidates - future[:, None], axis=-1)
    return distance.mean(axis=-1), distance[..., -1]


def oracle_metrics(ade: np.ndarray, fde: np.ndarray) -> dict[str, float]:
    rows = np.arange(len(ade))
    best_ade = ade.argmin(axis=1)
    return {
        "minADE": float(ade.min(axis=1).mean()),
        "minFDE": float(fde.min(axis=1).mean()),
        "same_path_FDE": float(fde[rows, best_ade].mean()),
        "combined_cost": float((ade + 0.25 * fde).min(axis=1).mean()),
    }


def overlap_metrics(left: np.ndarray, right: np.ndarray, chunk: int = 256) -> dict[str, float]:
    endpoint_parts, trajectory_parts = [], []
    for start in range(0, len(left), chunk):
        a, b = left[start:start + chunk], right[start:start + chunk]
        endpoint_pair = np.linalg.norm(a[:, :, None, -1] - b[:, None, :, -1], axis=-1)
        trajectory_pair = np.linalg.norm(a[:, :, None] - b[:, None, :], axis=-1).mean(axis=-1)
        endpoint_parts.append(0.5 * (endpoint_pair.min(2).mean(1) + endpoint_pair.min(1).mean(1)))
        trajectory_parts.append(0.5 * (trajectory_pair.min(2).mean(1) + trajectory_pair.min(1).mean(1)))
    return {
        "symmetric_endpoint_NN": float(np.concatenate(endpoint_parts).mean()),
        "symmetric_trajectory_NN": float(np.concatenate(trajectory_parts).mean()),
    }


def motion_features(past: np.ndarray, future: np.ndarray, num_valid: np.ndarray) -> dict[str, np.ndarray]:
    focal = past[:, 0]
    velocity = np.diff(focal, axis=1)
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration = np.diff(velocity, axis=1)
    cross = velocity[:, :-1, 0] * velocity[:, 1:, 1] - velocity[:, :-1, 1] * velocity[:, 1:, 0]
    dot = (velocity[:, :-1] * velocity[:, 1:]).sum(axis=-1)
    turns = np.arctan2(cross, dot)
    last_positions = past[:, :, -1]
    focal_last = last_positions[:, :1]
    neighbour_distance = np.linalg.norm(last_positions - focal_last, axis=-1)
    valid = np.arange(past.shape[1])[None] < num_valid[:, None]
    local_density = ((neighbour_distance <= 2.0) & valid).sum(axis=1) - 1
    gt_relative = future - focal[:, -1, None]
    gt_velocity = np.diff(np.concatenate((focal[:, -1:, :], future), axis=1), axis=1)
    gt_cross = gt_velocity[:, :-1, 0] * gt_velocity[:, 1:, 1] - gt_velocity[:, :-1, 1] * gt_velocity[:, 1:, 0]
    gt_dot = (gt_velocity[:, :-1] * gt_velocity[:, 1:]).sum(axis=-1)
    gt_turn = np.arctan2(gt_cross, gt_dot)
    history_disp = np.linalg.norm(focal[:, -1] - focal[:, 0], axis=-1)
    return {
        "observed_speed": speed.mean(axis=1),
        "recent_speed": speed[:, -1],
        "history_displacement": history_disp,
        "observed_heading_change": np.abs(turns).sum(axis=1),
        "observed_curvature": np.abs(turns).mean(axis=1),
        "observed_acceleration": np.linalg.norm(acceleration, axis=-1).mean(axis=1),
        "neighbour_count": np.maximum(num_valid - 1, 0),
        "local_density_2m": local_density,
        "gt_endpoint_radius": np.linalg.norm(gt_relative[:, -1], axis=-1),
        "gt_heading_change": np.abs(gt_turn).sum(axis=1),
        "gt_curvature": np.abs(gt_turn).mean(axis=1),
        "gt_displacement_ratio": np.linalg.norm(gt_relative[:, -1], axis=-1) / np.maximum(history_disp, 1.0e-4),
    }


def rank_rows(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(len(values))[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.float64)[None]
    return ranks


def spearman_per_sample(score: np.ndarray, quality: np.ndarray) -> np.ndarray:
    x, y = rank_rows(score), rank_rows(-quality)
    x -= x.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    denominator = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    return (x * y).sum(axis=1) / np.maximum(denominator, 1.0e-12)


def score_audit(ade: np.ndarray, fde: np.ndarray, logits: np.ndarray) -> dict:
    rows = np.arange(len(ade))
    cost = ade + 0.25 * fde
    top = logits.argmax(axis=1)
    best_ade, best_fde, best_cost = ade.argmin(1), fde.argmin(1), cost.argmin(1)
    selected_cost = cost[rows, top]
    regret = selected_cost - cost[rows, best_cost]
    fde_regret = fde[rows, top] - fde[rows, best_fde]
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    entropy = -(probability * np.log(np.maximum(probability, 1.0e-12))).sum(axis=1)
    order = np.argsort(-logits, axis=1)
    recovery = {}
    for n in (1, 2, 3, 5, 10):
        chosen = order[:, :n]
        chosen_cost = np.take_along_axis(cost, chosen, axis=1)
        recovery[str(n)] = {
            "combined_winner_hit": float((chosen == best_cost[:, None]).any(axis=1).mean()),
            "best_combined_cost": float(chosen_cost.min(axis=1).mean()),
            "combined_regret": float((chosen_cost.min(axis=1) - cost[rows, best_cost]).mean()),
        }
    return {
        "score_quality_spearman_mean": float(spearman_per_sample(logits, cost).mean()),
        "score_quality_spearman_median": float(np.median(spearman_per_sample(logits, cost))),
        "best_ADE_hit": float((top == best_ade).mean()),
        "best_FDE_hit": float((top == best_fde).mean()),
        "combined_winner_hit": float((top == best_cost).mean()),
        "top1_ADE": float(ade[rows, top].mean()),
        "top1_FDE": float(fde[rows, top].mean()),
        "mean_regret": float(regret.mean()),
        "p75_regret": float(np.quantile(regret, 0.75)),
        "p90_regret": float(np.quantile(regret, 0.90)),
        "p95_regret": float(np.quantile(regret, 0.95)),
        "mean_FDE_regret": float(fde_regret.mean()),
        "score_entropy": float(entropy.mean()),
        "active_modes": int(np.unique(top).size),
        "top_mode_usage": np.bincount(top, minlength=logits.shape[1]).astype(int).tolist(),
        "recovery": recovery,
    }


def flatten(prefix: str, data: dict) -> dict:
    row = {}
    for key, value in data.items():
        if isinstance(value, dict):
            row.update(flatten(f"{prefix}{key}_", value))
        elif not isinstance(value, list):
            row[f"{prefix}{key}"] = value
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def family_metrics(bundle: dict, future: np.ndarray):
    family_errors, metrics = {}, {}
    for family in FAMILIES:
        ade, fde = errors(bundle[family]["candidates"], future)
        family_errors[family] = (ade, fde)
        metrics[family] = oracle_metrics(ade, fde)
    return family_errors, metrics


def macro(rows: list[dict], key_path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value = row
        for key in key_path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def exp158_decision(fold_rows: list[dict]) -> tuple[str, dict]:
    pairs = {}
    for anchor in STAGES:
        gains = {}
        for metric in ("minADE", "minFDE", "same_path_FDE"):
            k = macro(fold_rows, ("families", "k20", metric))
            a = macro(fold_rows, ("families", anchor, metric))
            u = macro(fold_rows, ("unions", f"k20+{anchor}", metric))
            gains[metric] = (min(k, a) - u) / max(min(k, a), 1.0e-12)
        pairs[anchor] = gains
    best = max(STAGES, key=lambda name: pairs[name]["minADE"] + pairs[name]["minFDE"])
    gain = pairs[best]
    if gain["minADE"] >= 0.02 and gain["minFDE"] >= 0.02:
        decision = "PASSED"
    elif max(gain["minADE"], gain["minFDE"]) >= 0.01:
        decision = "CONDITIONAL"
    else:
        decision = "REJECTED"
    return decision, {"pair_gains": pairs, "recommended_anchor_family": best}


def exp160_decision(fold_rows: list[dict]) -> tuple[str, dict]:
    corr = {stage: macro(fold_rows, ("score_audit", stage, "score_quality_spearman_mean")) for stage in STAGES}
    oracle = {stage: macro(fold_rows, ("families", stage, "combined_cost")) for stage in STAGES}
    regret = {stage: macro(fold_rows, ("score_audit", stage, "mean_regret")) for stage in STAGES}
    strict_pattern = corr["h3"] > corr["v2"] > corr["m2"] and oracle["m2"] <= oracle["h3"] and regret["m2"] > regret["h3"]
    partial_pattern = corr["m2"] < corr["h3"] and oracle["m2"] <= oracle["h3"]
    decision = "PASSED" if strict_pattern else "CONDITIONAL" if partial_pattern else "REJECTED"
    return decision, {"macro_correlation": corr, "macro_oracle_combined_cost": oracle, "macro_regret": regret}


def analysis_text(exp: int, decision: str, detail: dict) -> str:
    if exp == 158:
        best = detail["recommended_anchor_family"]
        gain = detail["pair_gains"][best]
        route = "进入 EXP-159 固定 K=20 候选族混合。" if decision == "PASSED" else (
            "互补性有限，先人工核查逐折 winner 与 Univ 分布，再决定是否进入 EXP-159。" if decision == "CONDITIONAL" else
            "停止候选族混合，转向 H3 anchor scale/heading/diversity 的受限修复。"
        )
        return (
            f"# EXP-158 Result Analysis\n\n## Decision: {decision}\n\n"
            f"宏平均诊断中最有互补性的组合为 K20 + {best.upper()}。相对两者中更好的单一候选族，"
            f"union minADE 增益为 {gain['minADE'] * 100:.2f}%，minFDE 增益为 {gain['minFDE'] * 100:.2f}%，"
            f"same-path FDE 增益为 {gain['same_path_FDE'] * 100:.2f}%。\n\n## Next step\n\n{route}\n"
        )
    corr, oracle, regret = detail["macro_correlation"], detail["macro_oracle_combined_cost"], detail["macro_regret"]
    route = "进入 EXP-161，只训练冻结候选上的小型 listwise scorer。" if decision == "PASSED" else (
        "评分失配证据不完整，先结合逐折结果判断是否仅对 V2 或 M2 做同步。" if decision == "CONDITIONAL" else
        "不进入 EXP-161；当前主要瓶颈不是逐阶段评分失配。"
    )
    return (
        f"# EXP-160 Result Analysis\n\n## Decision: {decision}\n\n"
        f"宏平均 Spearman(H3/V2/M2) = {corr['h3']:.4f}/{corr['v2']:.4f}/{corr['m2']:.4f}；"
        f"oracle combined cost = {oracle['h3']:.4f}/{oracle['v2']:.4f}/{oracle['m2']:.4f}；"
        f"mean regret = {regret['h3']:.4f}/{regret['v2']:.4f}/{regret['m2']:.4f}。\n\n## Next step\n\n{route}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=WORKSPACE)
    parser.add_argument("--no-save-candidates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    preflight(folds, args.seed)
    if args.dry_run:
        print(json.dumps({
            "status": "dry-run passed", "folds": folds, "seed": args.seed,
            "families": FAMILIES, "checkpoints": len(folds) * len(FAMILIES),
        }))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    cfg = configure(device)
    from front_encoder_models import FrontEncoderASTRA
    helper = load_module("exp158_training_helper", TRAINER)

    exp158 = args.output_root / "CANDIDATE_COMPLEMENTARITY_SERIES" / "EXP_158_CANDIDATE_COMPLEMENTARITY_AUDIT"
    exp160 = args.output_root / "SCORE_MISMATCH_AUDIT_SERIES" / "EXP_160_SCORE_TRAJECTORY_MISMATCH"
    per_sample_dir = exp158 / "per_sample"
    exp158.mkdir(parents=True, exist_ok=True)
    exp160.mkdir(parents=True, exist_ok=True)
    if not args.no_save_candidates:
        per_sample_dir.mkdir(parents=True, exist_ok=True)

    fold_rows, sample_rows, distribution_rows, recovery_rows = [], [], [], []
    for fold in folds:
        dataset = helper.TrajectoryDataset(DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy", cfg)
        if args.max_samples:
            dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        bundle = {}
        context = None
        for family in FAMILIES:
            print(f"[collect] fold={fold} family={family}", flush=True)
            model = build_model(fold, family, args.seed, cfg, FrontEncoderASTRA, device)
            result = collect(model, loader, helper, device, keep_context=family == "k20")
            if family == "k20":
                context = {key: result.pop(key) for key in ("past", "future", "num_valid")}
            bundle[family] = result
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        assert context is not None
        future = context["future"]
        family_error, single = family_metrics(bundle, future)
        unions, overlaps = {}, {}
        for anchor in STAGES:
            ade = np.concatenate((family_error["k20"][0], family_error[anchor][0]), axis=1)
            fde = np.concatenate((family_error["k20"][1], family_error[anchor][1]), axis=1)
            unions[f"k20+{anchor}"] = oracle_metrics(ade, fde)
            overlaps[f"k20+{anchor}"] = overlap_metrics(bundle["k20"]["candidates"], bundle[anchor]["candidates"])
        audits = {stage: score_audit(*family_error[stage], bundle[stage]["logits"]) for stage in STAGES}

        winner_counts = {}
        features = motion_features(context["past"], future, context["num_valid"])
        for anchor in STAGES:
            ka, kf = family_error["k20"]
            aa, af = family_error[anchor]
            ade_winner = np.where(ka.min(1) <= aa.min(1), "k20", anchor)
            fde_winner = np.where(kf.min(1) <= af.min(1), "k20", anchor)
            winner_counts[f"k20+{anchor}"] = {
                "ADE": {"k20": float((ade_winner == "k20").mean()), anchor: float((ade_winner == anchor).mean())},
                "FDE": {"k20": float((fde_winner == "k20").mean()), anchor: float((fde_winner == anchor).mean())},
            }
            for family in ("k20", anchor):
                mask = ade_winner == family
                row = {"fold": fold, "pair": f"k20+{anchor}", "ADE_winner": family, "samples": int(mask.sum())}
                row.update({key: float(values[mask].mean()) if mask.any() else math.nan for key, values in features.items()})
                distribution_rows.append(row)

        fold_row = {
            "fold": fold, "seed": args.seed, "samples": len(future),
            "families": single, "unions": unions, "overlap": overlaps,
            "winner_family": winner_counts, "score_audit": audits,
        }
        fold_rows.append(fold_row)
        for stage in STAGES:
            for top_n, values in audits[stage]["recovery"].items():
                recovery_rows.append({"fold": fold, "stage": stage, "top_n": int(top_n), **values})
        for index in range(len(future)):
            row = {"fold": fold, "sample_id": index}
            row.update({key: float(value[index]) for key, value in features.items()})
            for family in FAMILIES:
                ade, fde = family_error[family]
                row[f"{family}_minADE"] = float(ade[index].min())
                row[f"{family}_minFDE"] = float(fde[index].min())
            sample_rows.append(row)
        if not args.no_save_candidates:
            np.savez_compressed(
                per_sample_dir / f"{fold}_seed_{args.seed}.npz",
                sample_id=np.arange(len(future)), past=context["past"], future=future,
                num_valid=context["num_valid"],
                **{f"{family}_candidates": bundle[family]["candidates"] for family in FAMILIES},
                **{f"{family}_logits": bundle[family]["logits"] for family in FAMILIES},
            )
        print(f"[done] fold={fold} samples={len(future)}", flush=True)

    decision158, detail158 = exp158_decision(fold_rows)
    decision160, detail160 = exp160_decision(fold_rows)
    summary158 = {"experiment": "EXP-158 Candidate Family Complementarity Audit", "protocol": "official ETH/UCY 8/12 at 2.5 Hz", "seed": args.seed, "folds": list(folds), "decision": decision158, **detail158, "per_fold": fold_rows}
    summary160 = {"experiment": "EXP-160 Score-Trajectory Mismatch Audit", "protocol": "official ETH/UCY 8/12 at 2.5 Hz", "seed": args.seed, "folds": list(folds), "decision": decision160, **detail160, "per_fold": [{"fold": row["fold"], "score_audit": row["score_audit"]} for row in fold_rows]}
    (exp158 / "summary.json").write_text(json.dumps(summary158, indent=2, ensure_ascii=False), encoding="utf-8")
    (exp160 / "summary.json").write_text(json.dumps(summary160, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(exp158 / "per_fold_summary.csv", [{"fold": row["fold"], "samples": row["samples"], **flatten("family_", row["families"]), **flatten("union_", row["unions"]), **flatten("overlap_", row["overlap"]), **flatten("winner_", row["winner_family"])} for row in fold_rows])
    write_csv(exp158 / "per_sample_metrics.csv", sample_rows)
    write_csv(exp158 / "winner_motion_distribution.csv", distribution_rows)
    write_csv(exp160 / "per_fold_score_audit.csv", [{"fold": row["fold"], **flatten("", row["score_audit"])} for row in fold_rows])
    write_csv(exp160 / "recovery.csv", recovery_rows)
    (exp158 / "RESULT_ANALYSIS.md").write_text(analysis_text(158, decision158, detail158), encoding="utf-8")
    (exp160 / "RESULT_ANALYSIS.md").write_text(analysis_text(160, decision160, detail160), encoding="utf-8")
    print(json.dumps({"EXP-158": decision158, "EXP-160": decision160, "output": str(args.output_root)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
