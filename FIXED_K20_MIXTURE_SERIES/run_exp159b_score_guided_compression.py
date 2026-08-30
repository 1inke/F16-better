"""EXP-159B: score-guided, diversity-assisted fixed-K20 candidate compression."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
BASE_SCRIPT = HERE / "run_exp159_fixed_k20_mixture.py"
DEFAULT_INPUT = WORKSPACE / "SHARED_ENCODER_COMPLEMENTARITY_SERIES" / "EXP_158B_SHARED_ENCODER_COMPLEMENTARITY" / "per_sample"
DEFAULT_OUT = HERE / "EXP_159B_SCORE_GUIDED_DIVERSITY_COMPRESSION"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
DEFAULT_ALLOCATIONS = (0, 2, 4, 6, 8)
DEFAULT_WEIGHTS = (0.0, 0.05, 0.10, 0.20)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def standardized_score(k20_logits: np.ndarray, v2_logits: np.ndarray) -> np.ndarray:
    normalized = []
    for logits in (k20_logits, v2_logits):
        mean = logits.mean(axis=1, keepdims=True)
        std = logits.std(axis=1, keepdims=True).clip(min=0.05)
        normalized.append((logits - mean) / std)
    return np.concatenate(normalized, axis=1)


def select_candidates(
    score: np.ndarray,
    distance: np.ndarray,
    endpoint_distance: np.ndarray,
    trajectory_distance: np.ndarray,
    k20_count: int,
    diversity_weight: float,
    endpoint_duplicate: float,
    trajectory_duplicate: float,
) -> tuple[np.ndarray, int]:
    batch, modes = score.shape
    if modes != 40 or not 0 <= k20_count < 20:
        raise ValueError((modes, k20_count))
    v2_count = 20 - k20_count
    family_pools = {
        "k20": np.broadcast_to(np.arange(modes)[None] < 20, (batch, modes)).copy(),
        "v2": np.broadcast_to(np.arange(modes)[None] >= 20, (batch, modes)).copy(),
    }
    selected_mask = np.zeros((batch, modes), dtype=bool)
    duplicate_blocked = np.zeros((batch, modes), dtype=bool)
    duplicate = (endpoint_distance < endpoint_duplicate) & (trajectory_distance < trajectory_duplicate)
    upper = np.triu_indices(modes, k=1)
    diversity_scale = np.median(distance[:, upper[0], upper[1]], axis=1).clip(min=0.05)
    rows = np.arange(batch)
    min_distance = None
    sequence = []
    fallback_count = 0

    for family, quota in (("v2", v2_count), ("k20", k20_count)):
        pool = family_pools[family]
        for _ in range(quota):
            allowed = pool & ~selected_mask & ~duplicate_blocked
            fallback = pool & ~selected_mask
            empty = ~allowed.any(axis=1)
            fallback_count += int(empty.sum())
            allowed[empty] = fallback[empty]
            objective = score.copy()
            if min_distance is not None and diversity_weight:
                objective += float(diversity_weight) * min_distance / diversity_scale[:, None]
            objective = np.where(allowed, objective, -np.inf)
            index = objective.argmax(axis=1)
            if not np.isfinite(objective[rows, index]).all():
                raise RuntimeError("Score-guided selection exhausted a candidate family")
            sequence.append(index)
            selected_mask[rows, index] = True
            duplicate_blocked |= duplicate[rows, index]
            duplicate_blocked[rows, index] = False
            current = distance[rows, index]
            min_distance = current if min_distance is None else np.minimum(min_distance, current)
    selected = np.stack(sequence, axis=1)
    if selected.shape[1] != 20 or np.any(np.diff(np.sort(selected, axis=1), axis=1) == 0):
        raise RuntimeError("EXP-159B did not produce 20 unique candidates")
    return selected, fallback_count


def update_accumulator(acc: dict, candidates: np.ndarray, future: np.ndarray, fallback_count: int) -> None:
    point = np.linalg.norm(candidates - future[:, None], axis=-1)
    ade = point.mean(axis=-1)
    fde = point[..., -1]
    rows = np.arange(len(ade))
    best_ade = ade.argmin(axis=1)
    upper = np.triu_indices(candidates.shape[1], k=1)
    pair = np.linalg.norm(candidates[:, :, None] - candidates[:, None, :], axis=-1)
    acc["samples"] += len(candidates)
    acc["minADE"] += float(ade.min(axis=1).sum(dtype=np.float64))
    acc["minFDE"] += float(fde.min(axis=1).sum(dtype=np.float64))
    acc["same_path_FDE"] += float(fde[rows, best_ade].sum(dtype=np.float64))
    acc["combined_cost"] += float((ade + 0.25 * fde).min(axis=1).sum(dtype=np.float64))
    acc["pairwise_endpoint"] += float(pair[..., -1][:, upper[0], upper[1]].sum(dtype=np.float64))
    acc["pairwise_trajectory"] += float(pair.mean(axis=-1)[:, upper[0], upper[1]].sum(dtype=np.float64))
    acc["pair_count"] += int(len(candidates) * len(upper[0]))
    acc["duplicate_fallbacks"] += int(fallback_count)


def finalize(acc: dict) -> dict:
    samples = acc["samples"]
    return {
        "samples": samples,
        "minADE": acc["minADE"] / samples,
        "minFDE": acc["minFDE"] / samples,
        "same_path_FDE": acc["same_path_FDE"] / samples,
        "combined_cost": acc["combined_cost"] / samples,
        "mean_pairwise_endpoint": acc["pairwise_endpoint"] / acc["pair_count"],
        "mean_pairwise_trajectory": acc["pairwise_trajectory"] / acc["pair_count"],
        "duplicate_fallbacks_per_sample": acc["duplicate_fallbacks"] / samples,
    }


def empty_accumulator() -> dict:
    return {key: 0 for key in (
        "samples", "minADE", "minFDE", "same_path_FDE", "combined_cost",
        "pairwise_endpoint", "pairwise_trajectory", "pair_count", "duplicate_fallbacks",
    )}


def configurations(allocations: tuple[int, ...], weights: tuple[float, ...]) -> tuple[tuple[int, float], ...]:
    configs = []
    for allocation in allocations:
        active_weights = (0.0,) if allocation == 0 else weights
        configs.extend((allocation, weight) for weight in active_weights)
    return tuple(configs)


def evaluate_fold(
    base, source: Path, configs: tuple[tuple[int, float], ...], chunk_size: int,
    endpoint_weight: float, endpoint_duplicate: float, trajectory_duplicate: float,
    max_samples: int,
) -> dict[tuple[int, float], dict]:
    with np.load(source) as raw:
        limit = min(max_samples, len(raw["future"])) if max_samples else len(raw["future"])
        accumulators = {config: empty_accumulator() for config in configs}
        for start in range(0, limit, chunk_size):
            stop = min(start + chunk_size, limit)
            future = raw["future"][start:stop]
            combined = np.concatenate((raw["k20_candidates"][start:stop], raw["v2_candidates"][start:stop]), axis=1)
            score = standardized_score(raw["k20_logits"][start:stop], raw["v2_logits"][start:stop])
            distance, endpoint, trajectory = base.pairwise_geometry(combined, endpoint_weight)
            rows = np.arange(len(combined))[:, None]
            for config in configs:
                allocation, weight = config
                selected, fallback_count = select_candidates(
                    score, distance, endpoint, trajectory, allocation, weight,
                    endpoint_duplicate, trajectory_duplicate,
                )
                update_accumulator(accumulators[config], combined[rows, selected], future, fallback_count)
    return {config: finalize(accumulator) for config, accumulator in accumulators.items()}


def decide(rows: list[dict], baselines: dict, folds: tuple[str, ...], configs: tuple[tuple[int, float], ...]):
    m2_ade = float(np.mean([baselines["m2"][fold]["min_ade_20"] for fold in folds]))
    m2_fde = float(np.mean([baselines["m2"][fold]["min_fde_20"] for fold in folds]))
    summaries = []
    for allocation, weight in configs:
        group = [row for row in rows if row["k20_count"] == allocation and row["diversity_weight"] == weight]
        macro_ade = float(np.mean([row["minADE"] for row in group]))
        macro_fde = float(np.mean([row["minFDE"] for row in group]))
        max_ade = max(row["minADE"] / baselines["k20"][row["fold"]]["min_ade_20"] - 1.0 for row in group)
        max_fde = max(row["minFDE"] / baselines["k20"][row["fold"]]["min_fde_20"] - 1.0 for row in group)
        hard = bool(max_ade <= 0.02 and max_fde <= 0.02)
        m2_gate = bool(macro_ade <= m2_ade and macro_fde <= m2_fde)
        summaries.append({
            "k20_count": allocation, "v2_count": 20 - allocation,
            "diversity_weight": weight,
            "macro_minADE": macro_ade, "macro_minFDE": macro_fde,
            "macro_same_path_FDE": float(np.mean([row["same_path_FDE"] for row in group])),
            "macro_combined_cost": float(np.mean([row["combined_cost"] for row in group])),
            "max_fold_ADE_regression_vs_K20": float(max_ade),
            "max_fold_FDE_regression_vs_K20": float(max_fde),
            "hard_fold_gate": hard, "m2_macro_gate": m2_gate,
            "passed": hard and m2_gate,
        })
    passed = [row for row in summaries if row["passed"]]
    if passed:
        best = min(passed, key=lambda row: row["macro_combined_cost"])
        decision = "PASSED"
    else:
        conditional = [row for row in summaries if row["max_fold_ADE_regression_vs_K20"] <= 0.05 and row["max_fold_FDE_regression_vs_K20"] <= 0.05 and row["macro_minADE"] <= 1.02 * m2_ade and row["macro_minFDE"] <= 1.02 * m2_fde]
        if conditional:
            best = min(conditional, key=lambda row: row["macro_combined_cost"])
            decision = "CONDITIONAL"
        else:
            best = min(summaries, key=lambda row: (max(row["max_fold_ADE_regression_vs_K20"], row["max_fold_FDE_regression_vs_K20"]), row["macro_combined_cost"]))
            decision = "REJECTED"
    return decision, best, summaries


def save_selected(
    base, source: Path, destination: Path, allocation: int, weight: float,
    chunk_size: int, endpoint_weight: float, endpoint_duplicate: float,
    trajectory_duplicate: float,
) -> None:
    with np.load(source) as raw:
        arrays = {key: raw[key] for key in raw.files}
    candidates_parts, logits_parts, family_parts, mode_parts = [], [], [], []
    for start in range(0, len(arrays["future"]), chunk_size):
        stop = min(start + chunk_size, len(arrays["future"]))
        combined = np.concatenate((arrays["k20_candidates"][start:stop], arrays["v2_candidates"][start:stop]), axis=1)
        raw_logits = np.concatenate((arrays["k20_logits"][start:stop], arrays["v2_logits"][start:stop]), axis=1)
        score = standardized_score(arrays["k20_logits"][start:stop], arrays["v2_logits"][start:stop])
        distance, endpoint, trajectory = base.pairwise_geometry(combined, endpoint_weight)
        selected, _ = select_candidates(
            score, distance, endpoint, trajectory, allocation, weight,
            endpoint_duplicate, trajectory_duplicate,
        )
        rows = np.arange(len(combined))[:, None]
        candidates_parts.append(combined[rows, selected])
        logits_parts.append(raw_logits[rows, selected])
        family_parts.append((selected >= 20).astype(np.int8))
        mode_parts.append(np.where(selected < 20, selected, selected - 20).astype(np.int8))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        past=arrays["past"], future=arrays["future"], num_valid=arrays["num_valid"],
        candidates=np.concatenate(candidates_parts), original_logits=np.concatenate(logits_parts),
        family_id=np.concatenate(family_parts), source_mode=np.concatenate(mode_parts),
        k20_count=np.asarray(allocation), v2_count=np.asarray(20 - allocation),
        diversity_weight=np.asarray(weight),
    )


def write_analysis(path: Path, decision: str, best: dict) -> None:
    if decision == "PASSED":
        route = "固定该score-guided候选压缩方案，进入EXP-161共享listwise scorer。"
    elif decision == "CONDITIONAL":
        route = "方案接近门槛。先检查具体退化fold，再决定进入EXP-161或进行可学习Top-20压缩。"
    else:
        route = "零学习压缩仍不足。停止静态规则搜索，转向训练集监督的轻量listwise Top-20选择器。"
    path.write_text(
        f"# EXP-159B Result Analysis\n\n## Decision: {decision}\n\n"
        f"推荐配置：K20={best['k20_count']}，V2={best['v2_count']}，diversity weight={best['diversity_weight']:.2f}。\n\n"
        f"宏平均 minADE/minFDE/same-path FDE = {best['macro_minADE']:.4f}/"
        f"{best['macro_minFDE']:.4f}/{best['macro_same_path_FDE']:.4f}。\n\n"
        f"相对matched K20最坏逐折ADE/FDE退化 = "
        f"{100*best['max_fold_ADE_regression_vs_K20']:.2f}%/"
        f"{100*best['max_fold_FDE_regression_vs_K20']:.2f}%。\n\n"
        f"## Next step\n\n{route}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--allocations", nargs="+", type=int, default=list(DEFAULT_ALLOCATIONS))
    parser.add_argument("--diversity-weights", nargs="+", type=float, default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--endpoint-weight", type=float, default=0.5)
    parser.add_argument("--endpoint-duplicate", type=float, default=0.08)
    parser.add_argument("--trajectory-duplicate", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-save-selected", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_module("exp159b_base", BASE_SCRIPT)
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    allocations = tuple(sorted(set(args.allocations)))
    weights = tuple(sorted(set(args.diversity_weights)))
    if any(value < 0 or value >= 20 for value in allocations):
        raise ValueError("K20 allocations must be between 0 and 19")
    if any(value < 0 for value in weights):
        raise ValueError("Diversity weights must be non-negative")
    base.preflight(args.input_dir, folds, args.seed)
    configs = configurations(allocations, weights)
    if args.dry_run:
        print(json.dumps({
            "status": "dry-run passed", "folds": folds, "configurations": configs,
            "selection_uses_GT": False, "final_K": 20,
        }))
        return
    baselines = base.load_baselines()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in folds:
        source = args.input_dir / f"{fold}_v2_host_seed_{args.seed}.npz"
        print(f"[evaluate] fold={fold} configs={len(configs)}", flush=True)
        results = evaluate_fold(
            base, source, configs, args.chunk_size, args.endpoint_weight,
            args.endpoint_duplicate, args.trajectory_duplicate, args.max_samples,
        )
        for (allocation, weight), metrics in results.items():
            row = {"fold": fold, "seed": args.seed, "k20_count": allocation,
                   "v2_count": 20 - allocation, "diversity_weight": weight, **metrics}
            rows.append(row)
            print(f"[done] fold={fold} config={allocation}+{20-allocation}@{weight:.2f} ADE={metrics['minADE']:.4f} FDE={metrics['minFDE']:.4f}", flush=True)
    decision, best, config_summary = decide(rows, baselines, folds, configs)
    summary = {
        "experiment": "EXP-159B Score-Guided Diversity Compression",
        "protocol": "official ETH/UCY 8/12 at 2.5 Hz; V2 shared encoder; seed 1",
        "selection_contract": {
            "uses_ground_truth": False, "final_candidates": 20,
            "score_normalization": "per sample, within candidate family",
            "endpoint_weight": args.endpoint_weight,
            "endpoint_duplicate_threshold": args.endpoint_duplicate,
            "trajectory_duplicate_threshold": args.trajectory_duplicate,
        },
        "decision": decision, "recommended": best,
        "configuration_summary": config_summary, "per_fold": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "per_fold_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_analysis(args.output_dir / "RESULT_ANALYSIS.md", decision, best)
    if decision != "REJECTED" and not args.no_save_selected and not args.max_samples:
        selected_dir = args.output_dir / "selected_candidates"
        for fold in folds:
            print(f"[save] fold={fold} config={best['k20_count']}+{best['v2_count']}@{best['diversity_weight']:.2f}", flush=True)
            save_selected(
                base, args.input_dir / f"{fold}_v2_host_seed_{args.seed}.npz",
                selected_dir / f"{fold}_k20_{best['k20_count']}_v2_{best['v2_count']}_w_{best['diversity_weight']:.2f}_seed_{args.seed}.npz",
                int(best["k20_count"]), float(best["diversity_weight"]), args.chunk_size,
                args.endpoint_weight, args.endpoint_duplicate, args.trajectory_duplicate,
            )
    print(json.dumps({"experiment": "EXP-159B", "decision": decision, "recommended": best, "output": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
