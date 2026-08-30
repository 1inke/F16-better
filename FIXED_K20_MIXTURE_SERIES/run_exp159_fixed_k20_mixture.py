"""EXP-159: fixed-K20 geometry-only mixture of shared-encoder K20 and V2 candidates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DEFAULT_INPUT = WORKSPACE / "SHARED_ENCODER_COMPLEMENTARITY_SERIES" / "EXP_158B_SHARED_ENCODER_COMPLEMENTARITY" / "per_sample"
DEFAULT_OUT = HERE / "EXP_159_FIXED_K20_FAMILY_MIXTURE"
K20_SUMMARY = WORKSPACE / "FOLD_STAGE_ATTRIBUTION_SERIES" / "EXP_157B_MATCHED_K20" / "summary.json"
M2_SUMMARY = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "EXP_156B_FOLD_LOCAL_CURRENT_DECODER" / "summary.json"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
DEFAULT_ALLOCATIONS = (4, 6, 8, 10, 12, 14, 16)


def preflight(input_dir: Path, folds: tuple[str, ...], seed: int) -> None:
    required = [K20_SUMMARY, M2_SUMMARY]
    required += [input_dir / f"{fold}_v2_host_seed_{seed}.npz" for fold in folds]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing EXP-159 inputs:\n" + "\n".join(missing))


def load_baselines() -> dict:
    k20_raw = json.loads(K20_SUMMARY.read_text(encoding="utf-8"))["rows"]
    m2_raw = json.loads(M2_SUMMARY.read_text(encoding="utf-8"))["rows"]
    k20 = {row["fold"]: row for row in k20_raw if int(row["seed"]) == 1}
    m2 = {row["fold"]: row for row in m2_raw if int(row["seed"]) == 1}
    if set(k20) != set(FOLDS) or set(m2) != set(FOLDS):
        raise RuntimeError("Official K20/M2 baseline summaries are incomplete")
    return {"k20": k20, "m2": m2}


def pairwise_geometry(candidates: np.ndarray, endpoint_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    displacement = candidates[:, :, None] - candidates[:, None, :]
    point_distance = np.linalg.norm(displacement, axis=-1)
    trajectory = point_distance.mean(axis=-1)
    endpoint = point_distance[..., -1]
    return trajectory + float(endpoint_weight) * endpoint, endpoint, trajectory


def choose_index(
    distance: np.ndarray,
    pool: np.ndarray,
    selected_mask: np.ndarray,
    min_distance: np.ndarray | None,
    duplicate_blocked: np.ndarray,
    initial_medoid: bool,
) -> np.ndarray:
    batch, modes = pool.shape
    allowed = pool & ~selected_mask & ~duplicate_blocked
    fallback = pool & ~selected_mask
    empty = ~allowed.any(axis=1)
    allowed[empty] = fallback[empty]
    if initial_medoid:
        pool_pair = pool[:, :, None] & pool[:, None, :]
        count = pool_pair.sum(axis=2).clip(min=1)
        mean_distance = np.where(pool_pair, distance, 0.0).sum(axis=2) / count
        objective = -mean_distance
    else:
        if min_distance is None:
            objective = distance.mean(axis=2)
        else:
            objective = min_distance
    objective = np.where(allowed, objective, -np.inf)
    index = objective.argmax(axis=1)
    if not np.isfinite(objective[np.arange(batch), index]).all():
        raise RuntimeError("Candidate selection exhausted a family before reaching its quota")
    return index


def select_allocation(
    candidates: np.ndarray,
    k20_count: int,
    distance: np.ndarray,
    endpoint_distance: np.ndarray,
    trajectory_distance: np.ndarray,
    endpoint_duplicate: float,
    trajectory_duplicate: float,
) -> np.ndarray:
    batch, modes = candidates.shape[:2]
    if modes != 40 or not 0 < k20_count < 20:
        raise ValueError((modes, k20_count))
    v2_count = 20 - k20_count
    family_pool = {
        "k20": np.broadcast_to(np.arange(modes)[None] < 20, (batch, modes)).copy(),
        "v2": np.broadcast_to(np.arange(modes)[None] >= 20, (batch, modes)).copy(),
    }
    selected_mask = np.zeros((batch, modes), dtype=bool)
    duplicate_blocked = np.zeros((batch, modes), dtype=bool)
    min_distance = None
    sequence = []
    duplicate = (endpoint_distance < endpoint_duplicate) & (trajectory_distance < trajectory_duplicate)
    rows = np.arange(batch)

    # V2 is the native decoder. Cover its geometry first, then add K20 candidates
    # which maximize new coverage relative to the already selected V2 set.
    for family, quota in (("v2", v2_count), ("k20", k20_count)):
        for step in range(quota):
            index = choose_index(
                distance, family_pool[family], selected_mask, min_distance,
                duplicate_blocked, initial_medoid=(family == "v2" and step == 0),
            )
            sequence.append(index)
            selected_mask[rows, index] = True
            duplicate_blocked |= duplicate[rows, index]
            duplicate_blocked[rows, index] = False
            current = distance[rows, index]
            min_distance = current if min_distance is None else np.minimum(min_distance, current)
    selected = np.stack(sequence, axis=1)
    if selected.shape[1] != 20 or np.any(np.diff(np.sort(selected, axis=1), axis=1) == 0):
        raise RuntimeError("EXP-159 selection did not produce 20 unique candidates")
    return selected


def gather(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rows = np.arange(len(values))[:, None]
    return values[rows, indices]


def oracle_metrics(candidates: np.ndarray, future: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(candidates - future[:, None], axis=-1)
    ade = distance.mean(axis=-1)
    fde = distance[..., -1]
    rows = np.arange(len(ade))
    best_ade = ade.argmin(axis=1)
    return {
        "samples": int(len(ade)),
        "minADE": float(ade.min(axis=1).mean()),
        "minFDE": float(fde.min(axis=1).mean()),
        "same_path_FDE": float(fde[rows, best_ade].mean()),
        "combined_cost": float((ade + 0.25 * fde).min(axis=1).mean()),
    }


def diversity_metrics(candidates: np.ndarray, chunk_size: int = 128) -> dict[str, float]:
    upper = np.triu_indices(candidates.shape[1], k=1)
    endpoint_sum = trajectory_sum = 0.0
    count = 0
    for start in range(0, len(candidates), chunk_size):
        part = candidates[start:start + chunk_size]
        point = np.linalg.norm(part[:, :, None] - part[:, None, :], axis=-1)
        endpoint_values = point[..., -1][:, upper[0], upper[1]]
        trajectory_values = point.mean(axis=-1)[:, upper[0], upper[1]]
        endpoint_sum += float(endpoint_values.sum(dtype=np.float64))
        trajectory_sum += float(trajectory_values.sum(dtype=np.float64))
        count += int(endpoint_values.size)
    return {
        "mean_pairwise_endpoint": endpoint_sum / count,
        "mean_pairwise_trajectory": trajectory_sum / count,
    }


def evaluate_fold(
    cache_path: Path,
    allocations: tuple[int, ...],
    chunk_size: int,
    endpoint_weight: float,
    endpoint_duplicate: float,
    trajectory_duplicate: float,
    max_samples: int,
) -> dict[int, dict]:
    with np.load(cache_path) as raw:
        limit = min(max_samples, len(raw["future"])) if max_samples else len(raw["future"])
        future = raw["future"][:limit]
        k20 = raw["k20_candidates"][:limit]
        v2 = raw["v2_candidates"][:limit]
    accumulators = {allocation: {"candidates": [], "future": []} for allocation in allocations}
    for start in range(0, len(future), chunk_size):
        stop = min(start + chunk_size, len(future))
        combined = np.concatenate((k20[start:stop], v2[start:stop]), axis=1)
        distance, endpoint, trajectory = pairwise_geometry(combined, endpoint_weight)
        for allocation in allocations:
            selected = select_allocation(
                combined, allocation, distance, endpoint, trajectory,
                endpoint_duplicate, trajectory_duplicate,
            )
            accumulators[allocation]["candidates"].append(gather(combined, selected))
            accumulators[allocation]["future"].append(future[start:stop])
    result = {}
    for allocation, parts in accumulators.items():
        candidates = np.concatenate(parts["candidates"])
        target = np.concatenate(parts["future"])
        result[allocation] = {**oracle_metrics(candidates, target), **diversity_metrics(candidates, chunk_size)}
    return result


def select_and_save(
    source: Path,
    destination: Path,
    allocation: int,
    chunk_size: int,
    endpoint_weight: float,
    endpoint_duplicate: float,
    trajectory_duplicate: float,
) -> None:
    with np.load(source) as raw:
        arrays = {key: raw[key] for key in raw.files}
    candidate_parts, logit_parts, family_parts, index_parts = [], [], [], []
    combined_logits = np.concatenate((arrays["k20_logits"], arrays["v2_logits"]), axis=1)
    for start in range(0, len(arrays["future"]), chunk_size):
        stop = min(start + chunk_size, len(arrays["future"]))
        combined = np.concatenate((arrays["k20_candidates"][start:stop], arrays["v2_candidates"][start:stop]), axis=1)
        distance, endpoint, trajectory = pairwise_geometry(combined, endpoint_weight)
        selected = select_allocation(
            combined, allocation, distance, endpoint, trajectory,
            endpoint_duplicate, trajectory_duplicate,
        )
        candidate_parts.append(gather(combined, selected))
        logit_parts.append(gather(combined_logits[start:stop], selected))
        family_parts.append((selected >= 20).astype(np.int8))
        index_parts.append(np.where(selected < 20, selected, selected - 20).astype(np.int8))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        past=arrays["past"], future=arrays["future"], num_valid=arrays["num_valid"],
        candidates=np.concatenate(candidate_parts), original_logits=np.concatenate(logit_parts),
        family_id=np.concatenate(family_parts), source_mode=np.concatenate(index_parts),
        k20_count=np.asarray(allocation, dtype=np.int64),
        v2_count=np.asarray(20 - allocation, dtype=np.int64),
    )


def decide(rows: list[dict], baselines: dict, folds: tuple[str, ...], allocations: tuple[int, ...]) -> tuple[str, int, list[dict]]:
    evaluations = []
    m2_macro_ade = np.mean([baselines["m2"][fold]["min_ade_20"] for fold in folds])
    m2_macro_fde = np.mean([baselines["m2"][fold]["min_fde_20"] for fold in folds])
    for allocation in allocations:
        group = [row for row in rows if row["k20_count"] == allocation]
        macro_ade = float(np.mean([row["minADE"] for row in group]))
        macro_fde = float(np.mean([row["minFDE"] for row in group]))
        max_ade_regression = max(
            row["minADE"] / baselines["k20"][row["fold"]]["min_ade_20"] - 1.0
            for row in group
        )
        max_fde_regression = max(
            row["minFDE"] / baselines["k20"][row["fold"]]["min_fde_20"] - 1.0
            for row in group
        )
        hard_fold_gate = bool(max_ade_regression <= 0.02 and max_fde_regression <= 0.02)
        m2_macro_gate = bool(macro_ade <= m2_macro_ade and macro_fde <= m2_macro_fde)
        evaluations.append({
            "k20_count": allocation, "v2_count": 20 - allocation,
            "macro_minADE": macro_ade, "macro_minFDE": macro_fde,
            "macro_same_path_FDE": float(np.mean([row["same_path_FDE"] for row in group])),
            "macro_combined_cost": float(np.mean([row["combined_cost"] for row in group])),
            "max_fold_ADE_regression_vs_K20": float(max_ade_regression),
            "max_fold_FDE_regression_vs_K20": float(max_fde_regression),
            "hard_fold_gate": hard_fold_gate, "m2_macro_gate": m2_macro_gate,
            "passed": hard_fold_gate and m2_macro_gate,
        })
    passed = [row for row in evaluations if row["passed"]]
    if passed:
        chosen = min(passed, key=lambda row: row["macro_combined_cost"])
        decision = "PASSED"
    else:
        conditional = [row for row in evaluations if row["max_fold_ADE_regression_vs_K20"] <= 0.05 and row["max_fold_FDE_regression_vs_K20"] <= 0.05 and row["macro_minADE"] <= 1.02 * m2_macro_ade and row["macro_minFDE"] <= 1.02 * m2_macro_fde]
        if conditional:
            chosen = min(conditional, key=lambda row: row["macro_combined_cost"])
            decision = "CONDITIONAL"
        else:
            chosen = min(evaluations, key=lambda row: (max(row["max_fold_ADE_regression_vs_K20"], row["max_fold_FDE_regression_vs_K20"]), row["macro_combined_cost"]))
            decision = "REJECTED"
    return decision, int(chosen["k20_count"]), evaluations


def write_analysis(path: Path, decision: str, chosen: int, evaluations: list[dict]) -> None:
    best = next(row for row in evaluations if row["k20_count"] == chosen)
    if decision == "PASSED":
        route = "固定该配额并进入 EXP-161，在冻结20条候选上训练共享 listwise scorer。"
    elif decision == "CONDITIONAL":
        route = "配额接近门槛但仍有轻微逐折退化；先检查退化fold，再决定直接评分同步还是调整候选选择。"
    else:
        route = "K40互补性未能在固定K=20下保留。暂不进入EXP-161，应先修正候选压缩策略或对移植K20分支做冻结编码器适配。"
    path.write_text(
        f"# EXP-159 Result Analysis\n\n## Decision: {decision}\n\n"
        f"自动推荐配额：K20={chosen}，V2={20-chosen}。\n\n"
        f"宏平均 minADE/minFDE/same-path FDE = {best['macro_minADE']:.4f}/"
        f"{best['macro_minFDE']:.4f}/{best['macro_same_path_FDE']:.4f}。\n\n"
        f"相对 matched K20 的最坏逐折 ADE/FDE 退化 = "
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
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--endpoint-weight", type=float, default=0.5)
    parser.add_argument("--endpoint-duplicate", type=float, default=0.08)
    parser.add_argument("--trajectory-duplicate", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-save-selected", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    allocations = tuple(sorted(set(args.allocations)))
    if any(value <= 0 or value >= 20 for value in allocations):
        raise ValueError("Every K20 allocation must be between 1 and 19")
    preflight(args.input_dir, folds, args.seed)
    if args.dry_run:
        print(json.dumps({
            "status": "dry-run passed", "folds": folds, "allocations": allocations,
            "selection_uses_GT": False, "final_K": 20,
        }))
        return
    baselines = load_baselines()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in folds:
        source = args.input_dir / f"{fold}_v2_host_seed_{args.seed}.npz"
        print(f"[evaluate] fold={fold} allocations={allocations}", flush=True)
        result = evaluate_fold(
            source, allocations, args.chunk_size, args.endpoint_weight,
            args.endpoint_duplicate, args.trajectory_duplicate, args.max_samples,
        )
        for allocation, metrics in result.items():
            rows.append({"fold": fold, "seed": args.seed, "k20_count": allocation, "v2_count": 20 - allocation, **metrics})
            print(f"[done] fold={fold} allocation={allocation}+{20-allocation} ADE={metrics['minADE']:.4f} FDE={metrics['minFDE']:.4f}", flush=True)
    decision, chosen, allocation_summary = decide(rows, baselines, folds, allocations)
    summary = {
        "experiment": "EXP-159 Fixed-K20 K20/V2 Family Mixture",
        "protocol": "official ETH/UCY 8/12 at 2.5 Hz; V2 shared encoder; seed 1",
        "selection_contract": {
            "uses_ground_truth": False, "final_candidates": 20,
            "endpoint_weight": args.endpoint_weight,
            "endpoint_duplicate_threshold": args.endpoint_duplicate,
            "trajectory_duplicate_threshold": args.trajectory_duplicate,
            "native_family_priority": "V2",
        },
        "decision": decision, "recommended_k20_count": chosen,
        "recommended_v2_count": 20 - chosen,
        "allocation_summary": allocation_summary, "per_fold": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "per_fold_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_analysis(args.output_dir / "RESULT_ANALYSIS.md", decision, chosen, allocation_summary)
    if not args.no_save_selected and not args.max_samples:
        selected_dir = args.output_dir / "selected_candidates"
        for fold in folds:
            print(f"[save] fold={fold} allocation={chosen}+{20-chosen}", flush=True)
            select_and_save(
                args.input_dir / f"{fold}_v2_host_seed_{args.seed}.npz",
                selected_dir / f"{fold}_k20_{chosen}_v2_{20-chosen}_seed_{args.seed}.npz",
                chosen, args.chunk_size, args.endpoint_weight,
                args.endpoint_duplicate, args.trajectory_duplicate,
            )
    print(json.dumps({"experiment": "EXP-159", "decision": decision, "recommended_allocation": f"{chosen}+{20-chosen}", "output": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
