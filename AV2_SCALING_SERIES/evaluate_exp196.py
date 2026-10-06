"""Evaluate EXP-196 checkpoints with paired cohorts and scenario bootstrap."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from exp196_common import (
    ScalingArrayDataset, build_model, load_metadata, model_forward,
    normalize_to_focal_heading, per_target_metrics,
    scenario_bootstrap_difference, summarize,
)
from exp196_constants import (
    ROOT, STAGE_VARIANTS, VARIANTS, checkpoint_for, config_for,
    data_paths, metadata_path,
)
from materialize_configs import materialize


OUTPUTS = {
    "fair": ROOT / "EXP_196A_CONTINUED_A1",
    "long": ROOT / "EXP_196C_LONG_HISTORY",
    "dense": ROOT / "EXP_196D_DENSE_NEIGHBOR",
    "joint": ROOT / "EXP_196E_JOINT_STRESS",
}
DEFAULT_BATCH = {"fair": 64, "long": 32, "dense": 16, "joint": 8}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collision_metrics(selected_prediction: np.ndarray, future: np.ndarray,
                      indices: np.ndarray, metadata: list[dict], max_agents: int):
    collision = np.zeros(len(indices), dtype=np.float32)
    near = np.zeros(len(indices), dtype=np.float32)
    for local, original in enumerate(indices):
        validity = metadata[int(original)].get("future_valid_agents", [])[:max_agents]
        valid_neighbors = [index for index in range(1, min(len(validity), future.shape[1])) if validity[index]]
        if not valid_neighbors:
            collision[local] = np.nan
            near[local] = np.nan
            continue
        distance = np.linalg.norm(
            selected_prediction[local, None] - future[local, valid_neighbors], axis=-1
        )
        minimum = float(distance.min())
        collision[local] = minimum < 0.20
        near[local] = minimum < 0.50
    return collision, near


def evaluate(name: str, seed: int, device: torch.device, requested_batch: int):
    spec = VARIANTS[name]
    data_path, num_path = data_paths(name, "val")
    dataset = ScalingArrayDataset(data_path, num_path, name)
    metadata = load_metadata(metadata_path(name, "val"), len(dataset))
    batch_size = requested_batch or DEFAULT_BATCH[str(spec["stage"])]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model, _ = build_model(config_for(name), checkpoint_for(name, seed), device)
    model.eval()
    predictions, targets, logits, all_future, all_indices = [], [], [], [], []
    edge_values, degree_values, max_degree_values = [], [], []
    with torch.inference_mode():
        for past, future, num_valid, indices in loader:
            past, future = past.to(device), future.to(device)
            past, future = normalize_to_focal_heading(past, future)
            output, score = model_forward(model, past, num_valid.to(device))
            predictions.append(output.cpu().numpy())
            targets.append(future[:, 0].cpu().numpy())
            logits.append(score.cpu().numpy())
            all_future.append(future.cpu().numpy())
            all_indices.append(indices.numpy())
            stats = model.last_graph_stats
            edge_values.append(stats.edge_count.detach().cpu().numpy())
            degree_values.append(stats.mean_out_degree.detach().cpu().numpy())
            max_degree_values.append(stats.max_out_degree.detach().cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    scores = np.concatenate(logits)
    future = np.concatenate(all_future)
    indices = np.concatenate(all_indices)
    per_target = per_target_metrics(prediction, target, scores)
    selected = prediction[np.arange(len(prediction)), scores.argmax(axis=1)]
    collision, near = collision_metrics(selected, future, indices, metadata, int(spec["max_agents"]))
    per_target["collision_rate"] = collision
    per_target["near_collision_rate"] = near
    summary = summarize({key: value for key, value in per_target.items() if not np.isnan(value).all()})
    summary.update({
        "config": name,
        "seed": seed,
        "stage": spec["stage"],
        "history_steps": spec["history_steps"],
        "history_hz": spec["history_hz"],
        "max_agents": spec["max_agents"],
        "graph_top_k": spec["top_k"],
        "samples": len(dataset),
        "effective_graph_edges": float(np.mean(np.concatenate(edge_values))),
        "mean_selected_neighbors": float(np.mean(np.concatenate(degree_values))),
        "p90_max_selected_neighbors": float(np.percentile(np.concatenate(max_degree_values), 90)),
        "truncation_rate": float(np.mean([
            metadata[int(index)].get("valid_neighbor_count", 0) >= int(spec["max_agents"])
            for index in indices
        ])) if any("valid_neighbor_count" in metadata[int(index)] for index in indices) else float("nan"),
        "av2_status": "NON-OFFICIAL AV2 DIAGNOSTIC",
    })
    scenario_ids = np.asarray([metadata[int(index)]["scenario_id"] for index in indices])
    return summary, per_target, metadata, indices, scenario_ids


def cohort_rows(name: str, seed: int, per_target: dict[str, np.ndarray],
                metadata: list[dict], indices: np.ndarray) -> list[dict]:
    stage = str(VARIANTS[name]["stage"])
    rows = []
    definitions = {"all": np.ones(len(indices), dtype=bool)}
    if stage == "long":
        for key in ("near_stationary", "accelerating", "turning", "high_curvature"):
            definitions[key] = np.asarray([bool(metadata[int(index)].get(key, False)) for index in indices])
    if stage in ("dense", "joint"):
        labels = np.asarray([metadata[int(index)].get("dense_cohort", "unknown") for index in indices])
        for label in sorted(set(labels)):
            definitions[label] = labels == label
    for cohort, mask in definitions.items():
        selected = np.flatnonzero(mask)
        if not len(selected):
            continue
        valid_metrics = {
            key: values for key, values in per_target.items()
            if not np.isnan(values[selected]).all()
        }
        rows.append({
            "config": name, "seed": seed, "cohort": cohort,
            "samples": len(selected), **summarize(valid_metrics, selected),
        })
    return rows


def improvement(candidate: float, baseline: float) -> float:
    return (baseline - candidate) / baseline


def decisions(rows: list[dict]) -> list[str]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    by_name = {
        name: {
            key: float(np.mean([row[key] for row in values]))
            for key in ("minADE20", "minFDE20", "selected_ADE", "selected_FDE")
        }
        for name, values in grouped.items()
    }
    labels = []
    if all(key in by_name for key in ("a1c", "a3")):
        base, high = by_name["a1c"], by_name["a3"]
        gain = max(improvement(high["minADE20"], base["minADE20"]), improvement(high["minFDE20"], base["minFDE20"]))
        selected_degradation = -improvement(high["selected_ADE"], base["selected_ADE"])
        labels.append("HIGH_RES_HISTORY_CONFIRMED" if gain >= 0.02 and selected_degradation <= 0.02 else "HIGH_RES_GAIN_NOT_CONFIRMED")
    if all(key in by_name for key in ("l2", "l4")):
        base, long = by_name["l2"], by_name["l4"]
        gain = max(improvement(long["minADE20"], base["minADE20"]), improvement(long["minFDE20"], base["minFDE20"]))
        labels.append("LONG_HISTORY_USEFUL" if gain >= 0.02 else ("LONG_HISTORY_HARMFUL" if gain < -0.03 else "LONG_HISTORY_SATURATED"))
    if all(key in by_name for key in ("d2", "d4")):
        base, wide = by_name["d2"], by_name["d4"]
        if improvement(wide["minADE20"], base["minADE20"]) >= 0.02:
            labels.append("AGENT_TRUNCATION_BOTTLENECK")
    if all(key in by_name for key in ("d3", "d5")):
        if improvement(by_name["d5"]["minADE20"], by_name["d3"]["minADE20"]) >= 0.02:
            labels.append("SOCIAL_CAPACITY_LIMITED")
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    materialize()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows, cohorts = [], []
    target_store = {}
    for name in args.variants:
        for seed in args.seeds:
            checkpoint = checkpoint_for(name, seed)
            if not checkpoint.is_file():
                print(f"[skip] {name} seed={seed}: {checkpoint}")
                continue
            summary, per_target, metadata, indices, scenario_ids = evaluate(name, seed, device, args.batch_size)
            rows.append(summary)
            cohorts.extend(cohort_rows(name, seed, per_target, metadata, indices))
            target_store[(name, seed)] = (per_target, scenario_ids)
            print(f"[evaluated] {name} seed={seed} minADE20={summary['minADE20']:.4f}")

    by_stage = defaultdict(list)
    for row in rows:
        by_stage[row["stage"]].append(row)
    for stage, stage_rows in by_stage.items():
        folder = OUTPUTS[stage]
        filename = {"fair": "comparison.csv", "long": "temporal_scaling.csv", "dense": "neighbor_scaling.csv", "joint": "joint_scaling.csv"}[stage]
        write_csv(folder / filename, stage_rows)
        write_csv(folder / "cohort_metrics.csv", [row for row in cohorts if VARIANTS[row["config"]]["stage"] == stage])

    multi = ROOT / "EXP_196B_MULTI_SEED_TEMPORAL"
    fair_rows = [row for row in rows if row["stage"] == "fair"]
    write_csv(multi / "all_runs.csv", fair_rows)
    summary_rows = []
    for name in STAGE_VARIANTS["fair"]:
        values = [row for row in fair_rows if row["config"] == name]
        if not values:
            continue
        item = {"config": name, "seeds": len(values)}
        for metric in ("minADE20", "minFDE20", "selected_ADE", "selected_FDE", "mean_regret", "hit_best_ade", "minADE6", "minFDE6", "MR6_at_2m"):
            numbers = np.asarray([row[metric] for row in values])
            item[f"{metric}_mean"] = float(numbers.mean())
            item[f"{metric}_std"] = float(numbers.std(ddof=1)) if len(numbers) > 1 else 0.0
        summary_rows.append(item)
    multi.mkdir(parents=True, exist_ok=True)
    (multi / "multiseed_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    pairs = (("a1c", "a2"), ("a1c", "a3"), ("l2", "l1"), ("l2", "l3"), ("l2", "l4"),
             ("d2", "d1"), ("d2", "d3"), ("d2", "d4"), ("d3", "d5"), ("d4", "d6"), ("d2", "j2"))
    bootstrap = []
    for baseline, candidate in pairs:
        for seed in args.seeds:
            if (baseline, seed) not in target_store or (candidate, seed) not in target_store:
                continue
            base_metrics, scenes = target_store[(baseline, seed)]
            candidate_metrics, candidate_scenes = target_store[(candidate, seed)]
            if len(scenes) != len(candidate_scenes) or not np.array_equal(scenes, candidate_scenes):
                continue
            for metric in ("minADE20", "minFDE20", "selected_ADE", "selected_FDE"):
                result = scenario_bootstrap_difference(candidate_metrics[metric], base_metrics[metric], scenes, args.bootstrap)
                bootstrap.append({"baseline": baseline, "candidate": candidate, "seed": seed, "metric": metric, **result})
    write_csv(ROOT / "bootstrap_comparisons.csv", bootstrap)
    payload = {"rows": rows, "decision_labels": decisions(rows), "bootstrap_samples": args.bootstrap}
    (ROOT / "exp196_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["decision_labels"], ensure_ascii=False))


if __name__ == "__main__":
    main()
