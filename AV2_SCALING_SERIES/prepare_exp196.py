"""Build matched full-history and dense-neighbor AV2 arrays for EXP-196."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from exp196_constants import FUTURE_INDICES, FUTURE_STEPS, MAX_SOURCE_AGENTS, ROOT, VARIANTS


TARGET_CATEGORIES = {2: "SCORED_TRACK", 3: "FOCAL_TRACK"}
FULL_HISTORY = tuple(range(50))
DENSE_HISTORY = tuple(range(18, 50))
COLUMNS = [
    "track_id", "object_type", "object_category", "timestep",
    "position_x", "position_y", "scenario_id", "city",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "av2_dataset")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--groups", nargs="+", choices=("temporal", "dense", "joint"), default=("temporal", "dense"))
    parser.add_argument("--temporal-train-cap", type=int, default=8000)
    parser.add_argument("--temporal-val-cap", type=int, default=2000)
    parser.add_argument("--dense-train-cap", type=int, default=8000)
    parser.add_argument("--dense-val-cap", type=int, default=2000)
    parser.add_argument("--dense-min-per-cohort", type=int, default=20)
    parser.add_argument("--joint-train-cap", type=int, default=200)
    parser.add_argument("--joint-val-cap", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 means no explicit scan limit")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_frame(path: Path):
    return pq.read_table(path, columns=COLUMNS).to_pandas()


def indexed_track(group) -> dict[int, np.ndarray]:
    return {
        int(row.timestep): np.asarray([row.position_x, row.position_y], dtype=np.float32)
        for row in group.itertuples(index=False)
    }


def base_cohort(neighbors: int) -> str:
    if neighbors <= 1:
        return "S0"
    if neighbors <= 4:
        return "S1"
    if neighbors <= 8:
        return "S2"
    return "S3"


def inspect_scenario(directory: Path, history_indices: tuple[int, ...]):
    path = directory / f"scenario_{directory.name}.parquet"
    if not path.is_file():
        return []
    frame = read_frame(path)
    pedestrians = frame[frame["object_type"].str.lower() == "pedestrian"]
    tracks = {
        str(track_id): indexed_track(group)
        for track_id, group in pedestrians.groupby("track_id", sort=False)
    }
    required_context = set(history_indices)
    required_target = required_context | set(FUTURE_INDICES)
    complete_context = [track_id for track_id, points in tracks.items() if required_context.issubset(points)]
    city = str(frame["city"].iloc[0]) if len(frame) else "unknown"
    candidates = []
    for track_id, group in pedestrians.groupby("track_id", sort=False):
        key = str(track_id)
        category = int(group["object_category"].iloc[0])
        if category not in TARGET_CATEGORIES or key not in tracks or not required_target.issubset(tracks[key]):
            continue
        current = tracks[key][49]
        neighbors = sum(
            key != other and float(np.linalg.norm(tracks[other][49] - current)) < 6.0
            for other in complete_context
        )
        candidates.append({
            "scenario_id": directory.name,
            "track_id": key,
            "object_type": "PEDESTRIAN",
            "category": TARGET_CATEGORIES[category],
            "city_name": city,
            "valid_neighbor_count": int(neighbors),
            "raw_cohort": base_cohort(int(neighbors)),
        })
    return candidates


def shuffled_directories(dataset_root: Path, split: str, seed: int):
    directories = [item for item in (dataset_root / split).iterdir() if item.is_dir()]
    rng = np.random.default_rng(seed + (0 if split == "train" else 1))
    rng.shuffle(directories)
    return directories


def scan_manifest(dataset_root: Path, split: str, cap: int, seed: int, workers: int,
                  max_scenarios: int, group: str, min_per_cohort: int):
    directories = shuffled_directories(dataset_root, split, seed)
    if max_scenarios:
        directories = directories[:max_scenarios]
    selected = []
    scanned = 0
    batch_size = max(64, workers * 16)
    history_indices = DENSE_HISTORY if group == "dense" else FULL_HISTORY
    inspector = partial(inspect_scenario, history_indices=history_indices)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in range(0, len(directories), batch_size):
            chunk = directories[start:start + batch_size]
            for candidates in pool.map(inspector, chunk):
                scanned += 1
                selected.extend(candidates)
            counts = Counter(item["raw_cohort"] for item in selected)
            dense_enough = (
                counts["S0"] >= min_per_cohort
                and counts["S1"] >= min_per_cohort
                and counts["S2"] + counts["S3"] >= min_per_cohort
            )
            if scanned % 1024 == 0:
                print(f"[{split}] scanned={scanned} candidates={len(selected)} cohorts={dict(counts)}", flush=True)
            high_count = counts["S2"] + counts["S3"]
            ready = (
                len(selected) >= cap if group == "temporal"
                else (len(selected) >= cap and dense_enough) if group == "dense"
                else high_count >= cap
            )
            if ready:
                break
    if group == "dense":
        selected = balance_dense(selected, cap, min_per_cohort, seed + (0 if split == "train" else 1))
    elif group == "joint":
        selected = [item for item in selected if item["raw_cohort"] in ("S2", "S3")]
        rng = np.random.default_rng(seed + (0 if split == "train" else 1))
        rng.shuffle(selected)
        selected = selected[:cap]
        for item in selected:
            item["dense_cohort"] = item["raw_cohort"]
    else:
        rng = np.random.default_rng(seed + (0 if split == "train" else 1))
        rng.shuffle(selected)
        selected = selected[:cap]
    if len(selected) < cap:
        raise RuntimeError(f"Only found {len(selected)}/{cap} eligible {split} targets after {scanned} scenarios")
    print(f"[{split}] selected={len(selected)} scanned={scanned}", flush=True)
    return selected, scanned


def balance_dense(pool: list[dict], cap: int, minimum: int, seed: int):
    rng = np.random.default_rng(seed)
    raw = Counter(item["raw_cohort"] for item in pool)
    split_s3 = raw["S3"] >= minimum
    groups = ("S0", "S1", "S2", "S3") if split_s3 else ("S0", "S1", "S2+")
    buckets = {key: [] for key in groups}
    for item in pool:
        cohort = item["raw_cohort"]
        if not split_s3 and cohort in ("S2", "S3"):
            cohort = "S2+"
        record = dict(item)
        record["dense_cohort"] = cohort
        buckets[cohort].append(record)
    for values in buckets.values():
        rng.shuffle(values)
    missing = {key: len(values) for key, values in buckets.items() if len(values) < minimum}
    if missing:
        raise RuntimeError(f"Dense cohort minimum {minimum} not met: {missing}")
    output = []
    cursor = {key: 0 for key in groups}
    while len(output) < cap:
        progressed = False
        for key in groups:
            if cursor[key] < len(buckets[key]) and len(output) < cap:
                output.append(buckets[key][cursor[key]])
                cursor[key] += 1
                progressed = True
        if not progressed:
            break
    rng.shuffle(output)
    return output


def motion_tags(points: np.ndarray) -> dict[str, bool | float]:
    velocity = np.diff(points, axis=0) * 10.0
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration = np.diff(velocity, axis=0) * 10.0
    acceleration_norm = np.linalg.norm(acceleration, axis=-1)
    recent = velocity[-16:]
    valid = np.linalg.norm(recent, axis=-1) > 0.15
    headings = np.unwrap(np.arctan2(recent[:, 1], recent[:, 0]))
    heading_change = float(np.abs(headings[valid][-1] - headings[valid][0])) if valid.sum() >= 2 else 0.0
    cross = velocity[:-1, 0] * velocity[1:, 1] - velocity[:-1, 1] * velocity[1:, 0]
    denom = np.maximum(np.linalg.norm(velocity[:-1], axis=-1) ** 3, 1.0e-4)
    curvature = float(np.percentile(np.abs(cross) / denom, 90)) if len(cross) else 0.0
    return {
        "mean_speed_mps": float(speed.mean()),
        "p90_acceleration_mps2": float(np.percentile(acceleration_norm, 90)) if len(acceleration_norm) else 0.0,
        "heading_change_rad": heading_change,
        "curvature_p90": curvature,
        "near_stationary": bool(speed[-10:].mean() < 0.20),
        "accelerating": bool(len(acceleration_norm) and np.percentile(acceleration_norm, 90) >= 0.75),
        "turning": bool(heading_change >= np.deg2rad(20.0)),
        "high_curvature": bool(curvature >= 0.35),
    }


def build_master(dataset_root: Path, split: str, manifest: list[dict], history_indices: tuple[int, ...]):
    by_scenario = defaultdict(list)
    for index, item in enumerate(manifest):
        by_scenario[item["scenario_id"]].append((index, item))
    city_names = sorted({item["city_name"] for item in manifest})
    city_ids = {name: index + 1 for index, name in enumerate(city_names)}
    history_count = len(history_indices)
    required_context = set(history_indices)
    master = np.zeros((len(manifest), MAX_SOURCE_AGENTS, history_count + FUTURE_STEPS, 3), dtype=np.float32)
    counts = np.ones(len(manifest), dtype=np.int64)
    metadata = [None] * len(manifest)
    for scenario_number, (scenario_id, targets) in enumerate(by_scenario.items(), start=1):
        path = dataset_root / split / scenario_id / f"scenario_{scenario_id}.parquet"
        frame = read_frame(path)
        pedestrians = frame[frame["object_type"].str.lower() == "pedestrian"]
        tracks = {
            str(track_id): indexed_track(group)
            for track_id, group in pedestrians.groupby("track_id", sort=False)
        }
        context = [track_id for track_id, points in tracks.items() if required_context.issubset(points)]
        for output_index, item in targets:
            target_id = item["track_id"]
            target = tracks[target_id]
            current = target[49]
            neighbors = [key for key in context if key != target_id]
            neighbors.sort(key=lambda key: float(np.linalg.norm(tracks[key][49] - current)))
            agent_ids = [target_id] + neighbors[:MAX_SOURCE_AGENTS - 1]
            counts[output_index] = len(agent_ids)
            city_id = float(city_ids[item["city_name"]])
            master[output_index, :, :, 0] = city_id
            future_valid = []
            for agent_index, track_id in enumerate(agent_ids):
                points = tracks[track_id]
                history = np.stack([points[t] for t in history_indices])
                master[output_index, agent_index, :history_count, 1:] = history
                valid_future = set(FUTURE_INDICES).issubset(points)
                future_valid.append(bool(valid_future))
                future = (
                    np.stack([points[t] for t in FUTURE_INDICES])
                    if valid_future else np.repeat(history[-1:], FUTURE_STEPS, axis=0)
                )
                master[output_index, agent_index, history_count:, 1:] = future
            metadata[output_index] = {
                **item,
                **motion_tags(np.stack([target[t] for t in history_indices])),
                "stored_agents": int(len(agent_ids)),
                "future_valid_agents": future_valid,
            }
        if scenario_number % 500 == 0:
            print(f"[{split}] materialized {scenario_number}/{len(by_scenario)} scenarios", flush=True)
    return master, counts, metadata


def save_group(output_root: Path, group: str, split: str, master: np.ndarray,
               counts: np.ndarray, metadata: list[dict]):
    prepared = output_root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    names = [name for name, spec in VARIANTS.items() if spec["data_source"] == group]
    for name in names:
        spec = VARIANTS[name]
        agents = int(spec["max_agents"])
        steps = int(spec["history_steps"])
        history_count = master.shape[2] - FUTURE_STEPS
        cropped = np.concatenate((master[:, :agents, history_count - steps:history_count], master[:, :agents, history_count:]), axis=2)
        np.save(prepared / f"{split}_{name}_data.npy", cropped)
        np.save(prepared / f"{split}_{name}_num.npy", np.minimum(counts, agents))
    payload = {"group": group, "split": split, "samples": metadata}
    (prepared / f"{split}_{group}_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    diagnostic = {
        "group": group,
        "split": split,
        "num_targets": len(metadata),
        "unique_scenarios": len({item["scenario_id"] for item in metadata}),
        "neighbor_cohorts": dict(Counter(item.get("dense_cohort", item["raw_cohort"]) for item in metadata)),
        "mean_neighbor_count_6m": float(np.mean([item["valid_neighbor_count"] for item in metadata])),
        "p90_neighbor_count_6m": float(np.percentile([item["valid_neighbor_count"] for item in metadata], 90)),
        "n9_truncation_rate": float(np.mean([item["valid_neighbor_count"] >= 9 for item in metadata])),
        "motion_cohorts": {
            key: int(sum(bool(item[key]) for item in metadata))
            for key in ("near_stationary", "accelerating", "turning", "high_curvature")
        },
    }
    (prepared / f"{split}_{group}_diagnostics.json").write_text(
        json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return diagnostic


def main():
    args = parse_args()
    manifest_dir = args.output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "future_indices": list(FUTURE_INDICES), "groups": {}}
    for group in args.groups:
        summary["groups"][group] = {}
        for split in ("train", "val"):
            cap = getattr(args, f"{group}_{split}_cap")
            manifest_path = manifest_dir / f"exp196_{group}_{split}_targets.json"
            if manifest_path.is_file() and not args.force:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                scanned = None
            else:
                manifest, scanned = scan_manifest(
                    args.dataset_root, split, cap, args.seed, args.workers,
                    args.max_scenarios, group, args.dense_min_per_cohort,
                )
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            history_indices = DENSE_HISTORY if group == "dense" else FULL_HISTORY
            master, counts, metadata = build_master(args.dataset_root, split, manifest, history_indices)
            diagnostic = save_group(args.output_root, group, split, master, counts, metadata)
            summary["groups"][group][split] = {"scanned_scenarios": scanned, **diagnostic}
    (args.output_root / "prepared" / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
