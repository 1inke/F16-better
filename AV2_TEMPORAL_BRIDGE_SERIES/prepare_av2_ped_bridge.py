"""Prepare deterministic pedestrian-only AV2 arrays for EXP-195A.

Run this script with a Python environment containing pyarrow and pandas. The
training environment only needs NumPy/PyTorch after these arrays are created.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from exp195_constants import CONFIGS, FUTURE_STEPS, MAX_AGENTS, ROOT


TARGET_CATEGORIES = {2: "SCORED_TRACK", 3: "FOCAL_TRACK"}
HISTORY_INDICES = {
    "a1": np.arange(21, 50, 4, dtype=np.int64),
    "a2": np.arange(19, 50, 2, dtype=np.int64),
    "a3": np.arange(18, 50, 1, dtype=np.int64),
}
# Observed current is t=49. These labels are +0.4, +0.8, ..., +4.8 seconds.
FUTURE_INDICES = np.arange(53, 98, 4, dtype=np.int64)
REQUIRED_TARGET = set(HISTORY_INDICES["a3"].tolist() + FUTURE_INDICES.tolist())
REQUIRED_CONTEXT = set(HISTORY_INDICES["a3"].tolist())
COLUMNS = [
    "track_id", "object_type", "object_category", "timestep",
    "position_x", "position_y", "scenario_id", "city",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "av2_dataset")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--train-cap", type=int, default=8000)
    parser.add_argument("--val-cap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 scans until cap")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parquet_path(dataset_root: Path, split: str, scenario_id: str) -> Path:
    return dataset_root / split / scenario_id / f"scenario_{scenario_id}.parquet"


def read_frame(path: Path):
    return pq.read_table(path, columns=COLUMNS).to_pandas()


def complete_track(group, required: set[int]) -> bool:
    return required.issubset(set(group["timestep"].astype(int).tolist()))


def inspect_scenario(directory: Path):
    path = directory / f"scenario_{directory.name}.parquet"
    if not path.is_file():
        return directory.name, "", []
    frame = read_frame(path)
    pedestrians = frame[frame["object_type"].str.lower() == "pedestrian"]
    candidates = []
    for track_id, track in pedestrians.groupby("track_id", sort=False):
        category = int(track["object_category"].iloc[0])
        if category in TARGET_CATEGORIES and complete_track(track, REQUIRED_TARGET):
            candidates.append((str(track_id), category))
    city = str(frame["city"].iloc[0]) if len(frame) else "unknown"
    return directory.name, city, candidates


def select_manifest(dataset_root: Path, split: str, cap: int, seed: int,
                    max_scenarios: int, workers: int):
    directories = [item for item in (dataset_root / split).iterdir() if item.is_dir()]
    rng = np.random.default_rng(seed + (0 if split == "train" else 1))
    rng.shuffle(directories)
    selected, scanned = [], 0
    limit = min(len(directories), max_scenarios) if max_scenarios else len(directories)
    batch_size = max(64, max(1, workers) * 16)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in range(0, limit, batch_size):
            chunk = directories[start : min(start + batch_size, limit)]
            for scenario_id, city, candidates in pool.map(inspect_scenario, chunk):
                scanned += 1
                rng.shuffle(candidates)
                for track_id, category in candidates:
                    selected.append({
                        "scenario_id": scenario_id,
                        "track_id": track_id,
                        "object_type": "PEDESTRIAN",
                        "category": TARGET_CATEGORIES[category],
                        "city_name": city,
                    })
                    if len(selected) >= cap:
                        break
                if scanned % 1000 == 0:
                    print(f"[{split}] scanned={scanned} targets={len(selected)}/{cap}", flush=True)
                if len(selected) >= cap:
                    break
            if len(selected) >= cap:
                break
    if not selected:
        raise RuntimeError(f"No eligible pedestrian targets found in {dataset_root / split}")
    print(f"[{split}] selected={len(selected)} after scanning {scanned} scenarios", flush=True)
    return selected, scanned


def indexed_track(group) -> dict[int, np.ndarray]:
    return {
        int(row.timestep): np.asarray([row.position_x, row.position_y], dtype=np.float32)
        for row in group.itertuples(index=False)
    }


def build_arrays(dataset_root: Path, split: str, manifest: list[dict], output_root: Path):
    by_scenario = defaultdict(list)
    for index, item in enumerate(manifest):
        by_scenario[item["scenario_id"]].append((index, item))
    city_names = sorted({item["city_name"] for item in manifest})
    city_ids = {name: index + 1 for index, name in enumerate(city_names)}
    arrays = {
        name: np.zeros(
            (len(manifest), MAX_AGENTS, spec["history_steps"] + FUTURE_STEPS, 3),
            dtype=np.float32,
        ) for name, spec in CONFIGS.items()
    }
    counts = np.ones(len(manifest), dtype=np.int64)
    neighbor_counts, truncations, speeds, accelerations = [], [], [], []

    for scenario_number, (scenario_id, targets) in enumerate(by_scenario.items(), start=1):
        frame = read_frame(parquet_path(dataset_root, split, scenario_id))
        pedestrians = frame[frame["object_type"].str.lower() == "pedestrian"]
        tracks = {
            str(track_id): indexed_track(group)
            for track_id, group in pedestrians.groupby("track_id", sort=False)
        }
        context_ids = [
            track_id for track_id, points in tracks.items()
            if REQUIRED_CONTEXT.issubset(points)
        ]
        for output_index, item in targets:
            target_id = item["track_id"]
            target = tracks[target_id]
            current = target[49]
            neighbors = [track_id for track_id in context_ids if track_id != target_id]
            neighbors.sort(key=lambda key: float(np.linalg.norm(tracks[key][49] - current)))
            selected_neighbors = neighbors[: MAX_AGENTS - 1]
            agent_ids = [target_id] + selected_neighbors
            counts[output_index] = len(agent_ids)
            within_radius = sum(
                np.linalg.norm(tracks[key][49] - current) < 6.0 for key in neighbors
            )
            neighbor_counts.append(int(within_radius))
            truncations.append(int(len(neighbors) > MAX_AGENTS - 1))

            dense_target = np.stack([target[t] for t in HISTORY_INDICES["a3"]])
            velocity = np.diff(dense_target, axis=0) * 10.0
            acceleration = np.diff(velocity, axis=0) * 10.0
            speeds.extend(np.linalg.norm(velocity, axis=-1).tolist())
            accelerations.extend(np.linalg.norm(acceleration, axis=-1).tolist())

            for name, history_indices in HISTORY_INDICES.items():
                array = arrays[name]
                city_id = float(city_ids[item["city_name"]])
                array[output_index, :, :, 0] = city_id
                history_steps = len(history_indices)
                for agent_index, track_id in enumerate(agent_ids):
                    points = tracks[track_id]
                    history = np.stack([points[int(t)] for t in history_indices])
                    array[output_index, agent_index, :history_steps, 1:] = history
                    if agent_index == 0:
                        future = np.stack([points[int(t)] for t in FUTURE_INDICES])
                    else:
                        # Context futures are never inputs and never enter the focal-only loss.
                        future = np.repeat(history[-1:], FUTURE_STEPS, axis=0)
                    array[output_index, agent_index, history_steps:, 1:] = future
        if scenario_number % 500 == 0:
            print(f"[{split}] materialized {scenario_number}/{len(by_scenario)} scenarios", flush=True)

    data_dir = output_root / "prepared"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(data_dir / f"{split}_{name}_data.npy", array)
        np.save(data_dir / f"{split}_{name}_num.npy", counts)
    diagnostic = {
        "split": split,
        "num_targets": len(manifest),
        "missing_history_rate": 0.0,
        "missing_future_rate": 0.0,
        "mean_speed_mps": float(np.mean(speeds)),
        "p50_speed_mps": float(np.percentile(speeds, 50)),
        "p90_speed_mps": float(np.percentile(speeds, 90)),
        "mean_acceleration_mps2": float(np.mean(accelerations)),
        "p90_acceleration_mps2": float(np.percentile(accelerations, 90)),
        "mean_neighbor_count_6m": float(np.mean(neighbor_counts)),
        "p50_neighbor_count_6m": float(np.percentile(neighbor_counts, 50)),
        "p90_neighbor_count_6m": float(np.percentile(neighbor_counts, 90)),
        "zero_neighbor_rate": float(np.mean(np.asarray(neighbor_counts) == 0)),
        "one_neighbor_rate": float(np.mean(np.asarray(neighbor_counts) == 1)),
        "max_neighbor_hit_rate": float(np.mean(np.asarray(neighbor_counts) >= 6)),
        "n9_truncation_rate": float(np.mean(truncations)),
    }
    (data_dir / f"{split}_diagnostics.json").write_text(
        json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return diagnostic


def main():
    args = parse_args()
    manifest_dir = args.output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "future_indices": FUTURE_INDICES.tolist(), "splits": {}}
    for split, cap in (("train", args.train_cap), ("val", args.val_cap)):
        manifest_path = manifest_dir / f"{split}_targets_exp195.json"
        if manifest_path.is_file() and not args.force:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            scanned = None
        else:
            manifest, scanned = select_manifest(
                args.dataset_root, split, cap, args.seed, args.max_scenarios, args.workers
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        diagnostic = build_arrays(args.dataset_root, split, manifest, args.output_root)
        summary["splits"][split] = {"scanned_scenarios": scanned, **diagnostic}
    (args.output_root / "prepared" / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
