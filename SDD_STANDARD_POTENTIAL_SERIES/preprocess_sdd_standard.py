"""Build the common PECNet/Y-Net SDD split for the current trajectory model."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# Recovered from the public PECNet/Y-Net SDD trajectory release.  The 31/12/17
# video split covers all 60 videos and is kept explicit for reproducibility.
TRAIN_VIDEOS = (
    "bookstore/video0", "bookstore/video1", "bookstore/video2", "bookstore/video3",
    "coupa/video3",
    "deathCircle/video0", "deathCircle/video1", "deathCircle/video2",
    "deathCircle/video3", "deathCircle/video4",
    "gates/video0", "gates/video1", "gates/video3", "gates/video4",
    "gates/video5", "gates/video6", "gates/video7", "gates/video8",
    "hyang/video5", "hyang/video6", "hyang/video7", "hyang/video9", "hyang/video14",
    "nexus/video1", "nexus/video2", "nexus/video3", "nexus/video4",
    "nexus/video7", "nexus/video8", "nexus/video9", "nexus/video10",
)
VAL_VIDEOS = (
    "bookstore/video4", "bookstore/video5", "bookstore/video6",
    "coupa/video2",
    "hyang/video2", "hyang/video3", "hyang/video4", "hyang/video10",
    "hyang/video11", "hyang/video12",
    "nexus/video0", "nexus/video11",
)
TEST_VIDEOS = (
    "coupa/video0", "coupa/video1", "gates/video2",
    "hyang/video0", "hyang/video1", "hyang/video8", "hyang/video13",
    "little/video0", "little/video1", "little/video2", "little/video3",
    "nexus/video5", "nexus/video6",
    "quad/video0", "quad/video1", "quad/video2", "quad/video3",
)


def read_pedestrian_tracks(path: Path) -> dict[int, dict[int, np.ndarray]]:
    tracks: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            fields = line.split(maxsplit=9)
            if len(fields) < 10:
                continue
            if fields[9].strip().strip('"').lower() != "pedestrian":
                continue
            if int(fields[6]) != 0:  # lost
                continue
            track_id, frame = int(fields[0]), int(fields[5])
            tracks[track_id][frame] = np.asarray(
                [
                    (float(fields[1]) + float(fields[3])) * 0.5,
                    (float(fields[2]) + float(fields[4])) * 0.5,
                ],
                dtype=np.float32,
            )
    return dict(tracks)


def continuous_segments(points: dict[int, np.ndarray]) -> list[list[int]]:
    frames = sorted(points)
    if not frames:
        return []
    segments, current = [], [frames[0]]
    for frame in frames[1:]:
        if frame == current[-1] + 1:
            current.append(frame)
        else:
            segments.append(current)
            current = [frame]
    segments.append(current)
    return segments


def focal_windows(
    tracks: dict[int, dict[int, np.ndarray]],
    frame_step: int,
    sequence_steps: int,
    window_stride_steps: int,
) -> list[tuple[int, list[int]]]:
    windows = []
    for track_id, points in tracks.items():
        for segment in continuous_segments(points):
            sampled = segment[::frame_step]
            for start in range(0, len(sampled) - sequence_steps + 1, window_stride_steps):
                windows.append((track_id, sampled[start : start + sequence_steps]))
    return windows


def evenly_limit(items: list, maximum: int) -> list:
    if maximum <= 0 or len(items) <= maximum:
        return items
    indices = np.linspace(0, len(items) - 1, maximum).round().astype(np.int64)
    return [items[int(index)] for index in indices]


def build_video(
    tracks: dict[int, dict[int, np.ndarray]],
    split_id: int,
    frame_step: int,
    obs_steps: int,
    pred_steps: int,
    window_stride_steps: int,
    max_agents: int,
    pixel_scale: float,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    sequence_steps = obs_steps + pred_steps
    all_windows = focal_windows(
        tracks, frame_step, sequence_steps, window_stride_steps
    )
    windows = evenly_limit(all_windows, max_samples)
    data = np.zeros((len(windows), max_agents, sequence_steps, 6), np.float32)
    counts = np.zeros(len(windows), np.int32)
    neighbor_counts = []

    for sample_index, (focal_id, frames) in enumerate(windows):
        observed_frames = frames[:obs_steps]
        focal_last = tracks[focal_id][observed_frames[-1]]
        neighbors = []
        for track_id, points in tracks.items():
            if track_id == focal_id or not all(frame in points for frame in observed_frames):
                continue
            distance = float(np.linalg.norm(points[observed_frames[-1]] - focal_last))
            neighbors.append((distance, track_id))
        neighbors.sort()
        ordered = [focal_id] + [track_id for _, track_id in neighbors[: max_agents - 1]]
        counts[sample_index] = len(ordered)
        neighbor_counts.append(len(neighbors))

        for agent_index, track_id in enumerate(ordered):
            points = tracks[track_id]
            last = points[observed_frames[-1]]
            trajectory = []
            for frame in frames:
                if frame in points:
                    last = points[frame]
                trajectory.append(last)
            trajectory_px = np.stack(trajectory)
            data[sample_index, agent_index, :, 0] = float(split_id)
            data[sample_index, agent_index, :, 1] = float(track_id)
            data[sample_index, agent_index, :, 2:4] = -1.0
            data[sample_index, agent_index, :, 4:6] = trajectory_px / pixel_scale

    return data, counts, {
        "pedestrian_tracks": len(tracks),
        "eligible_windows": len(all_windows),
        "samples": len(windows),
        "mean_valid_agents": float(counts.mean()) if len(counts) else 0.0,
        "max_valid_agents": int(counts.max()) if len(counts) else 0,
        "mean_available_neighbors": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
    }


def build_split(
    dataset_root: Path,
    videos: tuple[str, ...],
    split_name: str,
    split_id: int,
    max_agents: int,
    pixel_scale: float,
    max_samples_per_video: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    data_parts, count_parts, stats = [], [], []
    for video in videos:
        annotation = dataset_root / "annotations" / video / "annotations.txt"
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        tracks = read_pedestrian_tracks(annotation)
        data, counts, video_stats = build_video(
            tracks, split_id=split_id, frame_step=12, obs_steps=8, pred_steps=12,
            window_stride_steps=20, max_agents=max_agents,
            pixel_scale=pixel_scale, max_samples=max_samples_per_video,
        )
        video_stats.update({"video": video, "split": split_name})
        print(f"[{split_name}] {video}: {len(data)} samples", flush=True)
        data_parts.append(data)
        count_parts.append(counts)
        stats.append(video_stats)
    return np.concatenate(data_parts), np.concatenate(count_parts), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-agents", type=int, default=16)
    parser.add_argument("--pixel-scale", type=float, default=20.0)
    parser.add_argument("--max-samples-per-video", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and not args.rebuild:
        print(f"[skip] {manifest_path}")
        return

    split_data = {}
    all_stats = []
    for name, split_id, videos in (
        ("train", 0, TRAIN_VIDEOS), ("val", 1, VAL_VIDEOS),
        ("test", 2, TEST_VIDEOS),
    ):
        data, counts, stats = build_split(
            args.dataset_root.resolve(), videos, name, split_id,
            args.max_agents, args.pixel_scale, args.max_samples_per_video,
        )
        split_data[name] = (data, counts)
        all_stats.extend(stats)

    output.mkdir(parents=True, exist_ok=True)
    train_val_data = np.concatenate((split_data["train"][0], split_data["val"][0]))
    train_val_num = np.concatenate((split_data["train"][1], split_data["val"][1]))
    np.save(output / "sdd_standard_train_val_data.npy", train_val_data)
    np.save(output / "sdd_standard_train_val_num.npy", train_val_num)
    np.save(output / "sdd_standard_test_data.npy", split_data["test"][0])
    np.save(output / "sdd_standard_test_num.npy", split_data["test"][1])

    counts = {name: int(len(values[0])) for name, values in split_data.items()}
    manifest = {
        "dataset_root": str(args.dataset_root.resolve()),
        "protocol": "PECNet/Y-Net 31/12/17 video split; Pedestrian only; lost=0",
        "sampling": "30 fps -> 2.5 Hz (step 12), 8 observed + 12 predicted",
        "window_stride_steps": 20,
        "metric_unit": "raw pixels after multiplying model errors by pixel_scale",
        "pixel_scale": args.pixel_scale,
        "max_agents": args.max_agents,
        "max_samples_per_video": args.max_samples_per_video,
        "split_counts": counts,
        "videos": {
            "train": list(TRAIN_VIDEOS), "val": list(VAL_VIDEOS),
            "test": list(TEST_VIDEOS),
        },
        "video_stats": all_stats,
        "comparison_note": (
            "The scene split and trajectory definition match the public protocol. "
            "Neighbor packing is model-specific, so the result is comparable as an "
            "approximate benchmark rather than a bit-exact reproduction of another loader."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
