"""Build a small SDD trajectory archive compatible with the ASTRA trainer."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_VIDEOS = ("bookstore/video0", "bookstore/video1", "bookstore/video2")


def read_pedestrian_tracks(annotation_path: Path) -> dict[int, dict[int, np.ndarray]]:
    tracks: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split(maxsplit=9)
            if len(fields) < 10:
                continue
            label = fields[9].strip().strip('"').lower()
            if label != "pedestrian" or int(fields[6]) != 0:
                continue
            track_id = int(fields[0])
            frame = int(fields[5])
            center = np.asarray(
                (
                    0.5 * (float(fields[1]) + float(fields[3])),
                    0.5 * (float(fields[2]) + float(fields[4])),
                ),
                dtype=np.float32,
            )
            tracks[track_id][frame] = center
    return dict(tracks)


def candidate_windows(
    tracks: dict[int, dict[int, np.ndarray]],
    frame_step: int,
    sequence_steps: int,
    window_stride_steps: int,
) -> list[tuple[int, int]]:
    span = frame_step * (sequence_steps - 1)
    stride_frames = frame_step * window_stride_steps
    candidates: list[tuple[int, int]] = []
    for track_id, points in tracks.items():
        frames = sorted(points)
        if not frames:
            continue
        first, last = frames[0], frames[-1]
        for start in range(first, last - span + 1, stride_frames):
            required = range(start, start + span + 1, frame_step)
            if all(frame in points for frame in required):
                candidates.append((track_id, start))
    return candidates


def evenly_limit(
    candidates: list[tuple[int, int]], maximum: int
) -> list[tuple[int, int]]:
    if maximum <= 0 or len(candidates) <= maximum:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, maximum).round().astype(np.int64)
    return [candidates[int(index)] for index in indices]


def build_video_samples(
    tracks: dict[int, dict[int, np.ndarray]],
    scene_id: int,
    frame_step: int,
    obs_steps: int,
    pred_steps: int,
    window_stride_steps: int,
    max_agents: int,
    pixel_scale: float,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    sequence_steps = obs_steps + pred_steps
    candidates = candidate_windows(
        tracks, frame_step, sequence_steps, window_stride_steps
    )
    candidates = evenly_limit(candidates, max_samples)
    samples = np.zeros(
        (len(candidates), max_agents, sequence_steps, 6), dtype=np.float32
    )
    counts = np.zeros(len(candidates), dtype=np.int32)
    step_distances = []

    for sample_index, (focal_id, start) in enumerate(candidates):
        frames = [start + frame_step * step for step in range(sequence_steps)]
        present = [
            track_id
            for track_id, points in tracks.items()
            if all(frame in points for frame in frames)
        ]
        focal_position = tracks[focal_id][frames[obs_steps - 1]]
        neighbors = [track_id for track_id in present if track_id != focal_id]
        neighbors.sort(
            key=lambda track_id: float(
                np.linalg.norm(
                    tracks[track_id][frames[obs_steps - 1]] - focal_position
                )
            )
        )
        ordered = [focal_id] + neighbors[: max_agents - 1]
        counts[sample_index] = len(ordered)
        for agent_index, track_id in enumerate(ordered):
            trajectory_px = np.stack([tracks[track_id][frame] for frame in frames])
            samples[sample_index, agent_index, :, 0] = float(scene_id)
            samples[sample_index, agent_index, :, 1] = float(track_id)
            samples[sample_index, agent_index, :, 2:4] = -1.0
            samples[sample_index, agent_index, :, 4:6] = trajectory_px / pixel_scale
            step_distances.extend(
                np.linalg.norm(np.diff(trajectory_px, axis=0), axis=-1).tolist()
            )

    stats = {
        "tracks": len(tracks),
        "eligible_windows_before_limit": len(candidate_windows(
            tracks, frame_step, sequence_steps, window_stride_steps
        )),
        "samples": len(candidates),
        "mean_agents": float(counts.mean()) if len(counts) else 0.0,
        "max_agents_observed": int(counts.max()) if len(counts) else 0,
        "sampled_step_pixels": {
            "median": float(np.median(step_distances)) if step_distances else None,
            "p90": float(np.quantile(step_distances, 0.90)) if step_distances else None,
            "p99": float(np.quantile(step_distances, 0.99)) if step_distances else None,
        },
    }
    return samples, counts, stats


def build_dataset(
    dataset_root: Path,
    output_dir: Path,
    videos: tuple[str, ...] = DEFAULT_VIDEOS,
    frame_step: int = 12,
    obs_steps: int = 8,
    pred_steps: int = 12,
    window_stride_steps: int = 4,
    max_agents: int = 16,
    pixel_scale: float = 20.0,
    max_samples_per_video: int = 600,
) -> dict:
    all_samples, all_counts = [], []
    video_stats = []
    for scene_id, video in enumerate(videos):
        annotation = dataset_root / "annotations" / video / "annotations.txt"
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        print(f"[parse] {video}", flush=True)
        tracks = read_pedestrian_tracks(annotation)
        samples, counts, stats = build_video_samples(
            tracks=tracks,
            scene_id=scene_id,
            frame_step=frame_step,
            obs_steps=obs_steps,
            pred_steps=pred_steps,
            window_stride_steps=window_stride_steps,
            max_agents=max_agents,
            pixel_scale=pixel_scale,
            max_samples=max_samples_per_video,
        )
        stats["scene_id"] = scene_id
        stats["video"] = video
        video_stats.append(stats)
        all_samples.append(samples)
        all_counts.append(counts)

    data = np.concatenate(all_samples, axis=0)
    counts = np.concatenate(all_counts, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "sdd_bookstore_lowcost_data.npy"
    num_path = output_dir / "sdd_bookstore_lowcost_num.npy"
    np.save(data_path, data)
    np.save(num_path, counts)
    manifest = {
        "dataset_root": str(dataset_root),
        "videos": list(videos),
        "validation_scene_id": 2,
        "frame_step": frame_step,
        "sampling_hz_assuming_30fps": 30.0 / frame_step,
        "obs_steps": obs_steps,
        "pred_steps": pred_steps,
        "window_stride_steps": window_stride_steps,
        "max_agents": max_agents,
        "pixel_scale": pixel_scale,
        "max_samples_per_video": max_samples_per_video,
        "total_samples": int(len(data)),
        "data_shape": list(data.shape),
        "data_path": str(data_path),
        "num_path": str(num_path),
        "video_stats": video_stats,
        "scope_warning": (
            "Low-cost same-location video split; not the official SDD benchmark split. "
            "Metrics are normalized units and pixel equivalents, not meter errors."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[done] data={data_path} samples={len(data)}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples-per-video", type=int, default=600)
    parser.add_argument("--max-agents", type=int, default=16)
    parser.add_argument("--pixel-scale", type=float, default=20.0)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.is_file() and not args.rebuild:
        print(f"[skip] {manifest_path}")
        return
    build_dataset(
        dataset_root=args.dataset_root.resolve(),
        output_dir=args.output_dir.resolve(),
        max_agents=args.max_agents,
        pixel_scale=args.pixel_scale,
        max_samples_per_video=args.max_samples_per_video,
    )


if __name__ == "__main__":
    main()
