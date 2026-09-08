"""Comparable CUDA efficiency profile for trained A1/A2/A3 checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from exp195_common import CONFIGS, ROOT, build_model, model_forward
from materialize_configs import materialize


OUT = ROOT / "EXP_195A_SHAPE_PRESERVING_BRIDGE"


def checkpoint_for(name: str) -> Path:
    return ROOT / "results" / name / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"


def batch(name: str, size: int, device: torch.device):
    spec = CONFIGS[name]
    raw = np.load(ROOT / "prepared" / f"val_{name}_data.npy", mmap_mode="r")
    num = np.load(ROOT / "prepared" / f"val_{name}_num.npy", mmap_mode="r")
    past = np.asarray(raw[:size, :, : spec["history_steps"], -2:], dtype=np.float32)
    return torch.from_numpy(past.copy()).to(device), torch.from_numpy(np.asarray(num[:size]).copy()).to(device)


def latency(model, past, num, warmup: int, timed_forwards: int, group: int):
    with torch.inference_mode():
        for _ in range(warmup):
            model_forward(model, past, num)
        torch.cuda.synchronize()
        values = []
        remaining = timed_forwards
        while remaining:
            count = min(group, remaining)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(count):
                model_forward(model, past, num)
            end.record(); torch.cuda.synchronize()
            values.append(float(start.elapsed_time(end)) / count)
            remaining -= count
    return values


def peak_memory(model, past, num):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with torch.inference_mode():
        model_forward(model, past, num); torch.cuda.synchronize()
    return int(max(0, torch.cuda.max_memory_allocated() - baseline))


def profiler_flops(model, past, num):
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True, record_shapes=False,
    ) as profile:
        model_forward(model, past, num); torch.cuda.synchronize()
    return int(sum(event.flops for event in profile.key_averages() if event.flops))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--timed-forwards", type=int, default=1000)
    parser.add_argument("--forwards-per-measure", type=int, default=20)
    parser.add_argument("--throughput-batch", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for comparable profiling")
    device = torch.device(args.device)
    configs = materialize()
    rows = []
    for name, spec in CONFIGS.items():
        checkpoint = checkpoint_for(name)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, _ = build_model(configs[name], checkpoint, device)
        model.eval()
        parameters = sum(item.numel() for item in model.parameters())
        for size in (1, args.throughput_batch):
            past, num = batch(name, size, device)
            if args.smoke:
                with torch.inference_mode():
                    output, _ = model_forward(model, past, num)
                print(f"[smoke] {name} batch={size} output={tuple(output.shape)}")
                continue
            values = latency(
                model, past, num, args.warmup,
                args.timed_forwards, args.forwards_per_measure,
            )
            row = {
                "config": name, "history_hz": spec["history_hz"],
                "history_steps": spec["history_steps"], "batch_size": size,
                "parameters": parameters,
                "latency_ms_median": statistics.median(values),
                "latency_ms_p95": float(np.percentile(values, 95)),
                "latency_ms_mean": statistics.fmean(values),
                "throughput_targets_per_s": 1000.0 * size / statistics.fmean(values),
                "peak_gpu_memory_bytes": peak_memory(model, past, num),
                "timed_forwards": args.timed_forwards,
                "forwards_per_measure": args.forwards_per_measure,
            }
            if size == 1:
                row["profiler_accounted_flops"] = profiler_flops(model, past, num)
                row["profiler_accounted_macs"] = row["profiler_accounted_flops"] / 2.0
            rows.append(row)
            print(f"[profiled] {name} batch={size} median={row['latency_ms_median']:.3f} ms")
    if args.smoke:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (OUT / "efficiency.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    payload = {
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "protocol": vars(args), "rows": rows,
    }
    (OUT / "efficiency.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
