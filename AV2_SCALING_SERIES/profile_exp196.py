"""Interleaved repeated CUDA scaling profile for EXP-196."""
from __future__ import annotations

import argparse
import csv
import json
import statistics

import numpy as np
import torch

from exp196_common import build_model, model_forward, temporal_weight_mass
from exp196_constants import ROOT, VARIANTS, checkpoint_for, config_for, data_paths
from materialize_configs import materialize


def make_batch(name: str, size: int, device: torch.device):
    spec = VARIANTS[name]
    data_path, num_path = data_paths(name, "val")
    raw = np.load(data_path, mmap_mode="r")
    num = np.load(num_path, mmap_mode="r")
    history = int(spec["history_steps"])
    past = np.asarray(raw[:size, :, :history, -2:], dtype=np.float32)
    return (
        torch.from_numpy(past.copy()).to(device),
        torch.from_numpy(np.minimum(np.asarray(num[:size]), int(spec["max_agents"]))).to(device),
    )


def latency(model, past, num, warmup: int, forwards: int, group: int):
    with torch.inference_mode():
        for _ in range(warmup):
            model_forward(model, past, num)
        torch.cuda.synchronize()
        values = []
        remaining = forwards
        while remaining:
            count = min(group, remaining)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(count):
                model_forward(model, past, num)
            end.record()
            torch.cuda.synchronize()
            values.append(float(start.elapsed_time(end)) / count)
            remaining -= count
    return values


def peak_memory(model, past, num):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with torch.inference_mode():
        model_forward(model, past, num)
        torch.cuda.synchronize()
    return int(max(0, torch.cuda.max_memory_allocated() - baseline))


def profiler_flops(model, past, num):
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True, record_shapes=False,
    ) as profile:
        model_forward(model, past, num)
        torch.cuda.synchronize()
    return int(sum(event.flops for event in profile.key_averages() if event.flops))


def write_outputs(rows: list[dict], payload: dict):
    folder = ROOT / "EXP_196_SCALING_PROFILE"
    folder.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (folder / "efficiency_scaling.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (folder / "efficiency_scaling.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def scaling_decision(rows: list[dict]) -> dict:
    large_batch = max(int(row["batch_size"]) for row in rows)
    lookup = {row["config"]: row for row in rows if int(row["batch_size"]) == large_batch}
    evidence = {}

    def increase(candidate: str, baseline: str, metric: str):
        if candidate not in lookup or baseline not in lookup:
            return None
        value = float(lookup[candidate][metric]) / max(float(lookup[baseline][metric]), 1.0e-12) - 1.0
        evidence[f"{candidate}_vs_{baseline}_{metric}"] = value
        return value

    conditions = []
    for candidate, baseline, metric in (
        ("l4", "l2", "profiler_accounted_flops"),
        ("l4", "l2", "peak_gpu_memory_bytes"),
        ("d4", "d2", "latency_ms_median_across_cycles"),
        ("j2", "d2", "latency_ms_median_across_cycles"),
        ("j2", "d2", "peak_gpu_memory_bytes"),
    ):
        value = increase(candidate, baseline, metric)
        if value is not None:
            conditions.append(value >= 0.30)
    accuracy_stable = False
    summary_path = ROOT / "exp196_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fair_seeds = {int(row["seed"]) for row in summary.get("rows", []) if row.get("stage") == "fair"}
        accuracy_stable = len(fair_seeds) >= 3
    proceed = accuracy_stable and sum(conditions) >= 2
    return {
        "large_batch": large_batch,
        "accuracy_stable_three_seed_fair": accuracy_stable,
        "resource_conditions_met": int(sum(conditions)),
        "evidence": evidence,
        "decision_label": "PROCEED_TO_MAMBA" if proceed else "MAMBA_NOT_YET_JUSTIFIED",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--timed-forwards", type=int, default=400)
    parser.add_argument("--forwards-per-measure", type=int, default=20)
    parser.add_argument("--throughput-batch", type=int, default=32)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for comparable profiling")
    materialize()
    device = torch.device(args.device)
    available = [name for name in args.variants if checkpoint_for(name, args.seed).is_file()]
    if not available:
        raise FileNotFoundError("No requested EXP-196 checkpoints exist")
    rng = np.random.default_rng(2026)
    samples, static = [], {}
    for cycle in range(args.cycles):
        order = list(available)
        rng.shuffle(order)
        for name in order:
            model, _ = build_model(config_for(name), checkpoint_for(name, args.seed), device)
            model.eval()
            if name not in static:
                one_past, one_num = make_batch(name, 1, device)
                with torch.inference_mode():
                    output, _ = model_forward(model, one_past, one_num)
                if args.smoke:
                    print(f"[smoke] {name}: {tuple(output.shape)}")
                    del model
                    torch.cuda.empty_cache()
                    continue
                flops = profiler_flops(model, one_past, one_num)
                static[name] = {
                    "parameters": sum(item.numel() for item in model.parameters()),
                    "profiler_accounted_flops": flops,
                    "profiler_accounted_macs": flops / 2.0,
                    **temporal_weight_mass(int(VARIANTS[name]["history_steps"]), float(VARIANTS[name]["history_hz"])),
                }
            if args.smoke:
                del model
                torch.cuda.empty_cache()
                continue
            for size in (1, args.throughput_batch):
                past, num = make_batch(name, size, device)
                values = latency(
                    model, past, num,
                    args.warmup if cycle == 0 else max(10, args.warmup // 5),
                    args.timed_forwards, args.forwards_per_measure,
                )
                samples.append({
                    "config": name, "cycle": cycle + 1, "batch_size": size,
                    "latency_ms_median": statistics.median(values),
                    "latency_ms_p95": float(np.percentile(values, 95)),
                    "latency_ms_mean": statistics.fmean(values),
                    "throughput_targets_per_s": 1000.0 * size / statistics.fmean(values),
                    "peak_gpu_memory_bytes": peak_memory(model, past, num),
                })
            del model
            torch.cuda.empty_cache()
    if args.smoke:
        return
    aggregate = []
    for name in available:
        for size in (1, args.throughput_batch):
            subset = [row for row in samples if row["config"] == name and row["batch_size"] == size]
            aggregate.append({
                "config": name, "batch_size": size,
                "history_steps": VARIANTS[name]["history_steps"],
                "max_agents": VARIANTS[name]["max_agents"],
                "graph_top_k": VARIANTS[name]["top_k"],
                "latency_ms_median_across_cycles": float(np.median([row["latency_ms_median"] for row in subset])),
                "latency_cycle_std": float(np.std([row["latency_ms_median"] for row in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "throughput_targets_per_s": float(np.mean([row["throughput_targets_per_s"] for row in subset])),
                "peak_gpu_memory_bytes": int(max(row["peak_gpu_memory_bytes"] for row in subset)),
                **static[name],
            })
    payload = {
        "gpu": torch.cuda.get_device_name(device),
        "protocol": vars(args),
        "aggregate": aggregate,
        "cycles": samples,
        "decision": scaling_decision(aggregate),
    }
    write_outputs(aggregate, payload)


if __name__ == "__main__":
    main()
