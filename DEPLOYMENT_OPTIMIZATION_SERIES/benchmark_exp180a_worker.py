"""One fresh-process runtime repetition for EXP-180A."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


SERIES = Path(__file__).resolve().parent
ROOT = SERIES.parent
EFFICIENCY = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "run_exp174a_final_efficiency.py"
DENSE_HELPER = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "run_exp174b_no_retrain_runtime.py"
Q3KV_ADAPTER = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_174D_QUERY_REDUCED_TEMPORAL_SOCIAL" / "exp174d_encoder_adapter.py"
Q3KV_CHECKPOINT = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_174D_QUERY_REDUCED_TEMPORAL_SOCIAL" / "results" / "eth" / "v2" / "q3kv" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"
Q3KV_EXPERIMENT = "three_representative_temporal_social_kv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def build_models(device: torch.device):
    efficiency = load_module("exp180_efficiency", EFFICIENCY)
    dense = load_module("exp180_dense", DENSE_HELPER)
    adapter = load_module("exp180_q3kv_adapter", Q3KV_ADAPTER)
    baseline, cfg, trainer = efficiency.build_model("F16", 1, device, 9)
    q3kv, _, _ = efficiency.build_model("F16", 1, device, 9)
    adapter.replace_encoder(q3kv, cfg, Q3KV_EXPERIMENT)
    q3kv.load_state_dict(torch.load(Q3KV_CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    return {"F16_baseline": baseline.eval(), "Q3KV": q3kv.to(device).eval()}, cfg, dense


def benchmark(model, past, valid, device, warmup: int, iterations: int):
    timings = []
    with torch.inference_mode():
        probe = None
        for _ in range(warmup):
            center = past.mean(dim=-2, keepdim=True)
            probe = model(past - center, None, None, mode="testing", num_valid=valid, graph_loc=past)
        probe = probe[2] if isinstance(probe, (tuple, list)) else probe
        if probe is None or not torch.isfinite(probe).all():
            raise RuntimeError("Non-finite prediction before benchmark")
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        for _ in range(iterations):
            torch.cuda.synchronize(device); start = time.perf_counter_ns()
            center = past.mean(dim=-2, keepdim=True)
            model(past - center, None, None, mode="testing", num_valid=valid, graph_loc=past)
            torch.cuda.synchronize(device)
            timings.append((time.perf_counter_ns() - start) / 1_000_000.0)
    values = np.asarray(timings, np.float64); mean = float(values.mean())
    return {
        "latency_mean_ms": mean, "latency_median_ms": float(np.median(values)),
        "latency_p90_ms": float(np.percentile(values, 90)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_p99_ms": float(np.percentile(values, 99)),
        "latency_std_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "throughput_samples_s": float(len(past) * 1000.0 / mean),
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "warmup": warmup, "iterations": iterations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True); parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if not Q3KV_CHECKPOINT.is_file(): raise FileNotFoundError(Q3KV_CHECKPOINT)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("EXP-180A requires CUDA")
    torch.manual_seed(180000 + args.process_index); np.random.seed(180000 + args.process_index)
    models, cfg, dense = build_models(device)
    rows = []
    for batch_index, batch_size in enumerate(args.batch_sizes):
        past, valid = dense.dense_batch(cfg, batch_size, device)
        names = ["F16_baseline", "Q3KV"]
        if (args.process_index + batch_index) % 2: names.reverse()
        for name in names:
            print(f"[worker {args.process_index}] {name} batch={batch_size}", flush=True)
            row = {
                "process_index": args.process_index, "variant": name, "batch_size": batch_size,
                "parameters": sum(p.numel() for p in models[name].parameters()),
                "measurement_order": names.index(name),
            }
            row.update(benchmark(models[name], past, valid, device, args.warmup, args.iterations))
            rows.append(row)
    payload = {
        "experiment": "EXP-180A", "process_index": args.process_index,
        "pid": os.getpid(), "host": platform.node(), "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "cuda": torch.version.cuda, "precision": "FP32 eager",
        "checkpoint_protocol": "EXP173 F16 seed1 ETH vs separately-trained EXP174D Q3KV seed1 ETH",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[worker complete] {output}", flush=True)


if __name__ == "__main__":
    main()
