"""Correctness and CUDA runtime helpers for EXP-180B."""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def make_session(path: Path, provider: str):
    available = ort.get_available_providers()
    if provider not in available: raise RuntimeError(f"{provider} unavailable; providers={available}")
    options = ort.SessionOptions(); options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=[provider, "CPUExecutionProvider"])


def run_numpy(session, past: np.ndarray, valid: np.ndarray):
    return session.run(["candidates", "logits"], {"past": past.astype(np.float32), "num_valid": valid.astype(np.int64)})


def cuda_binding(session, past: np.ndarray, valid: np.ndarray):
    binding = session.io_binding()
    binding.bind_ortvalue_input("past", ort.OrtValue.ortvalue_from_numpy(past.astype(np.float32), "cuda", 0))
    binding.bind_ortvalue_input("num_valid", ort.OrtValue.ortvalue_from_numpy(valid.astype(np.int64), "cuda", 0))
    binding.bind_output("candidates", "cuda", 0); binding.bind_output("logits", "cuda", 0)
    return binding


def benchmark_ort(session, past, valid, batch_size, warmup, iterations):
    binding = cuda_binding(session, past, valid)
    for _ in range(warmup): session.run_with_iobinding(binding)
    binding.synchronize_outputs(); timings = []
    for _ in range(iterations):
        start = time.perf_counter_ns(); session.run_with_iobinding(binding); binding.synchronize_outputs()
        timings.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return summarize(timings, batch_size)


def benchmark_pytorch(wrapper, past, valid, device, batch_size, warmup, iterations):
    timings = []
    with torch.inference_mode():
        for _ in range(warmup): wrapper(past, valid)
        torch.cuda.synchronize(device)
        for _ in range(iterations):
            torch.cuda.synchronize(device); start = time.perf_counter_ns(); wrapper(past, valid)
            torch.cuda.synchronize(device); timings.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return summarize(timings, batch_size)


def summarize(timings, batch_size):
    values = np.asarray(timings, np.float64); mean = float(values.mean())
    return {
        "latency_mean_ms": mean, "latency_median_ms": float(np.median(values)),
        "latency_p90_ms": float(np.percentile(values, 90)), "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_p99_ms": float(np.percentile(values, 99)),
        "latency_std_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "throughput_samples_s": float(batch_size * 1000.0 / mean),
    }


def metric_bundle(prediction, future, logits):
    distance = np.linalg.norm(prediction - future[:, None], axis=-1); ade = distance.mean(-1); fde = distance[..., -1]
    rows = np.arange(len(ade)); top1 = logits.argmax(1)
    return {"minADE": float(ade.min(1).mean()), "minFDE": float(fde.min(1).mean()),
            "top1_ADE": float(ade[rows, top1].mean()), "top1_FDE": float(fde[rows, top1].mean())}
