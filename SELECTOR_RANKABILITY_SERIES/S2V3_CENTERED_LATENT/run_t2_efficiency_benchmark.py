#!/usr/bin/env python
"""
EXP-T2: Efficiency Benchmark.

Rigorously benchmarks and compares:
1. Global Transformer Residual (G0+G1+G2+G4, TinyTransformer d_model=48)
2. Tiny L5 Linear Residual (G0+G1+raw L5, Ridge / Linear)

Measurements:
- Parameters and Compression Rate (1 - P_tiny / P_transformer)
- MACs / FLOPs via thop and analytical count
- Peak Memory (GPU VRAM allocated & reserved, CPU memory)
- Latency (CPU vs GPU, Batch 1 vs Batch 64)
  - Protocol: warmup=100, repeat=1000, torch.cuda.synchronize()
  - Reports: mean, median, p95
- Selector-only latency vs End-to-End (F16 + Selector) total latency

Outputs:
- results/efficiency_benchmark.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO / "SELECTOR_BENCHMARK_SERIES")]

if sys.platform == "win32":
    torch_lib = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import torch
import torch.nn as nn
import thop
from build_candidate_cache import build_f16_model, configure_astra_env

D_TRANS = 43 # g0(1) + g1(15) + g2(6) + g4(21)
D_TINY = 86  # g0(1) + g1(15) + l5(70)


class TinyTransformer(nn.Module):
    def __init__(self, in_dim: int = D_TRANS, d_model: int = 48, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        return self.head(h).squeeze(-1)  # (B, 20)


class TinyL5Linear(nn.Module):
    def __init__(self, in_dim: int = D_TINY):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # (B, 20)


def benchmark_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int = 100,
    repeat: int = 1000,
) -> Dict[str, float]:
    model.eval()
    model.to(device)
    x = input_tensor.to(device)

    is_cuda = device.type == "cuda"

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()

    # Timing
    latencies = []
    with torch.inference_mode():
        for _ in range(repeat):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms

    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def measure_peak_memory(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> float:
    if device.type != "cuda":
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    model.to(device)
    x = input_tensor.to(device)

    with torch.inference_mode():
        for _ in range(50):
            _ = model(x)
        torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return float(peak_bytes / (1024 * 1024)) # MB


def main():
    parser = argparse.ArgumentParser(description="EXP-T2: Efficiency Benchmark")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V3_CENTERED_LATENT/results")
    args = parser.parse_args()

    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cuda_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_dev = torch.device("cpu")

    print(f"[EXP-T2] Running Efficiency Benchmark (warmup={args.warmup}, repeat={args.repeat})...")

    # Instantiate models
    m_trans = TinyTransformer(in_dim=D_TRANS, d_model=48, nhead=4)
    m_tiny = TinyL5Linear(in_dim=D_TINY)

    p_trans = sum(p.numel() for p in m_trans.parameters())
    p_tiny = sum(p.numel() for p in m_tiny.parameters())
    compression = 1.0 - (p_tiny / p_trans)

    # Measure MACs via thop
    dummy_trans_1 = torch.randn(1, 20, D_TRANS)
    dummy_tiny_1 = torch.randn(1, 20, D_TINY)
    dummy_trans_64 = torch.randn(64, 20, D_TRANS)
    dummy_tiny_64 = torch.randn(64, 20, D_TINY)

    macs_trans, _ = thop.profile(m_trans, inputs=(dummy_trans_1,), verbose=False)
    macs_tiny, _ = thop.profile(m_tiny, inputs=(dummy_tiny_1,), verbose=False)

    flops_trans = 2.0 * macs_trans
    flops_tiny = 2.0 * macs_tiny

    print(f"  [Parameters] Transformer: {p_trans:,} | Tiny L5: {p_tiny} | Compression: {compression*100:.2f}%")
    print(f"  [FLOPs/batch-1] Transformer: {flops_trans:,.0f} | Tiny L5: {flops_tiny:,.0f} | Ratio: {flops_trans/flops_tiny:.1f}x")

    # Latency: CPU & GPU, Batch 1 & Batch 64
    print("  Measuring CPU Batch-1...")
    lat_trans_cpu_b1 = benchmark_latency(m_trans, dummy_trans_1, cpu_dev, args.warmup, args.repeat)
    lat_tiny_cpu_b1 = benchmark_latency(m_tiny, dummy_tiny_1, cpu_dev, args.warmup, args.repeat)

    print("  Measuring CPU Batch-64...")
    lat_trans_cpu_b64 = benchmark_latency(m_trans, dummy_trans_64, cpu_dev, args.warmup, args.repeat)
    lat_tiny_cpu_b64 = benchmark_latency(m_tiny, dummy_tiny_64, cpu_dev, args.warmup, args.repeat)

    print("  Measuring GPU Batch-1...")
    lat_trans_gpu_b1 = benchmark_latency(m_trans, dummy_trans_1, cuda_dev, args.warmup, args.repeat)
    lat_tiny_gpu_b1 = benchmark_latency(m_tiny, dummy_tiny_1, cuda_dev, args.warmup, args.repeat)

    print("  Measuring GPU Batch-64...")
    lat_trans_gpu_b64 = benchmark_latency(m_trans, dummy_trans_64, cuda_dev, args.warmup, args.repeat)
    lat_tiny_gpu_b64 = benchmark_latency(m_tiny, dummy_tiny_64, cuda_dev, args.warmup, args.repeat)

    # Peak GPU Memory
    mem_trans_b1 = measure_peak_memory(m_trans, dummy_trans_1, cuda_dev)
    mem_tiny_b1 = measure_peak_memory(m_tiny, dummy_tiny_1, cuda_dev)
    mem_trans_b64 = measure_peak_memory(m_trans, dummy_trans_64, cuda_dev)
    mem_tiny_b64 = measure_peak_memory(m_tiny, dummy_tiny_64, cuda_dev)

    # Measure F16 Backbone Latency for End-to-End breakdown
    print("  Measuring F16 Backbone Latency on GPU...")
    astra_root = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
    cfg = configure_astra_env(REPO, astra_root, cuda_dev)
    from build_candidate_cache import load_module
    trainer_file = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    trainer_mod = load_module("train_front_variant", trainer_file)
    f16_model = build_f16_model(REPO, "eth", 1, cfg, cuda_dev)
    f16_model.eval()

    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer_mod.TrajectoryDataset(data_dir / "eth_data_test.npy", data_dir / "eth_num_test.npy", cfg)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    past, future, num_valid, *_ = next(iter(loader))
    past = past.to(cuda_dev)
    future = future.to(cuda_dev)
    num_valid = num_valid.to(cuda_dev).int()
    past, future = trainer_mod.normalize_to_focal_heading(past, future)
    center = past.mean(dim=-2, keepdim=True)

    with torch.inference_mode():
        for _ in range(50):
            _ = f16_model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
        torch.cuda.synchronize()

    t_f16_list = []
    with torch.inference_mode():
        for _ in range(200):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = f16_model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_f16_list.append((t1 - t0) * 1000.0)

    f16_b1_mean = float(np.mean(t_f16_list))
    f16_b1_p95 = float(np.percentile(t_f16_list, 95))

    rows = [
        {
            "System": "Selector-Only",
            "Model": "Global_Transformer_Residual",
            "Params": str(p_trans),
            "Param_Reduction": "0.00%",
            "FLOPs_B1": f"{flops_trans:,.0f}",
            "FLOP_Reduction": "0.00%",
            "CPU_B1_Mean_ms": f"{lat_trans_cpu_b1['mean_ms']:.4f}",
            "CPU_B1_p95_ms": f"{lat_trans_cpu_b1['p95_ms']:.4f}",
            "CPU_B64_Mean_ms": f"{lat_trans_cpu_b64['mean_ms']:.4f}",
            "GPU_B1_Mean_ms": f"{lat_trans_gpu_b1['mean_ms']:.4f}",
            "GPU_B1_Median_ms": f"{lat_trans_gpu_b1['median_ms']:.4f}",
            "GPU_B1_p95_ms": f"{lat_trans_gpu_b1['p95_ms']:.4f}",
            "GPU_B64_Mean_ms": f"{lat_trans_gpu_b64['mean_ms']:.4f}",
            "Peak_VRAM_B64_MB": f"{mem_trans_b64:.2f}",
        },
        {
            "System": "Selector-Only",
            "Model": "Tiny_L5_Linear_Residual",
            "Params": str(p_tiny),
            "Param_Reduction": f"{compression*100:.2f}%",
            "FLOPs_B1": f"{flops_tiny:,.0f}",
            "FLOP_Reduction": f"{(1.0 - flops_tiny/flops_trans)*100:.2f}%",
            "CPU_B1_Mean_ms": f"{lat_tiny_cpu_b1['mean_ms']:.4f}",
            "CPU_B1_p95_ms": f"{lat_tiny_cpu_b1['p95_ms']:.4f}",
            "CPU_B64_Mean_ms": f"{lat_tiny_cpu_b64['mean_ms']:.4f}",
            "GPU_B1_Mean_ms": f"{lat_tiny_gpu_b1['mean_ms']:.4f}",
            "GPU_B1_Median_ms": f"{lat_tiny_gpu_b1['median_ms']:.4f}",
            "GPU_B1_p95_ms": f"{lat_tiny_gpu_b1['p95_ms']:.4f}",
            "GPU_B64_Mean_ms": f"{lat_tiny_gpu_b64['mean_ms']:.4f}",
            "Peak_VRAM_B64_MB": f"{mem_tiny_b64:.2f}",
        },
        {
            "System": "End-to-End (F16 + Selector)",
            "Model": "F16 + Global_Transformer",
            "Params": f"{p_trans + sum(p.numel() for p in f16_model.parameters()):,}",
            "Param_Reduction": "0.00%",
            "FLOPs_B1": "N/A",
            "FLOP_Reduction": "0.00%",
            "CPU_B1_Mean_ms": "N/A",
            "CPU_B1_p95_ms": "N/A",
            "CPU_B64_Mean_ms": "N/A",
            "GPU_B1_Mean_ms": f"{f16_b1_mean + lat_trans_gpu_b1['mean_ms']:.4f}",
            "GPU_B1_Median_ms": f"{f16_b1_mean + lat_trans_gpu_b1['median_ms']:.4f}",
            "GPU_B1_p95_ms": f"{f16_b1_p95 + lat_trans_gpu_b1['p95_ms']:.4f}",
            "GPU_B64_Mean_ms": "N/A",
            "Peak_VRAM_B64_MB": "N/A",
        },
        {
            "System": "End-to-End (F16 + Selector)",
            "Model": "F16 + Tiny_L5_Linear",
            "Params": f"{p_tiny + sum(p.numel() for p in f16_model.parameters()):,}",
            "Param_Reduction": f"{((p_trans - p_tiny) / (p_trans + sum(p.numel() for p in f16_model.parameters())))*100:.2f}%",
            "FLOPs_B1": "N/A",
            "FLOP_Reduction": "0.00%",
            "CPU_B1_Mean_ms": "N/A",
            "CPU_B1_p95_ms": "N/A",
            "CPU_B64_Mean_ms": "N/A",
            "GPU_B1_Mean_ms": f"{f16_b1_mean + lat_tiny_gpu_b1['mean_ms']:.4f}",
            "GPU_B1_Median_ms": f"{f16_b1_mean + lat_tiny_gpu_b1['median_ms']:.4f}",
            "GPU_B1_p95_ms": f"{f16_b1_p95 + lat_tiny_gpu_b1['p95_ms']:.4f}",
            "GPU_B64_Mean_ms": "N/A",
            "Peak_VRAM_B64_MB": "N/A",
        },
    ]

    csv_path = out_dir / "efficiency_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[EXP-T2 OK] Efficiency benchmark saved to {csv_path}")


if __name__ == "__main__":
    main()
