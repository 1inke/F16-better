from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
ASTRA_ROOT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
EXP173_ROOT = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
EXP173_RESULTS = EXP173_ROOT / "results"
CFG_PATH = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
TRAINER = ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
MODEL_FILE = ROOT / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
ADAPTER = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
DEFAULT_OUT = Path(__file__).resolve().parent / "EXP_174A_FINAL_EFFICIENCY"

FOLD = "eth"
VARIANT = "f16_temporal_social_cross"
MODELS = {
    "F16": ("e3", "f16_temporal_social_cross"),
    "E2": ("e2", "f4_relation_tokens"),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(device: torch.device, max_agents: int = 64):
    os.chdir(ASTRA_ROOT)
    sys.path[:0] = [str(ASTRA_ROOT), str(ASTRA_ROOT / "scripts"), str(ROOT / "FRONT_ENCODER_SERIES")]
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    cfg = load_locked_q20(CFG_PATH)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = max_agents
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    return cfg


def checkpoint(label: str, seed: int, stage: str) -> Path:
    disk_label, variant = MODELS[label]
    return EXP173_RESULTS / FOLD / stage / disk_label / variant / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"


def build_model(label: str, seed: int, device: torch.device, max_agents: int):
    sys.path.insert(0, str(ASTRA_ROOT))
    sys.path.insert(0, str(ROOT / "FRONT_ENCODER_SERIES"))
    cfg = configure(device, max_agents)
    trainer = load_module("exp174_trainer", TRAINER)
    models = load_module("exp174_models", MODEL_FILE)
    adapter = load_module("exp174_joint_v2", ADAPTER)
    h3 = checkpoint(label, seed, "h3")
    final = checkpoint(label, seed, "v2")
    if not h3.exists() or not final.exists():
        raise FileNotFoundError(f"Missing EXP173 checkpoint: {h3 if not h3.exists() else final}")
    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3)
    os.environ["EXP173_DECODER_INIT_SEED"] = str(seed)
    model = models.FrontEncoderASTRA(cfg, MODELS[label][1])
    adapter.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
    state = torch.load(final, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg, trainer


def real_batch(trainer, cfg, batch_size: int, device: torch.device):
    data = ASTRA_ROOT / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer.TrajectoryDataset(data / f"{FOLD}_data_test.npy", data / f"{FOLD}_num_test.npy", cfg)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, num_workers=0)
    past, _, valid, *_ = next(iter(loader))
    if past.shape[0] < batch_size:
        reps = int(np.ceil(batch_size / past.shape[0]))
        past, valid = past.repeat((reps, 1, 1, 1)), valid.repeat(reps)
    return past[:batch_size].to(device), valid[:batch_size].to(device).int()


def synthetic_batch(batch_size: int, agents: int, device: torch.device):
    idx = torch.arange(agents, dtype=torch.float32, device=device)
    base = torch.stack(((idx % 8) * 0.45, torch.floor(idx / 8) * 0.45), dim=-1)
    vel = torch.stack((0.07 + 0.012 * torch.sin(idx), 0.025 * torch.cos(idx * 0.7)), dim=-1)
    timeline = torch.arange(8, dtype=torch.float32, device=device)
    past = base[:, None, :] + timeline[None, :, None] * vel[:, None, :]
    return past.unsqueeze(0).repeat(batch_size, 1, 1, 1), torch.full((batch_size,), agents, dtype=torch.long, device=device)


def forward(model, past, valid):
    center = past.mean(dim=-2, keepdim=True)
    result = model(past - center, None, None, mode="testing", num_valid=valid, graph_loc=past)
    return result[2] if isinstance(result, (tuple, list)) else result


def edge_count(past: torch.Tensor, valid: torch.Tensor, radius: float = 6.0, topk: int = 4) -> int:
    total = 0
    last = past[:, :, -1]
    for b, n_tensor in enumerate(valid):
        n = int(n_tensor.item())
        if n <= 1:
            continue
        dist = torch.cdist(last[b, :n], last[b, :n])
        dist.fill_diagonal_(float("inf"))
        total += int(torch.minimum((dist <= radius).sum(dim=-1), torch.full((n,), topk, device=dist.device)).sum().item())
    return total


def profile_flops(model, past, valid, device: torch.device) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode(), torch.profiler.profile(activities=activities, with_flops=True) as prof:
        forward(model, past, valid)
    return int(sum(int(event.flops or 0) for event in prof.key_averages()))


def benchmark(model, past, valid, device: torch.device, warmup: int, iterations: int):
    with torch.inference_mode():
        for _ in range(warmup):
            forward(model, past, valid)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            forward(model, past, valid)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "latency_mean_ms": float(statistics.fmean(timings)),
        "latency_median_ms": float(statistics.median(timings)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
        "throughput_samples_s": float(past.shape[0] * 1000.0 / statistics.fmean(timings)),
        "peak_vram_mib": float(peak / 1024**2),
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-174A final F16/E2 efficiency audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")

    rows = []
    for label in MODELS:
        configs = [("real_batch", b, 9) for b in (1, 16, 64)] + [("agent_scaling", 1, n) for n in (4, 8, 16, 32, 64)]
        if args.smoke:
            configs = [("real_batch", 1, 9), ("agent_scaling", 1, 64)]
        model = cfg = trainer = None
        built_agents = None
        for family, batch_size, agents in configs:
            if built_agents != agents:
                if model is not None:
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                print(f"[load] EXP173 final {label} max_agents={agents}", flush=True)
                model, cfg, trainer = build_model(label, args.seed, device, agents)
                built_agents = agents
            params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            past, valid = (real_batch(trainer, cfg, batch_size, device) if family == "real_batch" else synthetic_batch(batch_size, agents, device))
            with torch.inference_mode():
                prediction = forward(model, past, valid)
            if prediction.shape[0] != batch_size or not torch.isfinite(prediction).all():
                raise RuntimeError(f"Invalid output for {label}/{family}/B{batch_size}/N{agents}: {tuple(prediction.shape)}")
            row = {
                "model": label,
                "variant": MODELS[label][1],
                "family": family,
                "batch_size": batch_size,
                "agents": agents,
                "parameters": params,
                "trainable_parameters": trainable,
                "directed_sparse_edges": edge_count(past, valid),
                "dense_attention_score_elements": int(batch_size * 4 * agents * agents * 8),
                "sparse_attention_score_elements": int(batch_size * 4 * agents * min(4, max(0, agents - 1)) * 8),
                "output_shape": list(prediction.shape),
            }
            if not args.smoke:
                row["profiler_flops"] = profile_flops(model, past, valid, device)
                row.update(benchmark(model, past, valid, device, args.warmup, args.iterations))
            rows.append(row)
            print(f"[done] {label} {family} B={batch_size} N={agents}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.smoke:
        print(json.dumps({"status": "smoke_ok", "cases": rows}, ensure_ascii=False, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": "EXP-174A",
        "checkpoint_source": str(EXP173_RESULTS),
        "fold": FOLD,
        "seed": args.seed,
        "precision": "FP32",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "flops_note": "Profiler-accounted FLOPs; operators without profiler formulas are not counted.",
        "rows": rows,
    }
    (args.output_dir / "efficiency_results.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "efficiency_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        keys = sorted({key for row in rows for key in row if key != "output_shape"})
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "output_shape"} for row in rows)
    print(f"[complete] {args.output_dir / 'efficiency_results.json'}")


if __name__ == "__main__":
    main()
