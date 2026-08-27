from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DECODER = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS"
    / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)
RUNS = {
    "f16_temporal_social_cross": WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_025_TEMPORAL_SOCIAL_CROSS_ATTENTION",
    "f28_temporal_landmark4": WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_043_TEMPORAL_LANDMARK4",
    "f29_temporal_landmark2": WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_044_TEMPORAL_LANDMARK2",
    "f30_graph_additive_attention": WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_045_GRAPH_ADDITIVE_ATTENTION",
    "f31_cascaded_group_attention": WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_046_CASCADED_GROUP_ATTENTION",
    "f32_grouped_query_attention": WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_047_GROUPED_QUERY_ATTENTION",
    "f33_multi_query_attention": WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_048_MULTI_QUERY_ATTENTION",
    "f34_identity_value_attention": WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_049_IDENTITY_VALUE_ATTENTION",
    "f35_temporal_half_pool": WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_052_TEMPORAL_HALF_POOL",
    "f36_fixed_half_projection": WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_052_TEMPORAL_HALF_POOL",
}

CHECKPOINT_VARIANTS = {
    "f36_fixed_half_projection": "f35_temporal_half_pool",
}


def load_decoder_module():
    spec = importlib.util.spec_from_file_location("locked_decoder", DECODER)
    if spec is None or spec.loader is None:
        raise ImportError(DECODER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_for(variant: str, root: Path) -> Path:
    stored_variant = CHECKPOINT_VARIANTS.get(variant, variant)
    return root / "results" / stored_variant / "seed_1" / "val_scene_1" / "final_model.pth"


def make_batch(cfg, size: int, dense: bool) -> tuple[torch.Tensor, torch.Tensor]:
    data_path = PROJECT / "datasets/eth_ucy/processed_data_diverse" / f"{cfg.SUBSET}_data_train.npy"
    num_path = PROJECT / "datasets/eth_ucy/processed_data_diverse" / f"{cfg.SUBSET}_num_train.npy"
    raw = np.load(data_path, mmap_mode="r")
    counts = np.asarray(np.load(num_path, mmap_mode="r"), dtype=np.int64)
    if dense:
        order = np.argsort(-counts)[:size]
    else:
        order = np.arange(min(size, len(raw)))
    obs_len = int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME)
    scale = float(cfg.PREDICTION.TRAJ_SCALE)
    coordinates = np.asarray(raw[order, :, :obs_len, -2:], dtype=np.float32) / scale
    return torch.from_numpy(coordinates.copy()), torch.from_numpy(counts[order].copy()).int()


def benchmark(model, past: torch.Tensor, num_valid: torch.Tensor, warmup: int, iterations: int) -> dict:
    center = past.mean(dim=-2, keepdim=True)

    def call():
        return model(
            past - center, None, None, mode="testing",
            num_valid=num_valid, graph_loc=past,
        )[2]

    with torch.inference_mode():
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        # Time long blocks and synchronize once per block.  Per-call CUDA
        # synchronization is larger than this short model's own latency.
        samples = []
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                call()
            end.record()
            torch.cuda.synchronize()
            samples.append(float(start.elapsed_time(end)) / iterations)
        peak_delta = torch.cuda.max_memory_allocated() - baseline
    return {
        "latency_ms_mean": statistics.fmean(samples),
        "latency_ms_median": statistics.median(samples),
        "latency_ms_p95": float(np.quantile(samples, 0.95)),
        "peak_forward_bytes": int(max(0, peak_delta)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the comparable benchmark")

    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(WORKSPACE / "FRONT_ENCODER_SERIES")]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA

    device = torch.device(args.device)
    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.device = str(device)
    cfg.device_list = [device.index or 0]
    decoder_module = load_decoder_module()
    rows = []
    run_items = (
        [(variant, RUNS[variant]) for variant in args.variants]
        if args.variants else list(RUNS.items())
    )
    for variant, root in run_items:
        checkpoint = checkpoint_for(variant, root)
        if not checkpoint.is_file():
            print(f"[skip] {variant}: no checkpoint", flush=True)
            continue
        model = FrontEncoderASTRA(cfg, variant)
        decoder_module.replace_decoder(model, cfg, "kinematic_k20")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model = model.to(device).eval()
        parameter_count = sum(p.numel() for p in model.parameters())
        for batch_size, dense in ((1, True), (64, False)):
            past, num_valid = make_batch(cfg, batch_size, dense)
            result = benchmark(
                model, past.to(device), num_valid.to(device),
                args.warmup, args.iterations,
            )
            rows.append({
                "variant": variant, "batch_size": batch_size,
                "parameters": parameter_count, **result,
            })
        del model
        torch.cuda.empty_cache()
        print(f"[done] {variant}", flush=True)

    output = Path(__file__).resolve().parent / "benchmark_results.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
