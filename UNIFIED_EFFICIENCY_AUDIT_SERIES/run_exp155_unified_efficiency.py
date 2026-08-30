"""EXP-155: comparable efficiency audit for the frozen current decoder.

Measures F0, F16 and F35 using the exact checkpoints and decoder protocol of
EXP-154.  No training or target-test data are used.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FORMAL = WORKSPACE / "FORMAL_ENCODER_ABLATION_SERIES"
ADAPTER = FORMAL / "formal_decoder_adapter.py"
RESULTS = FORMAL / "EXP_154_FORMAL_ENCODER_ABLATION" / "results" / "general"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
OUT = HERE / "EXP_155_UNIFIED_EFFICIENCY" / "results"

VARIANTS = {
    "f0_baseline": "a_trajectory_only",
    "f3_factorized": "b_sparse_graph",
    "f4_relation_tokens": "c_relation_prior",
    "f16_temporal_social_cross": "d_f16_full_temporal",
    "f35_temporal_half_pool": "e_f35_half_pool",
}


def make_batch(cfg, size: int, dense: bool) -> tuple[torch.Tensor, torch.Tensor]:
    raw = np.load(DATA / "eth_data_train.npy", mmap_mode="r")
    counts = np.asarray(np.load(DATA / "eth_num_train.npy", mmap_mode="r"), dtype=np.int64)
    order = np.argsort(-counts)[:size] if dense else np.arange(min(size, len(raw)))
    obs_len = int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME)
    scale = float(cfg.PREDICTION.TRAJ_SCALE)
    coordinates = np.asarray(raw[order, :, :obs_len, -2:], dtype=np.float32) / scale
    return torch.from_numpy(coordinates.copy()), torch.from_numpy(counts[order].copy()).int()


def call(model, past: torch.Tensor, num_valid: torch.Tensor) -> torch.Tensor:
    center = past.mean(dim=-2, keepdim=True)
    return model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)[2]


def params(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    decoder = sum(p.numel() for p in model.decoder.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable,
            "decoder_parameters": decoder, "encoder_and_other_parameters": total - decoder}


def forward_memory(model, past, num_valid) -> int:
    with torch.inference_mode():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        call(model, past, num_valid); torch.cuda.synchronize()
        return int(max(0, torch.cuda.max_memory_allocated() - baseline))


def training_memory(model, past, num_valid) -> int:
    model.train(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated(); model.zero_grad(set_to_none=True)
    prediction = call(model, past, num_valid)
    prediction.float().square().mean().backward(); torch.cuda.synchronize()
    peak = int(max(0, torch.cuda.max_memory_allocated() - baseline))
    model.zero_grad(set_to_none=True); model.eval()
    return peak


def profiled_flops(model, past, num_valid) -> int:
    # Operator-accounted FLOPs. Unsupported custom operations contribute zero;
    # this is reported explicitly rather than presented as a symbolic exact value.
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True, record_shapes=False,
    ) as profile:
        call(model, past, num_valid); torch.cuda.synchronize()
    return int(sum(event.flops for event in profile.key_averages() if event.flops))


def score_elements(batch: int, agents: int, heads: int, steps: int, topk: int, enabled: bool) -> dict[str, int]:
    if not enabled:
        return {"temporal_social_score_elements_dense": 0,
                "temporal_social_score_elements_nominal_topk": 0}
    return {
        "temporal_social_score_elements_dense": batch * heads * agents * agents * steps,
        # Nominal slots in the directed sparse graph. Actual valid edges can be
        # fewer in sparse scenes; that dynamic count is intentionally not guessed.
        "temporal_social_score_elements_nominal_topk": batch * heads * agents * min(topk, max(0, agents - 1)) * steps,
    }


def sparse_graph_elements(batch: int, agents: int, heads: int, topk: int, enabled: bool) -> dict[str, int]:
    if not enabled:
        return {"sparse_graph_directed_edge_slots": 0,
                "sparse_graph_attention_score_elements_nominal": 0}
    edge_slots = batch * agents * min(topk, max(0, agents - 1))
    return {"sparse_graph_directed_edge_slots": edge_slots,
            "sparse_graph_attention_score_elements_nominal": heads * edge_slots}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--skip-training-memory", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="load every model and run one forward only")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a comparable GPU audit")
    for path in (PROJECT, ADAPTER, DATA / "eth_data_train.npy", DATA / "eth_num_train.npy"):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(WORKSPACE / "FRONT_ENCODER_SERIES"), str(FORMAL)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    import formal_decoder_adapter as decoder_adapter

    device = torch.device(args.device)
    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.device = str(device); cfg.device_list = [device.index or 0]
    models: dict[str, torch.nn.Module] = {}
    metadata: dict[str, dict] = {}
    for variant, label in VARIANTS.items():
        checkpoint = RESULTS / label / variant / "seed_1" / "val_scene_-1" / "best_selected.pth"
        source = RESULTS / "d_f16_full_temporal" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"
        if not checkpoint.is_file() or not source.is_file():
            raise FileNotFoundError(f"checkpoint missing: {checkpoint} or {source}")
        os.environ["FORMAL_DECODER_SOURCE"] = str(source)
        model = FrontEncoderASTRA(cfg, variant)
        decoder_adapter.replace_decoder(model, cfg, "fixed_current_mode_anchor")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        models[variant] = model.to(device).eval(); metadata[variant] = params(model)
        print(f"[loaded] {variant}", flush=True)

    smoke_past, smoke_num = make_batch(cfg, 1, True)
    smoke_past, smoke_num = smoke_past.to(device), smoke_num.to(device)
    with torch.inference_mode():
        for name, model in models.items():
            output = call(model, smoke_past, smoke_num)
            print(f"[smoke-ok] {name}: {tuple(output.shape)}", flush=True)
    if args.smoke:
        return

    rows = []
    for batch_size, dense in ((1, True), (64, False)):
        past, num_valid = make_batch(cfg, batch_size, dense)
        past, num_valid = past.to(device), num_valid.to(device)
        with torch.inference_mode():
            for model in models.values():
                for _ in range(args.warmup): call(model, past, num_valid)
            torch.cuda.synchronize()
            samples = {name: [] for name in models}
            names = list(models)
            for cycle in range(args.cycles):
                order = names[cycle % len(names):] + names[:cycle % len(names)]
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations): call(models[name], past, num_valid)
                    end.record(); torch.cuda.synchronize()
                    samples[name].append(float(start.elapsed_time(end)) / args.iterations)
        for name, model in models.items():
            values = samples[name]
            row = {"variant": name, "batch_size": batch_size, **metadata[name],
                   "latency_ms_mean": statistics.fmean(values), "latency_ms_std": statistics.stdev(values),
                   "latency_ms_median": statistics.median(values), "latency_ms_p95": float(np.quantile(values, .95)),
                   "throughput_samples_per_s": 1000.0 * batch_size / statistics.fmean(values),
                   "forward_peak_bytes": forward_memory(model, past, num_valid),
                   "benchmark_cycles": args.cycles, "iterations_per_cycle": args.iterations,
                   **score_elements(batch_size, int(past.shape[1]), 4, 2 if name == "f35_temporal_half_pool" else 8, 4,
                                    enabled=name in {"f16_temporal_social_cross", "f35_temporal_half_pool"}),
                   **sparse_graph_elements(batch_size, int(past.shape[1]), 4, 4,
                                           enabled=name != "f0_baseline")}
            if not args.skip_training_memory:
                row["training_peak_bytes_proxy"] = training_memory(model, past, num_valid)
            rows.append(row)
            print(f"[done] batch={batch_size} {name}", flush=True)

    # Full-model profiler FLOPs are collected on one representative dense sample.
    for row in rows:
        if row["batch_size"] != 1: continue
        row["profiler_accounted_flops_batch1"] = profiled_flops(models[row["variant"]], smoke_past, smoke_num)
        row["profiler_accounted_macs_batch1"] = row["profiler_accounted_flops_batch1"] / 2.0
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": "EXP-155 unified efficiency audit", "device": str(device),
               "gpu": torch.cuda.get_device_name(device), "protocol": {"decoder": "fixed_current_mode_anchor", "source": "EXP-154 seed_1 general", "interleaved": True}, "rows": rows}
    (OUT / "efficiency_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "efficiency_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
