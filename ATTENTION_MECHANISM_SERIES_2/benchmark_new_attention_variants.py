from __future__ import annotations

import importlib.util
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
DECODER = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS"
    / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)
CHECKPOINTS = {
    "f16_temporal_social_cross": WORKSPACE / "AUTONOMOUS_EXPERIMENTS/EXP_025_TEMPORAL_SOCIAL_CROSS_ATTENTION/results/f16_temporal_social_cross/seed_1/val_scene_1/final_model.pth",
    "f35_temporal_half_pool": WORKSPACE / "AUTONOMOUS_EXPERIMENTS/EXP_052_TEMPORAL_HALF_POOL/results/f35_temporal_half_pool/seed_1/val_scene_1/final_model.pth",
    "f38_recent_exact_early_pool": WORKSPACE / "AUTONOMOUS_EXPERIMENTS/EXP_057_RECENT_EXACT_EARLY_POOL/results/f38_recent_exact_early_pool/seed_1/val_scene_1/final_model.pth",
    "f39_motion_event_sampling": WORKSPACE / "AUTONOMOUS_EXPERIMENTS/EXP_058_MOTION_EVENT_SAMPLING/results/f39_motion_event_sampling/seed_1/val_scene_1/final_model.pth",
    "f40_motion_token_squeeze": WORKSPACE / "AUTONOMOUS_EXPERIMENTS/EXP_059_MOTION_TOKEN_SQUEEZE/results/f40_motion_token_squeeze/seed_1/val_scene_1/final_model.pth",
    "f41_edge_list_attention": WORKSPACE / "AUTONOMOUS_EXPERIMENTS/EXP_060_EDGE_LIST_ATTENTION/results/f41_edge_list_attention/seed_1/val_scene_1/final_model.pth",
}


def main() -> None:
    sys.path[:0] = [
        str(PROJECT), str(PROJECT / "scripts"),
        str(WORKSPACE / "FRONT_ENCODER_SERIES"),
        str(WORKSPACE / "ATTENTION_COMPUTE_SERIES"),
    ]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from benchmark_attention_variants import make_batch
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA

    spec = importlib.util.spec_from_file_location("locked_decoder", DECODER)
    if spec is None or spec.loader is None:
        raise ImportError(DECODER)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    cfg = load_locked_q20(PROJECT / "configs/motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.device = "cuda:0"
    cfg.device_list = [0]
    models = {}
    for variant, checkpoint in CHECKPOINTS.items():
        model = FrontEncoderASTRA(cfg, variant)
        decoder.replace_decoder(model, cfg, "kinematic_k20")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        models[variant] = model.cuda().eval()

    variants = list(models)
    rows = []
    for batch_size in (1, 64):
        past, num_valid = make_batch(cfg, batch_size, dense=batch_size == 1)
        past = past.cuda()
        num_valid = num_valid.cuda()
        center = past.mean(dim=-2, keepdim=True)

        def call(model):
            return model(
                past - center, None, None, mode="testing",
                num_valid=num_valid, graph_loc=past,
            )[2]

        with torch.inference_mode():
            for model in models.values():
                for _ in range(10):
                    call(model)
            torch.cuda.synchronize()
            samples = {name: [] for name in variants}
            for cycle in range(12):
                offset = cycle % len(variants)
                order = variants[offset:] + variants[:offset]
                for variant in order:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(30):
                        call(models[variant])
                    end.record()
                    torch.cuda.synchronize()
                    samples[variant].append(float(start.elapsed_time(end)) / 30.0)
            for variant in variants:
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                call(models[variant])
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - baseline
                values = samples[variant]
                rows.append({
                    "variant": variant,
                    "batch_size": batch_size,
                    "parameters": sum(p.numel() for p in models[variant].parameters()),
                    "latency_ms_mean": statistics.fmean(values),
                    "latency_ms_median": statistics.median(values),
                    "latency_ms_p95": float(np.quantile(values, 0.95)),
                    "peak_forward_bytes": int(max(0, peak)),
                })

    output = HERE / "benchmark_results.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

