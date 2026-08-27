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
W = HERE.parent
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DECODER = W / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
CHECKPOINT = W / "AUTONOMOUS_EXPERIMENTS" / "EXP_052_TEMPORAL_HALF_POOL" / "results" / "f35_temporal_half_pool" / "seed_1" / "val_scene_1" / "final_model.pth"
F16_CHECKPOINT = W / "AUTONOMOUS_EXPERIMENTS" / "EXP_025_TEMPORAL_SOCIAL_CROSS_ATTENTION" / "results" / "f16_temporal_social_cross" / "seed_1" / "val_scene_1" / "final_model.pth"
F29_CHECKPOINT = W / "AUTONOMOUS_EXPERIMENTS" / "EXP_044_TEMPORAL_LANDMARK2" / "results" / "f29_temporal_landmark2" / "seed_1" / "val_scene_1" / "final_model.pth"

sys.path[:0] = [str(P), str(P / "scripts"), str(W / "FRONT_ENCODER_SERIES"), str(HERE)]
os.chdir(P)
import sitecustomize  # noqa: F401
from experiment_config import load_locked_q20
from front_encoder_models import FrontEncoderASTRA
from benchmark_attention_variants import make_batch

spec = importlib.util.spec_from_file_location("locked_decoder", DECODER)
decoder = importlib.util.module_from_spec(spec); spec.loader.exec_module(decoder)
cfg = load_locked_q20(P / "configs" / "motif_rbf_univ.yaml")
cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
cfg.device = "cuda:0"; cfg.device_list = [0]
variants = ["f16_temporal_social_cross", "f29_temporal_landmark2", "f35_temporal_half_pool", "f36_fixed_half_projection"]
models = {}
for variant in variants:
    model = FrontEncoderASTRA(cfg, variant)
    decoder.replace_decoder(model, cfg, "kinematic_k20")
    checkpoint = F16_CHECKPOINT if variant.startswith("f16_") else F29_CHECKPOINT if variant.startswith("f29_") else CHECKPOINT
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    models[variant] = model.cuda().eval()

results = []
for batch_size in (1, 64):
    past, num_valid = make_batch(cfg, batch_size, dense=batch_size == 1)
    past = past.cuda(); num_valid = num_valid.cuda(); center = past.mean(dim=-2, keepdim=True)
    def call(model):
        return model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)[2]
    with torch.inference_mode():
        for model in models.values():
            for _ in range(20): call(model)
        torch.cuda.synchronize()
        samples = {variant: [] for variant in variants}
        for cycle in range(18):
            offset = cycle % len(variants)
            order = variants[offset:] + variants[:offset]
            for variant in order:
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50): call(models[variant])
                end.record(); torch.cuda.synchronize()
                samples[variant].append(float(start.elapsed_time(end)) / 50.0)
    for variant in variants:
        values = samples[variant]
        results.append({"variant": variant, "batch_size": batch_size,
            "latency_ms_mean": statistics.fmean(values),
            "latency_ms_median": statistics.median(values),
            "latency_ms_p95": float(np.quantile(values, 0.95)),
            "cycles": len(values)})

output = HERE / "interleaved_benchmark_results.json"
output.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
