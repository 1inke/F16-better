"""Validate annotation parsing and one ASTRA forward/backward without training."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATASET = Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
DECODER = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS" /
    "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT), str(ROOT)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from preprocess_sdd_lowcost import build_video_samples, read_pedestrian_tracks

    annotation = DATASET / "annotations/bookstore/video0/annotations.txt"
    tracks = read_pedestrian_tracks(annotation)
    samples, counts, stats = build_video_samples(
        tracks=tracks, scene_id=0, frame_step=12, obs_steps=8, pred_steps=12,
        window_stride_steps=4, max_agents=16, pixel_scale=20.0, max_samples=4,
    )
    if len(samples) == 0:
        raise RuntimeError("SDD parser produced no samples")

    cfg = load_locked_q20(PROJECT / "configs/motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 16
    cfg.MODEL.GRAPH.TOP_K = 6
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = "cpu"
    cfg.device_list = ["cpu"]
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    decoder = load_module("sdd_kinematic_decoder", DECODER)
    decoder.replace_decoder(model, cfg, "kinematic_k20")
    model.train()

    coordinates = torch.from_numpy(samples[:2, :, :, -2:])
    past, future = coordinates[:, :, :8], coordinates[:, :, 8:]
    num_valid = torch.from_numpy(counts[:2]).int()
    center = past.mean(dim=-2, keepdim=True)
    prediction = model(
        past - center, future - center, None, mode="train",
        num_valid=num_valid, graph_loc=past,
    )[2]
    logits = model.last_mode_logits
    loss = prediction[:, :1].square().mean() + 0.01 * logits[:, :1].square().mean()
    loss.backward()
    finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if not finite:
        raise RuntimeError("non-finite gradient")
    result = {
        "parser": "passed",
        "forward_backward": "passed",
        "sample_shape": list(samples.shape),
        "valid_agents": counts.tolist(),
        "prediction_shape": list(prediction.shape),
        "parameters_n16": sum(p.numel() for p in model.parameters()),
        "video_stats": stats,
    }
    (ROOT / "pipeline_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
