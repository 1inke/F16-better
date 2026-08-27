"""Numerical and model checks for EXP-138."""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent; WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]; os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from locked_mode_query_loss import ModeQueryLoss
    from train_front_variant import augment_trajectories, future_motion_difficulty_weights
    custom = load("targeted_smoke", ROOT / "targeted_models.py"); decoder = load("decoder_smoke", DECODER)
    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml"); cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); cfg.device = str(device); cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    baseline = FrontEncoderASTRA(cfg, "f16_temporal_social_cross"); model = copy.deepcopy(baseline)
    custom.replace_encoder(model, cfg, "t1_residual_shape_scale")
    past_cpu = torch.randn(2, 9, 8, 2) * 0.2
    with torch.no_grad():
        difference = (baseline.robust_state(past_cpu, float(cfg.DATA.FREQUENCY)) - model.robust_state(past_cpu, float(cfg.DATA.FREQUENCY))).abs().max().item()
    if difference > 1.0e-7: raise AssertionError(f"residual is not zero-initialized: {difference}")
    decoder.replace_decoder(model, cfg, "kinematic_k20"); model = model.to(device).train()
    past = past_cpu.to(device); num_valid = torch.tensor([3, 9], device=device)
    prediction = model(past, None, None, mode="train", num_valid=num_valid, graph_loc=past)[2]
    if prediction.shape != (2, 9, 20, 12, 2): raise AssertionError(prediction.shape)
    loss_fn = ModeQueryLoss(cfg).to(device)
    model_future = torch.randn(2, 9, 12, 2, device=device) * 0.3
    model_weights = future_motion_difficulty_weights(past, model_future, 0.8)
    model_loss, _ = loss_fn(
        prediction[:, :1], model_future[:, :1], model.last_mode_logits[:, :1], cfg,
        sample_weight=model_weights,
    )
    model_loss.backward()
    target = torch.randn(4, 1, 12, 2, device=device); predictions = torch.randn(4, 1, 20, 12, 2, device=device, requires_grad=True); logits = torch.randn(4, 1, 20, device=device, requires_grad=True)
    unweighted, _ = loss_fn(predictions, target, logits, cfg); ones, _ = loss_fn(predictions, target, logits, cfg, sample_weight=torch.ones(4, device=device))
    if not torch.allclose(unweighted, ones, atol=1.0e-6, rtol=1.0e-6): raise AssertionError((unweighted.item(), ones.item()))
    ones.backward()
    augment_args = SimpleNamespace(
        augment_rotation=False, augment_reflection=False,
        augment_speed_min=1.0, augment_speed_max=1.0,
        augment_probability=1.0, augment_curvature_max=0.025,
    )
    curve_past = torch.randn(3, 9, 8, 2, device=device)
    curve_future = torch.randn(3, 9, 12, 2, device=device)
    before_distance = torch.linalg.vector_norm(
        curve_future[:, :, 5, None] - curve_future[:, None, :, 5], dim=-1
    )
    bent_past, bent_future = augment_trajectories(
        curve_past, curve_future, augment_args
    )
    after_distance = torch.linalg.vector_norm(
        bent_future[:, :, 5, None] - bent_future[:, None, :, 5], dim=-1
    )
    rigidity_error = (before_distance - after_distance).abs().max().item()
    if rigidity_error > 2.0e-5: raise AssertionError(rigidity_error)
    print(f"[ok] residual_exact_zero={difference:.2e} shape={tuple(prediction.shape)} params={sum(p.numel() for p in model.parameters())}")
    print(f"[ok] weighted_loss_unit_equivalence delta={abs(unweighted.item()-ones.item()):.2e}")
    print(f"[ok] future_weighted_model_backward weights={model_weights.tolist()}")
    print(f"[ok] curvature_geometry_rigidity max_error={rigidity_error:.2e}")


if __name__ == "__main__": main()
