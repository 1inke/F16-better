"""Structural, zero-init, gradient and invalid-agent checks for RB1."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
SOURCE = (
    WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" / "EXP_146B_ANCHORED_MODE_TEMPLATE" /
    "results" / "general" / "f16_temporal_social_cross" / "seed_1" /
    "val_scene_-1" / "best_selected.pth"
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def build(with_response: bool, device: torch.device):
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device); cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    if with_response:
        load("raib_smoke_encoder", ROOT / "response_aware_models.py").replace_encoder(
            model, cfg, "rb1_response_aware_bias"
        )
    load("raib_smoke_decoder", ROOT / "current_mode_anchor_adapter.py").replace_decoder(
        model, cfg, "current_mode_anchor_template"
    )
    return model.to(device)


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
    os.chdir(PROJECT); import sitecustomize  # noqa: F401
    os.environ["RAIB_SOURCE_CHECKPOINT"] = str(SOURCE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline = build(False, device).eval(); response = build(True, device).eval()
    past = torch.randn(2, 9, 8, 2, device=device) * 0.20
    past[..., 0] += torch.linspace(0.0, 0.7, 8, device=device)
    num_valid = torch.tensor([4, 9], device=device)
    with torch.inference_mode():
        base_prediction = baseline(past, None, None, mode="testing", num_valid=num_valid, graph_loc=past)[2]
        response_prediction = response(past, None, None, mode="testing", num_valid=num_valid, graph_loc=past)[2]
    if not torch.allclose(base_prediction, response_prediction, atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError("zero-init RB1 does not reproduce B0")
    if tuple(response.decoder.anchor_steps) != (4, 8, 12) or response.decoder.cv_reference:
        raise AssertionError("decoder contract violated")
    module = response.agent_aware_trans_encoder.response_aware_bias
    if float(module.last_max_abs_bias) != 0.0: raise AssertionError("initial response bias is not zero")

    changed = past.clone(); changed[0, 4:] = torch.randn_like(changed[0, 4:]) * 50.0
    with torch.inference_mode():
        reference = response(past, None, None, mode="testing", num_valid=num_valid, graph_loc=past)[2]
        padded_changed = response(changed, None, None, mode="testing", num_valid=num_valid, graph_loc=changed)[2]
    if not torch.allclose(reference[0, :4], padded_changed[0, :4], atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError("invalid padded agents affect valid predictions")

    response.train(); response.decoder.apply_training_policy(response)
    prediction = response(past, None, None, mode="train", num_valid=num_valid, graph_loc=past)[2]
    loss = prediction.square().mean() + response.last_mode_logits.square().mean()
    loss.backward()
    gradient = module.projection.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or float(gradient.norm()) <= 0.0:
        raise AssertionError("response bias has no finite nonzero gradient")
    if any(parameter.requires_grad for parameter in response.decoder.parameters()):
        raise AssertionError("decoder is not frozen")
    print(
        f"[ok] shape={tuple(prediction.shape)} anchors={response.decoder.anchor_steps} "
        f"cv={response.decoder.cv_reference} added_params={sum(p.numel() for p in module.parameters())} "
        f"grad_norm={float(gradient.norm()):.6e}"
    )


if __name__ == "__main__": main()
