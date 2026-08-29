"""End-to-end zero-init, gradient, EMA-copy and geometry checks for IC1."""

from __future__ import annotations

import copy
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
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(ic1: bool, device: torch.device):
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA

    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    module = load(
        "icmab_smoke_ic1" if ic1 else "icmab_smoke_baseline",
        ROOT / ("icmab_models.py" if ic1 else "current_baseline_adapter.py"),
    )
    module.replace_decoder(
        model, cfg,
        "ic1_social_mode_anchor_bridge" if ic1 else "current_mode_anchor_template",
    )
    return model.to(device), cfg


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401

    os.environ["ICMAB_SOURCE_CHECKPOINT"] = str(SOURCE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline, _ = build(False, device)
    ic1, _ = build(True, device)
    baseline.eval()
    ic1.eval()
    past = torch.randn(3, 9, 8, 2, device=device) * 0.20
    past[..., 0] += torch.linspace(0.0, 0.7, 8, device=device)
    num_valid = torch.tensor([4, 7, 9], device=device)

    with torch.inference_mode():
        b0_prediction = baseline(
            past, None, None, mode="testing", num_valid=num_valid, graph_loc=past
        )[2]
        b0_anchors = baseline.decoder.last_corrected_anchors.clone()
        ic1_prediction = ic1(
            past, None, None, mode="testing", num_valid=num_valid, graph_loc=past
        )[2]
        ic1_anchors = ic1.decoder.last_corrected_anchors.clone()
    if not torch.allclose(b0_prediction, ic1_prediction, atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError("zero-init IC1 does not reproduce B0 predictions")
    if not torch.allclose(b0_anchors, ic1_anchors, atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError("zero-init IC1 does not reproduce B0 anchors")
    if tuple(ic1.decoder.anchor_steps) != (4, 8, 12) or ic1.decoder.cv_reference:
        raise AssertionError("decoder contract violated")

    # EMA uses deepcopy; the social-token transport must still connect the copied pair.
    ema_copy = copy.deepcopy(ic1).eval()
    with torch.inference_mode():
        copied = ema_copy(
            past, None, None, mode="testing", num_valid=num_valid, graph_loc=past
        )[2]
    if not torch.allclose(ic1_prediction, copied, atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError("EMA deepcopy breaks IC1 social-token transport")

    ic1.train()
    ic1.decoder.apply_training_policy(ic1)
    prediction = ic1(
        past, None, None, mode="train", num_valid=num_valid, graph_loc=past
    )[2]
    loss = prediction.square().mean() + ic1.last_mode_logits.square().mean()
    loss.backward()
    up_gradient = ic1.decoder.interaction_bridge.up.weight.grad
    if up_gradient is None or not torch.isfinite(up_gradient).all() or float(up_gradient.norm()) <= 0.0:
        raise AssertionError("bridge output projection has no finite nonzero gradient")
    trainable_decoder = [
        name for name, parameter in ic1.decoder.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_decoder or any(
        not name.startswith(("interaction_bridge.", "mode_gate."))
        for name in trainable_decoder
    ):
        raise AssertionError(trainable_decoder)

    # Force a nonzero bridge to prove that mode and anchor geometry can change.
    ic1.zero_grad(set_to_none=True)
    with torch.no_grad():
        torch.nn.init.normal_(ic1.decoder.interaction_bridge.up.weight, std=0.02)
        torch.nn.init.normal_(ic1.decoder.mode_gate.weight, std=0.02)
    ic1.eval()
    with torch.inference_mode():
        changed = ic1(
            past, None, None, mode="testing", num_valid=num_valid, graph_loc=past
        )[2]
        changed_anchors = ic1.decoder.last_corrected_anchors
        mode_delta = ic1.decoder.last_conditioned_mode_delta[:, 0].norm(dim=-1)
    anchor_shift = torch.linalg.vector_norm(
        changed_anchors[:, 0] - b0_anchors[:, 0], dim=-1
    )
    if float(anchor_shift.mean()) <= 0.0 or float(mode_delta.std(dim=-1).mean()) <= 0.0:
        raise AssertionError("nonzero bridge does not produce mode-differential anchor shift")
    if torch.allclose(changed, b0_prediction):
        raise AssertionError("nonzero bridge does not alter trajectories")

    bridge_params = sum(
        parameter.numel() for name, parameter in ic1.decoder.named_parameters()
        if name.startswith(("interaction_bridge.", "mode_gate."))
    )
    print(
        f"[ok] shape={tuple(prediction.shape)} anchors={ic1.decoder.anchor_steps} "
        f"cv={ic1.decoder.cv_reference} bridge_params={bridge_params} "
        f"grad_norm={float(up_gradient.norm()):.6e} "
        f"forced_anchor_shift={float(anchor_shift.mean()):.6e}"
    )


if __name__ == "__main__":
    main()
