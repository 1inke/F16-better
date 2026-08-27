"""Structural and gradient validation for EXP-140A."""

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
    WORKSPACE / "HIERARCHICAL_DECODER_SERIES"
    / "EXP_139C_THREE_ANCHOR_PROPOSAL_ONLY" / "results" / "general"
    / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from locked_mode_query_loss import ModeQueryLoss

    os.environ["EXP140_SOURCE_CHECKPOINT"] = str(SOURCE)
    custom = load("exp140_smoke", ROOT / "frozen_all20_refiner_models.py")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]

    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    custom.replace_decoder(model, cfg, "r1_frozen_all20_refine")
    model = model.to(device)
    model.train(); model.decoder.apply_training_policy(model)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("decoder.refiner.") for name in trainable):
        raise AssertionError(f"Unexpected trainable parameters: {trainable}")

    past = torch.randn(2, 9, 8, 2, device=device) * 0.20
    future = torch.randn(2, 9, 12, 2, device=device) * 0.30
    num_valid = torch.tensor([3, 9], device=device)
    prediction = model(past, future, None, mode="train", num_valid=num_valid, graph_loc=past)[2]
    coarse = model.decoder.last_coarse_trajectories
    selected = model.decoder.last_selected_indices
    if prediction.shape != (2, 9, 20, 12, 2):
        raise AssertionError(prediction.shape)
    if selected is None or selected.shape != (2, 9, 20):
        raise AssertionError(f"All 20 modes were not selected: {None if selected is None else selected.shape}")
    if coarse is None:
        raise AssertionError("Missing frozen coarse trajectories")
    endpoint_error = (prediction[..., -1, :] - coarse[..., -1, :]).abs().max()
    if endpoint_error > 1.0e-6:
        raise AssertionError(f"Refiner changed endpoints: {endpoint_error}")

    loss, _ = ModeQueryLoss(cfg).to(device)(
        prediction[:, :1], future[:, :1], model.last_mode_logits[:, :1], cfg
    )
    loss.backward()
    bad = []
    finite = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                bad.append(name)
            else:
                finite += 1
        elif parameter.grad is not None:
            bad.append(f"frozen_has_gradient:{name}")
    if bad:
        raise AssertionError(bad)
    print(
        f"[ok] source={SOURCE}\n"
        f"[ok] total_params={sum(p.numel() for p in model.parameters())} "
        f"trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad)}\n"
        f"[ok] all20_shape={tuple(selected.shape)} endpoint_error={endpoint_error.item():.3g} "
        f"finite_gradient_tensors={finite}"
    )


if __name__ == "__main__":
    main()

