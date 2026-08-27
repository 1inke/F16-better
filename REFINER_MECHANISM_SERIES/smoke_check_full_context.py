"""Forward/backward validation for EXP-140B."""

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
SOURCE = WORKSPACE / "HIERARCHICAL_DECODER_SERIES" / "EXP_139C_THREE_ANCHOR_PROPOSAL_ONLY" / "results" / "general" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"


def load(path: Path):
    spec = importlib.util.spec_from_file_location("exp140b_smoke", path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from locked_mode_query_loss import ModeQueryLoss
    os.environ["EXP140_SOURCE_CHECKPOINT"] = str(SOURCE)
    custom = load(ROOT / "full_context_all20_refiner_models.py")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device); cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    custom.replace_decoder(model, cfg, "r2_frozen_full_context_all20")
    model = model.to(device); model.train(); model.decoder.apply_training_policy(model)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not name.startswith("decoder.refiner.") for name in trainable):
        raise AssertionError(trainable)
    past = torch.randn(2, 9, 8, 2, device=device) * 0.2
    future = torch.randn(2, 9, 12, 2, device=device) * 0.3
    valid = torch.tensor([3, 9], device=device)
    prediction = model(past, future, None, mode="train", num_valid=valid, graph_loc=past)[2]
    coarse = model.decoder.last_coarse_trajectories
    selected = model.decoder.last_selected_indices
    if prediction.shape != (2, 9, 20, 12, 2) or selected.shape != (2, 9, 20):
        raise AssertionError((prediction.shape, selected.shape))
    endpoint_error = (prediction[..., -1, :] - coarse[..., -1, :]).abs().max()
    if endpoint_error > 1e-6: raise AssertionError(endpoint_error)
    loss, _ = ModeQueryLoss(cfg).to(device)(prediction[:, :1], future[:, :1], model.last_mode_logits[:, :1], cfg)
    loss.backward()
    bad = [name for name, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    if bad: raise AssertionError(bad)
    print(f"[ok] shape={tuple(prediction.shape)} endpoint_error={endpoint_error.item():.3g} total={sum(p.numel() for p in model.parameters())} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)} gradients={len(trainable)}")


if __name__ == "__main__": main()

