"""Forward/backward and interpolation checks for 2/4/6-anchor decoders."""

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


def load(path: Path):
    spec = importlib.util.spec_from_file_location("exp141_smoke", path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from locked_mode_query_loss import ModeQueryLoss
    custom = load(ROOT / "anchor_count_models.py")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for name, expected_steps in custom.EXPERIMENTS.items():
        cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
        cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
        cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0
        cfg.device = str(device); cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
        model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
        custom.replace_decoder(model, cfg, name)
        model = model.to(device).train()
        if model.decoder.anchor_steps != expected_steps or model.decoder.use_refiner:
            raise AssertionError((name, model.decoder.anchor_steps, model.decoder.use_refiner))
        past = torch.randn(2, 9, 8, 2, device=device) * 0.2
        future = torch.randn(2, 9, 12, 2, device=device) * 0.3
        valid = torch.tensor([3, 9], device=device)
        prediction = model(past, future, None, mode="train", num_valid=valid, graph_loc=past)[2]
        if prediction.shape != (2, 9, 20, 12, 2): raise AssertionError(prediction.shape)
        anchors = model.decoder.last_anchor_proposals
        if anchors is None or anchors.shape[-2] != len(expected_steps): raise AssertionError(anchors.shape)
        for anchor_index, step in enumerate(expected_steps):
            error = (prediction[..., step - 1, :] - anchors[..., anchor_index, :]).abs().max()
            if error > 1e-6: raise AssertionError((name, step, error))
        loss, _ = ModeQueryLoss(cfg).to(device)(prediction[:, :1], future[:, :1], model.last_mode_logits[:, :1], cfg)
        loss.backward()
        finite = sum(int(p.grad is not None and torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
        print(f"[ok] {name} steps={expected_steps} params={sum(p.numel() for p in model.parameters())} gradients={finite}")


if __name__ == "__main__": main()

