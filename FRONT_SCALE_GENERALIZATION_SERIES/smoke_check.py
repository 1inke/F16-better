"""One-batch forward/backward check for EXP-137 model plugins."""

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
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    custom = load_module("scale_generalization_smoke", ROOT / "scale_generalization_models.py")
    decoder = load_module("kinematic_decoder_smoke", DECODER)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for experiment in sorted(custom.EXPERIMENTS):
        cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
        cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
        cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0
        cfg.device = str(device); cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
        model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
        custom.replace_encoder(model, cfg, experiment); decoder.replace_decoder(model, cfg, "kinematic_k20")
        model = model.to(device).train()
        past = torch.randn(2, 9, 8, 2, device=device) * 0.2
        num_valid = torch.tensor([3, 9], device=device)
        prediction = model(past, None, None, mode="train", num_valid=num_valid, graph_loc=past)[2]
        if prediction.shape != (2, 9, 20, 12, 2):
            raise AssertionError((experiment, prediction.shape))
        prediction.square().mean().backward()
        finite = sum(int(p.grad is not None and torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
        print(f"[ok] {experiment} shape={tuple(prediction.shape)} params={sum(p.numel() for p in model.parameters())} gradient_tensors={finite}")


if __name__ == "__main__":
    main()
