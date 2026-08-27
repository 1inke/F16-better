"""Forward/backward and structural checks for all EXP-139 decoders."""

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


def load(name: str, path: Path):
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
    from locked_mode_query_loss import ModeQueryLoss
    custom = load("hierarchical_decoder_smoke", ROOT / "hierarchical_decoder_models.py")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for experiment in sorted(custom.EXPERIMENTS):
        cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
        cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
        cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0
        cfg.device = str(device); cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
        model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
        custom.replace_decoder(model, cfg, experiment)
        model = model.to(device).train()
        past = torch.randn(2, 9, 8, 2, device=device) * 0.20
        future = torch.randn(2, 9, 12, 2, device=device) * 0.30
        num_valid = torch.tensor([3, 9], device=device)
        model.decoder.set_training_target(future)
        prediction = model(
            past, future, None, mode="train", num_valid=num_valid, graph_loc=past
        )[2]
        if prediction.shape != (2, 9, 20, 12, 2):
            raise AssertionError((experiment, prediction.shape))
        if model.last_mode_logits.shape != (2, 9, 20):
            raise AssertionError(model.last_mode_logits.shape)
        coarse = model.decoder.last_coarse_trajectories
        if coarse is None or coarse.shape != prediction.shape:
            raise AssertionError("missing coarse trajectories")
        if model.decoder.use_refiner:
            selected = model.decoder.last_selected_indices
            if selected is None or selected.shape != (2, 9, 8):
                raise AssertionError("invalid Top-M selection")
            endpoint_error = (prediction[..., -1, :] - coarse[..., -1, :]).abs().max()
            if endpoint_error > 1.0e-6:
                raise AssertionError(f"refiner changed endpoint: {endpoint_error}")
        else:
            if model.decoder.last_selected_indices is not None:
                raise AssertionError("proposal-only decoder selected modes")
        loss_fn = ModeQueryLoss(cfg).to(device)
        loss, _ = loss_fn(
            prediction[:, :1], future[:, :1], model.last_mode_logits[:, :1], cfg
        )
        loss.backward()
        finite_gradients = sum(
            int(parameter.grad is not None and torch.isfinite(parameter.grad).all())
            for parameter in model.parameters() if parameter.requires_grad
        )
        print(
            f"[ok] {experiment} params={sum(p.numel() for p in model.parameters())} "
            f"shape={tuple(prediction.shape)} gradients={finite_gradients}"
        )


if __name__ == "__main__":
    main()
