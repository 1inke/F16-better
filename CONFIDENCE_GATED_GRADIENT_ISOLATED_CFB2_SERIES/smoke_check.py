from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
W = ROOT.parent
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
F = W / "FRONT_ENCODER_SERIES"
SOURCE = W / "INTERACTION_RESIDUAL_SUPERVISION_SERIES" / "results" / "general" / "baseline" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def config():
    from experiment_config import load_locked_q20
    cfg = load_locked_q20(P / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.MODEL.DECODER.NUM_MODES = 20; cfg.device = "cpu"; cfg.device_list = ["cpu"]
    return cfg


def main():
    sys.path[:0] = [str(P), str(P / "scripts"), str(F)]
    os.chdir(P)
    import sitecustomize  # noqa
    from front_encoder_models import FrontEncoderASTRA
    torch.manual_seed(13)
    past = torch.randn(2, 9, 8, 2) * 0.1
    future = torch.randn(2, 1, 12, 2) * 0.2
    num = torch.tensor([4, 6])
    cache = ROOT / "_smoke_cache.npz"
    pseudo = np.random.default_rng(3).normal(0, 0.3, (4, 2, 12, 2)).astype(np.float32)
    confidence = np.array([[0.8, 0.7], [0.9, 0.0], [0.0, 0.0], [0.6, 0.6]], np.float32)
    valid = confidence > 0
    np.savez(cache, pseudo_futures=pseudo, confidence=confidence, valid_mask=valid, metadata_json=np.array(json.dumps({"bank_policy": "smoke"})))
    try:
        cfg = config(); model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
        os.environ["CFB2_SOURCE_CHECKPOINT"] = str(SOURCE); os.environ["CFB2_CACHE"] = str(cache)
        module = load("cfb2_smoke_model", ROOT / "cfb2_models.py")
        module.replace_decoder(model, cfg, "cfb2_confidence_gradient_isolated")
        model.eval(); model.decoder.set_scene_batch(torch.tensor([-2, -3]), past)
        output = model(past, future, None, mode="train", num_valid=num, graph_loc=past)[2]
        base = model.decoder.last_base_trajectories
        assert float((output - base).abs().max()) == 0.0
        loss, parts = module.build_loss(cfg, model)(output[:, :1], future, model.last_mode_logits[:, :1], cfg)
        loss.backward()
        adapter_grad = sum(float(p.grad.abs().sum()) for p in model.decoder.pseudo_adapter.parameters() if p.grad is not None)
        other_grad = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if "pseudo_adapter" not in n and p.grad is not None)
        assert adapter_grad > 0 and other_grad == 0
        before = output.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        model.decoder.set_scene_batch(torch.tensor([-2, -3]), past)
        after = model(past, future, None, mode="train", num_valid=num, graph_loc=past)[2].detach()
        assert float((after[:, :, :18] - before[:, :, :18]).abs().max()) == 0.0
        assert float((after[:, :, 18:20] - before[:, :, 18:20]).abs().max()) > 0.0
        print(f"[ok] zero_init_parity=True isolated_grad=True fixed_base18=True adapter_grad={adapter_grad:.4f} pseudo={float(parts['pseudo']):.6f}")
    finally:
        if cache.exists():
            cache.unlink()


if __name__ == "__main__":
    main()
