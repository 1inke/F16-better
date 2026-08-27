"""Initialization, grouping and gradient checks for EXP-146."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent; WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
SOURCE = WORKSPACE / "ANCHOR_CORRECTION_VALIDATION_SERIES" / "EXP_143B_WIDE_ENDPOINT_CORRECTION" / "results" / "general" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"


def load(path: Path):
    spec = importlib.util.spec_from_file_location("exp146_smoke", path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]; os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from locked_mode_query_loss import ModeQueryLoss
    os.environ["EXP146_SOURCE_CHECKPOINT"] = str(SOURCE)
    custom = load(ROOT / "grouped_mode_template_models.py")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for name, (use_template, use_groups) in custom.EXPERIMENTS.items():
        cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
        cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
        cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0
        cfg.device = str(device); cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
        model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross"); custom.replace_decoder(model, cfg, name)
        model = model.to(device); model.train(); model.decoder.apply_training_policy(model)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        allowed = ("decoder.template_head.", "decoder.group_head.", "decoder.group_embedding")
        if not trainable or any(not n.startswith(allowed) for n in trainable): raise AssertionError(trainable)
        counts = torch.bincount(model.decoder.mode_groups.cpu(), minlength=5)
        if (counts == 0).any(): raise AssertionError(counts)
        past = torch.randn(16, 9, 8, 2, device=device) * 0.2
        future = torch.randn(16, 9, 12, 2, device=device) * 0.3
        valid = torch.full((16,), 9, device=device)
        prediction = model(past, future, None, mode="train", num_valid=valid, graph_loc=past)[2]
        source_error = (prediction - model.decoder.last_coarse_trajectories).abs().max()
        if source_error > 2e-6: raise AssertionError(source_error)
        logits = model.last_mode_logits
        loss, _ = ModeQueryLoss(cfg).to(device)(prediction[:, :1], future[:, :1], logits[:, :1], cfg)
        auxiliary = model.decoder.auxiliary_training_loss(prediction[:, :1], future[:, :1], logits[:, :1])
        total = loss + auxiliary; total.backward()
        bad = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        if bad: raise AssertionError(bad)
        print(f"[ok] {name} template={use_template} groups={use_groups} group_sizes={counts.tolist()} total={sum(p.numel() for p in model.parameters())} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)} aux={auxiliary.item():.4f}")


if __name__ == "__main__": main()
