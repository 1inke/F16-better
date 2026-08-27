"""Frozen EXP-139C proposal/scoring model with an all-mode residual refiner.

The source model is supplied through EXP140_SOURCE_CHECKPOINT.  Every source
parameter is frozen; only the newly initialized refiner is trainable.  All 20
proposals are refined in both training and inference, so this experiment has
no hard-selection or teacher-forcing mismatch.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SOURCE_MODULE = ROOT.parent / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"
EXPERIMENTS = {"r1_frozen_all20_refine"}


def _load_source_module():
    spec = importlib.util.spec_from_file_location("exp139_decoder_source", SOURCE_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source_module()


class FrozenAll20Refiner(SOURCE.HierarchicalLearnedAnchorDecoder):
    """Three-anchor proposal model whose 20 trajectories are all refined."""

    def apply_training_policy(self, owner: torch.nn.Module) -> None:
        # The trainer calls model.train() every epoch.  Restore the frozen source
        # network to deterministic evaluation behavior and train only the refiner.
        owner.eval()
        if self.refiner is None:
            raise RuntimeError("The all-20 experiment requires a refiner")
        self.refiner.train(True)


def replace_decoder(model, cfg, name: str) -> None:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown EXP-140 decoder: {name}")
    checkpoint_text = os.environ.get("EXP140_SOURCE_CHECKPOINT", "").strip()
    if not checkpoint_text:
        raise RuntimeError("EXP140_SOURCE_CHECKPOINT must point to EXP-139C best_selected.pth")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model.decoder = FrozenAll20Refiner(
        token_dim=model.token_dim,
        pred_len=model.pred_len,
        num_modes=20,
        anchor_steps=(4, 8, model.pred_len),
        use_refiner=True,
        top_m=20,
        hidden_dim=64,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    expected_missing = {
        key for key in model.state_dict() if key.startswith("decoder.refiner.")
    }
    if missing != expected_missing:
        raise RuntimeError(
            f"Unexpected missing source keys: {sorted(missing)}; "
            f"expected only refiner keys: {sorted(expected_missing)}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected source checkpoint keys: {incompatible.unexpected_keys}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.decoder.refiner is not None
    for parameter in model.decoder.refiner.parameters():
        parameter.requires_grad_(True)

    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.frozen_source = True
    cfg.MODEL.DECODER.NUM_MODES = 20

