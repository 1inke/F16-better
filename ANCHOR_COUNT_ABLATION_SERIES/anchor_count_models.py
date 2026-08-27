"""Proposal-only learned-anchor count ablations for EXP-141."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT.parent / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"
EXPERIMENTS = {
    "a2_anchor_proposal": (6, 12),
    "a4_anchor_proposal": (3, 6, 9, 12),
    "a6_anchor_proposal": (2, 4, 6, 8, 10, 12),
}


def _load_source():
    spec = importlib.util.spec_from_file_location("exp139_anchor_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source()


def replace_decoder(model, cfg, name: str) -> None:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown EXP-141 decoder: {name}")
    anchor_steps = tuple(
        model.pred_len if step == 12 else step for step in EXPERIMENTS[name]
    )
    model.decoder = SOURCE.HierarchicalLearnedAnchorDecoder(
        token_dim=model.token_dim,
        pred_len=model.pred_len,
        num_modes=20,
        anchor_steps=anchor_steps,
        use_refiner=False,
        top_m=8,
        hidden_dim=64,
    )
    cfg.MODEL.DECODER.NUM_MODES = 20

