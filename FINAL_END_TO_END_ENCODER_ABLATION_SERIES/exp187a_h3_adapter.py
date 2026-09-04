"""H3 adapter with encoder-independent matched initialization for EXP-187A."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT.parent / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"


def _load_source():
    spec = importlib.util.spec_from_file_location("exp187a_h3_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source()


def replace_decoder(model, cfg, name: str) -> None:
    if name != "h3_anchor_proposal":
        raise ValueError(f"Unknown EXP-187A H3 decoder: {name}")
    seed = int(os.environ.get("EXP187A_H3_INIT_SEED", "187001"))
    # Encoder variants have different parameter counts and therefore consume
    # different RNG streams during construction. Isolate decoder construction
    # so E0 and E1 receive bit-identical H3 initial weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        SOURCE.replace_decoder(model, cfg, name)
    model.decoder.initialization_seed = seed
    model.decoder.training_policy_name = "joint_encoder_h3_score"

