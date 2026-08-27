"""K20 with two sine residual modes instead of four."""

from __future__ import annotations

import importlib.util
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[1] / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
SPEC = importlib.util.spec_from_file_location("exp002_decoder", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def replace_decoder(model, cfg, name: str) -> None:
    if name != "low_rank_k20":
        raise ValueError(name)
    model.decoder = BASE.KinematicAnchorDecoder(
        token_dim=model.token_dim,
        pred_len=model.pred_len,
        num_modes=20,
        hidden_dim=64,
        residual_modes=2,
    )
    cfg.MODEL.DECODER.NUM_MODES = 20

