"""Load the common A1 source checkpoint and preserve physical time units."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "AV2_TEMPORAL_BRIDGE_SERIES")]

from exp195_common import load_module, patch_physical_time  # noqa: E402
from exp196_constants import DECODER_SOURCE  # noqa: E402


def replace_decoder(model, cfg, name: str) -> None:
    if name != "exp196_finetune":
        raise ValueError(name)
    source = Path(os.environ.get("EXP196_SOURCE_CHECKPOINT", ""))
    if not source.is_file():
        raise FileNotFoundError(f"EXP196_SOURCE_CHECKPOINT missing: {source}")
    module = load_module("exp196_train_decoder", DECODER_SOURCE)
    model.decoder = module.GroupedModeTemplateDecoder(
        model.token_dim, model.pred_len, use_template=True, use_groups=False
    )
    state = torch.load(source, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    patch_physical_time(model, float(cfg.DATA.FREQUENCY))
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.decoder.cv_reference = False
    model.decoder.source_checkpoint = str(source)
    model.decoder.exp196_time_unit_patch = True
    cfg.MODEL.DECODER.NUM_MODES = 20
