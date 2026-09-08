"""Training adapter: load a matched source checkpoint and fix time units."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp195_common import DECODER_SOURCE, load_module, patch_physical_time


def replace_decoder(model, cfg, name: str) -> None:
    if name != "exp195_finetune":
        raise ValueError(name)
    source = Path(os.environ.get("EXP195_SOURCE_CHECKPOINT", ""))
    if not source.is_file():
        raise FileNotFoundError(f"EXP195_SOURCE_CHECKPOINT missing: {source}")
    module = load_module("exp195_train_decoder", DECODER_SOURCE)
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
    model.decoder.exp195_time_unit_patch = True
    cfg.MODEL.DECODER.NUM_MODES = 20
