"""Frozen current EXP-146B decoder for the matched ICMAB baseline."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT.parent / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"


def _load_source():
    spec = importlib.util.spec_from_file_location("icmab_baseline_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source()


class FrozenCurrentDecoder(SOURCE.GroupedModeTemplateDecoder):
    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(token_dim, pred_len, use_template=True, use_groups=False)

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.train(True)
        self.eval()


def replace_decoder(model, cfg, name: str) -> None:
    if name != "current_mode_anchor_template":
        raise ValueError(name)
    checkpoint = Path(os.environ.get("ICMAB_SOURCE_CHECKPOINT", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model.decoder = FrozenCurrentDecoder(model.token_dim, model.pred_len)
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    missing_decoder = [key for key in incompatible.missing_keys if key.startswith("decoder.")]
    skipped_decoder = [key for key in source if key.startswith("decoder.") and key not in compatible]
    if missing_decoder or skipped_decoder:
        raise RuntimeError((missing_decoder, skipped_decoder))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.agent_aware_trans_encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(False)
    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.cv_reference = False
    cfg.MODEL.DECODER.NUM_MODES = 20
