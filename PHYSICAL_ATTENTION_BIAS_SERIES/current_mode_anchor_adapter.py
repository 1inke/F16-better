"""Load the selected EXP-146B CV-free mode/three-anchor decoder for PB tests.

The source checkpoint is loaded after the physical-attention encoder plugin is
installed.  Decoder and non-interaction modules remain frozen; the F16
temporal/social interaction encoder is fine-tuned identically in B0/PB1/PB2/PB3.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
SOURCE_MODULE = ROOT.parent / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"


def _load_source_module():
    spec = importlib.util.spec_from_file_location("pb_current_mode_anchor_source", SOURCE_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source_module()


class InteractionFineTuneModeAnchorDecoder(SOURCE.GroupedModeTemplateDecoder):
    """EXP-146B decoder kept in evaluation mode during encoder fine-tuning."""

    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(token_dim, pred_len, use_template=True, use_groups=False)

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.train(True)
        self.eval()


def _compatible_checkpoint_state(model: nn.Module, source: dict[str, torch.Tensor]):
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    skipped = sorted(set(source) - set(compatible))
    return compatible, skipped


def replace_decoder(model, cfg, name: str) -> None:
    if name != "current_mode_anchor_template":
        raise ValueError(f"Unknown current PB decoder: {name}")
    checkpoint_text = os.environ.get("PHYSICAL_BIAS_SOURCE_CHECKPOINT", "").strip()
    if not checkpoint_text:
        raise RuntimeError("PHYSICAL_BIAS_SOURCE_CHECKPOINT is required")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model.decoder = InteractionFineTuneModeAnchorDecoder(model.token_dim, model.pred_len)
    source_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    compatible, skipped_source = _compatible_checkpoint_state(model, source_state)
    incompatible = model.load_state_dict(compatible, strict=False)

    # A physical-bias plugin adds its own zero-initialized RBF coefficients.
    # PB2/PB3 intentionally remove some old relation modules, so their source
    # keys are skipped.  All decoder keys must still load exactly.
    missing_decoder = [key for key in incompatible.missing_keys if key.startswith("decoder.")]
    skipped_decoder = [key for key in skipped_source if key.startswith("decoder.")]
    if missing_decoder or skipped_decoder:
        raise RuntimeError(
            f"EXP-146B decoder checkpoint mismatch: missing={missing_decoder}, "
            f"skipped={skipped_decoder}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.agent_aware_trans_encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(False)

    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.source_method = "EXP-146B anchored mode template"
    model.decoder.cv_reference = False
    model.decoder.skipped_source_keys = skipped_source
    model.decoder.missing_plugin_keys = sorted(incompatible.missing_keys)
    cfg.MODEL.DECODER.NUM_MODES = 20
