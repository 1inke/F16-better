"""Freeze the exact fold-local official V2 decoder while training an encoder."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
CONTROLLED_PATH = ROOT.parent / "CONTROLLED_ANCHOR_CORRECTION_SERIES" / "controlled_anchor_correction_models.py"


def _load_controlled():
    spec = importlib.util.spec_from_file_location("official_ablation_v2_source", CONTROLLED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(CONTROLLED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROLLED = _load_controlled()


class FrozenOfficialV2Decoder(CONTROLLED.FrozenControlledAnchorDecoder):
    """Final V2 geometry and score, frozen; keep the selected encoder trainable."""

    def apply_training_policy(self, owner: torch.nn.Module) -> None:
        owner.train(True)
        self.eval()


def replace_decoder(model, cfg, name: str) -> None:
    if name != "fixed_official_v2":
        raise ValueError(name)
    checkpoint_text = os.environ.get("OFFICIAL_V2_DECODER_SOURCE", "").strip()
    if not checkpoint_text:
        raise RuntimeError("OFFICIAL_V2_DECODER_SOURCE is required")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    decoder = FrozenOfficialV2Decoder(model.token_dim, model.pred_len)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    decoder_state = {
        key[len("decoder."):]: value
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        raise RuntimeError(f"No decoder state in {checkpoint}")
    decoder.load_state_dict(decoder_state, strict=True)
    model.decoder = decoder
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(False)
    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.ablation_policy = "freeze_exact_fold_local_v2"
    model.decoder.cv_reference = False
    cfg.MODEL.DECODER.NUM_MODES = 20
