"""Trainable H3+V2 adapter for the matched EXP-173 comparison."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
CONTROLLED_PATH = ROOT.parent / "CONTROLLED_ANCHOR_CORRECTION_SERIES" / "controlled_anchor_correction_models.py"
EXPERIMENT = "v2_joint_wide_endpoint_correction"


def _load_controlled():
    spec = importlib.util.spec_from_file_location("exp173_controlled_source", CONTROLLED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(CONTROLLED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROLLED = _load_controlled()


class JointTrainableV2Decoder(CONTROLLED.FrozenControlledAnchorDecoder):
    """V2 geometry with encoder, H3 proposal, score and correction jointly trainable."""

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.train(True)


def replace_decoder(model, cfg, name: str) -> None:
    if name != EXPERIMENT:
        raise ValueError(f"Unknown EXP-173 decoder: {name}")
    source_text = os.environ.get("EXP173_H3_SOURCE_CHECKPOINT", "").strip()
    if not source_text:
        raise RuntimeError("EXP173_H3_SOURCE_CHECKPOINT is required")
    source = Path(source_text)
    if not source.is_file():
        raise FileNotFoundError(source)

    init_seed = int(os.environ.get("EXP173_DECODER_INIT_SEED", "173"))
    # Make the fresh V2 correction initialization independent of encoder RNG use.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed + 173000)
        decoder = JointTrainableV2Decoder(model.token_dim, model.pred_len)
    decoder.refiner.max_offsets.copy_(
        torch.tensor([0.20, 0.25, 0.50], dtype=torch.float32)
    )
    model.decoder = decoder
    state = torch.load(source, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {
        key for key in model.state_dict() if key.startswith("decoder.refiner.")
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            f"Unexpected H3 missing keys: {sorted(incompatible.missing_keys)}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected H3 keys: {incompatible.unexpected_keys}")

    # Do not freeze anything here. Intentional pre-existing constants such as
    # self_state_gate keep their original requires_grad flag.
    required_groups = {
        "encoder": any(
            parameter.requires_grad
            for key, parameter in model.named_parameters()
            if not key.startswith("decoder.")
        ),
        "h3_proposal": any(
            parameter.requires_grad
            for key, parameter in model.decoder.named_parameters()
            if not key.startswith("refiner.") and not key.startswith("score_head.")
        ),
        "score": any(
            parameter.requires_grad
            for parameter in model.decoder.score_head.parameters()
        ),
        "v2_correction": any(
            parameter.requires_grad for parameter in model.decoder.refiner.parameters()
        ),
    }
    if not all(required_groups.values()):
        raise RuntimeError(f"EXP-173 trainable group failure: {required_groups}")

    model.decoder.source_checkpoint = str(source)
    model.decoder.initialization_seed = init_seed
    model.decoder.training_policy_name = "joint_encoder_h3_score_v2"
    model.decoder.trainable_groups = required_groups
    model.decoder.cv_reference = False
    cfg.MODEL.DECODER.NUM_MODES = 20

