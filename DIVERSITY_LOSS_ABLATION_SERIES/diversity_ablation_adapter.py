"""EXP-173 V2 diversity-loss ablation adapter.

Keeps the exact JointTrainableV2Decoder architecture and initialization policy,
changing ONLY cfg.MODEL.DECODER.DIVERSITY_WEIGHT / DIVERSITY_MARGIN before the
locked ModeQueryLoss is constructed by FRONT_ENCODER_SERIES/train_front_variant.py.

Experiment name format:
    divw_0p050_m_0p400
    divw_0p000_m_0p400
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BASE_PATH = REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("exp173_joint_v2_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import EXP-173 joint V2 adapter: {BASE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()
_PATTERN = re.compile(r"^divw_(\d+)p(\d+)_m_(\d+)p(\d+)$")


def _decode_number(a: str, b: str) -> float:
    return float(f"{int(a)}.{b}")


def parse_experiment(name: str) -> tuple[float, float]:
    match = _PATTERN.match(name)
    if match is None:
        raise ValueError(
            f"Invalid diversity experiment name: {name!r}. "
            "Expected e.g. 'divw_0p025_m_0p400'."
        )
    weight = _decode_number(match.group(1), match.group(2))
    margin = _decode_number(match.group(3), match.group(4))
    if weight < 0.0:
        raise ValueError("Diversity weight must be non-negative")
    if margin < 0.0:
        raise ValueError("Diversity margin must be non-negative")
    return weight, margin


def replace_decoder(model, cfg, name: str) -> None:
    weight, margin = parse_experiment(name)

    # Exact EXP-173 V2 decoder construction + H3 checkpoint loading.
    BASE.replace_decoder(model, cfg, BASE.EXPERIMENT)

    # The only intended intervention.
    cfg.MODEL.DECODER.DIVERSITY_WEIGHT = float(weight)
    cfg.MODEL.DECODER.DIVERSITY_MARGIN = float(margin)

    # Diagnostic metadata only; does not affect forward pass.
    model.decoder.diversity_ablation_weight = float(weight)
    model.decoder.diversity_ablation_margin = float(margin)
    model.decoder.training_policy_name = (
        f"joint_encoder_h3_score_v2_divw_{weight:.3f}_margin_{margin:.3f}"
    )
