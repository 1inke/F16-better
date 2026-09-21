"""
Adapter module for F16 Confidence Objective Refinement V2.

Reuses the exact EXP-173 joint V2 decoder initialization and architecture from
E2_F16_END_TO_END_COMPARISON_SERIES.joint_v2_adapter, while providing build_loss()
to inject ConfidenceObjectiveModeQueryLoss via the native train_front_variant.py interface.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from E2_F16_END_TO_END_COMPARISON_SERIES import joint_v2_adapter
from QUALITY_AWARE_F16_SERIES_V2.confidence_objective_loss import (
    ConfidenceObjectiveModeQueryLoss,
)


def replace_decoder(model, cfg, name: str):
    """
    Reuses the exact EXP-173 V2 decoder setup (H3 checkpoint loading + V2 refiner).
    """
    return joint_v2_adapter.replace_decoder(model, cfg, name)


def build_loss(cfg, model):
    """
    Builds the confidence objective loss from environment configuration.
    """
    return ConfidenceObjectiveModeQueryLoss.from_environment(cfg=cfg, model=model)
