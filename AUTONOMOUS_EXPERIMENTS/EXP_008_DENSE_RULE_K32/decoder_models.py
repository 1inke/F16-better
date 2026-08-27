"""Denser symmetric rule anchors with the lightweight decoder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

BASE_PATH = Path(__file__).resolve().parents[1] / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
SPEC = importlib.util.spec_from_file_location("exp002_decoder", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def dense_anchors() -> torch.Tensor:
    points = [(0.0, 0.0)]
    candidates = []
    for longitudinal in (-2.0, -1.2, -0.6, 0.0, 0.6, 1.2, 2.0):
        for lateral in (-1.2, -0.6, 0.0, 0.6, 1.2):
            if longitudinal == 0.0 and lateral == 0.0:
                continue
            candidates.append((longitudinal, lateral))
    candidates.sort(key=lambda item: (item[0] ** 2 + item[1] ** 2, abs(item[0]), item[0], item[1]))
    points.extend(candidates[:31])
    return torch.tensor(points, dtype=torch.float32)


class DenseRuleK32(BASE.KinematicAnchorDecoder):
    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(token_dim, pred_len, num_modes=20, hidden_dim=64, residual_modes=4)
        self.num_modes = 32
        self.mode_embedding = nn.Parameter(torch.empty(32, 64))
        nn.init.normal_(self.mode_embedding, std=0.02)
        self.anchors = dense_anchors()


def replace_decoder(model, cfg, name: str) -> None:
    if name != "dense_rule_k32":
        raise ValueError(name)
    model.decoder = DenseRuleK32(model.token_dim, model.pred_len)
    cfg.MODEL.DECODER.NUM_MODES = 32

