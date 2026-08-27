"""K20 with fixed anchors fitted only on source training scenes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

BASE_PATH = Path(__file__).resolve().parents[1] / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
SPEC = importlib.util.spec_from_file_location("exp002_decoder", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


class DataDrivenK20(BASE.KinematicAnchorDecoder):
    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(token_dim, pred_len, num_modes=20, hidden_dim=64, residual_modes=4)
        anchor_path = Path(__file__).resolve().parent / "anchors_scene1_train_only.npy"
        anchors = np.load(anchor_path).astype(np.float32)
        if anchors.shape != (20, 2):
            raise ValueError(anchors.shape)
        self.anchors.copy_(torch.from_numpy(anchors))


def replace_decoder(model, cfg, name: str) -> None:
    if name != "data_anchor_k20":
        raise ValueError(name)
    model.decoder = DataDrivenK20(model.token_dim, model.pred_len)
    cfg.MODEL.DECODER.NUM_MODES = 20

