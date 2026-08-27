"""K20 with a small context-conditioned rotation of the fixed anchor set."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch
from torch import nn

BASE_PATH = Path(__file__).resolve().parents[2] / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
SPEC = importlib.util.spec_from_file_location("exp002_decoder_rotation", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(BASE)


class ConditionalRotationK20(BASE.KinematicAnchorDecoder):
    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(token_dim, pred_len, num_modes=20, hidden_dim=64, residual_modes=4)
        self.anchor_rotation = nn.Linear(64, 1)
        nn.init.zeros_(self.anchor_rotation.weight)
        nn.init.zeros_(self.anchor_rotation.bias)

    def forward(self, encoded_tokens: torch.Tensor, past_loc: torch.Tensor):
        pooled = encoded_tokens.mean(dim=-2)
        context = self.context(pooled)
        mode = self.mode_mlp(context[:, :, None, :] + self.mode_embedding[None, None])
        parameters = self.parameter_head(mode)
        last_step = past_loc[:, :, -1] - past_loc[:, :, -2]
        speed = torch.linalg.vector_norm(last_step, dim=-1, keepdim=True)
        default = torch.zeros_like(last_step); default[..., 0] = 1.0
        heading = torch.where(speed > 0.02, last_step / speed.clamp_min(1.0e-6), default)
        base_endpoint = last_step * float(self.pred_len)
        angle = (25.0 * math.pi / 180.0) * torch.tanh(self.anchor_rotation(context))
        cos, sin = torch.cos(angle), torch.sin(angle)
        anchors = self.anchors[None, None]
        ax, ay = anchors[..., 0], anchors[..., 1]
        rotated = torch.stack((ax * cos - ay * sin, ax * sin + ay * cos), dim=-1)
        local_endpoint = rotated + 0.75 * torch.tanh(parameters[..., :2])
        endpoint = base_endpoint[:, :, None] + self.rotate_local(local_endpoint, heading)
        alpha = self.alpha.view(1, 1, 1, self.pred_len, 1)
        relative = alpha * endpoint[..., None, :]
        coefficients = 0.35 * torch.tanh(parameters[..., 2:]).reshape(*parameters.shape[:-1], 4, 2)
        residual = torch.einsum("bakmc,mt->baktc", coefficients, self.sine_basis)
        trajectories = past_loc[:, :, None, -1:, :] + relative + residual
        logits = self.score_head(torch.cat((mode, parameters), dim=-1)).squeeze(-1)
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name != "conditional_rotation_k20": raise ValueError(name)
    model.decoder = ConditionalRotationK20(model.token_dim, model.pred_len)
    cfg.MODEL.DECODER.NUM_MODES = 20
