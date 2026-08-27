"""K20 with a conservative context-gated velocity-trend prior."""

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


class GatedTrendK20(BASE.KinematicAnchorDecoder):
    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(token_dim, pred_len, 20, 64, 4)
        self.trend_gate = nn.Linear(64, 1)
        nn.init.zeros_(self.trend_gate.weight)
        nn.init.constant_(self.trend_gate.bias, -2.0)

    def forward(self, encoded_tokens: torch.Tensor, past_loc: torch.Tensor):
        pooled = encoded_tokens.mean(dim=-2)
        context = self.context(pooled)
        mode = self.mode_mlp(context[:, :, None, :] + self.mode_embedding[None, None])
        parameters = self.parameter_head(mode)
        last_step = past_loc[:, :, -1] - past_loc[:, :, -2]
        recent_average = (past_loc[:, :, -1] - past_loc[:, :, -4]) / 3.0
        trend = last_step - recent_average
        gate = torch.sigmoid(self.trend_gate(context))
        base_endpoint = last_step * float(self.pred_len) + gate * trend * float(self.pred_len)
        speed = torch.linalg.vector_norm(last_step, dim=-1, keepdim=True)
        default = torch.zeros_like(last_step)
        default[..., 0] = 1.0
        heading = torch.where(speed > 0.02, last_step / speed.clamp_min(1.0e-6), default)
        local_endpoint = self.anchors[None, None] + 0.75 * torch.tanh(parameters[..., :2])
        endpoint = base_endpoint[:, :, None] + self.rotate_local(local_endpoint, heading)
        alpha = self.alpha.view(1, 1, 1, self.pred_len, 1)
        relative = alpha * endpoint[..., None, :]
        coefficients = 0.35 * torch.tanh(parameters[..., 2:]).reshape(
            *parameters.shape[:-1], self.residual_modes, 2
        )
        residual = torch.einsum("bakmc,mt->baktc", coefficients, self.sine_basis)
        trajectories = past_loc[:, :, None, -1:, :] + relative + residual
        logits = self.score_head(torch.cat((mode, parameters), dim=-1)).squeeze(-1)
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name != "gated_trend_k20":
        raise ValueError(name)
    model.decoder = GatedTrendK20(model.token_dim, model.pred_len)
    cfg.MODEL.DECODER.NUM_MODES = 20

