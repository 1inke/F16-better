"""K20 decoder with explicit candidate-geometry scoring features."""

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


class GeometryAwareK20(BASE.KinematicAnchorDecoder):
    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(token_dim, pred_len, num_modes=20, hidden_dim=64, residual_modes=4)
        parameter_dim = 2 + 2 * self.residual_modes
        geometry_dim = 6
        self.geometry_score_head = nn.Sequential(
            nn.LayerNorm(64 + parameter_dim + geometry_dim),
            nn.Linear(64 + parameter_dim + geometry_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        del self.score_head

    def forward(self, encoded_tokens: torch.Tensor, past_loc: torch.Tensor):
        pooled = encoded_tokens.mean(dim=-2)
        context = self.context(pooled)
        mode = self.mode_mlp(context[:, :, None, :] + self.mode_embedding[None, None])
        parameters = self.parameter_head(mode)
        last_step = past_loc[:, :, -1] - past_loc[:, :, -2]
        speed = torch.linalg.vector_norm(last_step, dim=-1, keepdim=True)
        default = torch.zeros_like(last_step)
        default[..., 0] = 1.0
        heading = torch.where(speed > 0.02, last_step / speed.clamp_min(1.0e-6), default)
        base_endpoint = last_step * float(self.pred_len)
        local_endpoint = self.anchors[None, None] + 0.75 * torch.tanh(parameters[..., :2])
        endpoint = base_endpoint[:, :, None] + self.rotate_local(local_endpoint, heading)
        alpha = self.alpha.view(1, 1, 1, self.pred_len, 1)
        relative = alpha * endpoint[..., None, :]
        coefficients = 0.35 * torch.tanh(parameters[..., 2:]).reshape(
            *parameters.shape[:-1], self.residual_modes, 2
        )
        residual = torch.einsum("bakmc,mt->baktc", coefficients, self.sine_basis)
        trajectories = past_loc[:, :, None, -1:, :] + relative + residual

        base_length = torch.linalg.vector_norm(base_endpoint, dim=-1, keepdim=True)
        base_length = base_length[:, :, None].expand(-1, -1, self.num_modes, -1)
        endpoint_length = torch.linalg.vector_norm(endpoint, dim=-1, keepdim=True)
        coefficient_energy = coefficients.square().mean(dim=-2)
        geometry = torch.cat(
            (local_endpoint, base_length, endpoint_length, coefficient_energy), dim=-1
        )
        score_input = torch.cat((mode, parameters, geometry), dim=-1)
        logits = self.geometry_score_head(score_input).squeeze(-1)
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name != "geometry_score_k20":
        raise ValueError(name)
    model.decoder = GeometryAwareK20(model.token_dim, model.pred_len)
    cfg.MODEL.DECODER.NUM_MODES = 20

