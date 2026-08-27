"""K20 with globally learnable but endpoint-preserving residual bases."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

BASE_PATH = Path(__file__).resolve().parents[2] / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
SPEC = importlib.util.spec_from_file_location("exp002_decoder_basis", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(BASE)


class LearnableBasisK20(BASE.KinematicAnchorDecoder):
    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(token_dim, pred_len, num_modes=20, hidden_dim=64, residual_modes=4)
        self.basis_delta = nn.Parameter(torch.zeros(4, pred_len))
        self.register_buffer("basis_envelope", torch.sin(torch.pi * self.alpha))

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
        local_endpoint = self.anchors[None, None] + 0.75 * torch.tanh(parameters[..., :2])
        endpoint = base_endpoint[:, :, None] + self.rotate_local(local_endpoint, heading)
        alpha = self.alpha.view(1, 1, 1, self.pred_len, 1)
        relative = alpha * endpoint[..., None, :]
        coefficients = 0.35 * torch.tanh(parameters[..., 2:]).reshape(*parameters.shape[:-1], 4, 2)
        basis = self.sine_basis + 0.25 * torch.tanh(self.basis_delta) * self.basis_envelope
        residual = torch.einsum("bakmc,mt->baktc", coefficients, basis)
        trajectories = past_loc[:, :, None, -1:, :] + relative + residual
        logits = self.score_head(torch.cat((mode, parameters), dim=-1)).squeeze(-1)
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name != "learnable_basis_k20": raise ValueError(name)
    model.decoder = LearnableBasisK20(model.token_dim, model.pred_len)
    cfg.MODEL.DECODER.NUM_MODES = 20
