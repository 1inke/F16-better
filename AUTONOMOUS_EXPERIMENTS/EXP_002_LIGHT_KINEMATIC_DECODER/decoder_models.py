"""Lightweight kinematic-anchor decoders for EXP-002."""

from __future__ import annotations

import math

import torch
from torch import nn


def make_anchors(num_modes: int) -> torch.Tensor:
    anchors = [(0.0, 0.0)]
    for longitudinal in (-1.5, -0.75, 0.0, 0.75, 1.5):
        for lateral in (-0.8, -0.25, 0.25, 0.8):
            if len(anchors) < num_modes:
                anchors.append((longitudinal, lateral))
    return torch.tensor(anchors[:num_modes], dtype=torch.float32)


class KinematicAnchorDecoder(nn.Module):
    """CV trajectory plus heading-aligned anchors and low-rank sine residuals."""

    def __init__(
        self,
        token_dim: int,
        pred_len: int,
        num_modes: int,
        hidden_dim: int = 64,
        residual_modes: int = 4,
    ) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.pred_len = int(pred_len)
        self.residual_modes = int(residual_modes)
        self.context = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
        )
        self.mode_embedding = nn.Parameter(torch.empty(num_modes, hidden_dim))
        nn.init.normal_(self.mode_embedding, std=0.02)
        self.mode_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        parameter_dim = 2 + 2 * residual_modes
        self.parameter_head = nn.Linear(hidden_dim, parameter_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + parameter_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.parameter_head.weight)
        nn.init.zeros_(self.parameter_head.bias)
        self.register_buffer("anchors", make_anchors(num_modes))
        alpha = torch.arange(1, pred_len + 1, dtype=torch.float32) / float(pred_len)
        sine = torch.stack(
            [torch.sin(math.pi * (index + 1) * alpha) for index in range(residual_modes)]
        )
        self.register_buffer("alpha", alpha)
        self.register_buffer("sine_basis", sine)

    @staticmethod
    def rotate_local(vectors: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        x, y = vectors[..., 0], vectors[..., 1]
        cos = heading[..., 0].unsqueeze(-1)
        sin = heading[..., 1].unsqueeze(-1)
        return torch.stack((x * cos - y * sin, x * sin + y * cos), dim=-1)

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
        logits = self.score_head(torch.cat((mode, parameters), dim=-1)).squeeze(-1)
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    settings = {
        "kinematic_k12": (12, 3),
        "kinematic_k20": (20, 4),
    }
    if name not in settings:
        raise ValueError(f"Unknown EXP-002 decoder: {name}")
    num_modes, residual_modes = settings[name]
    model.decoder = KinematicAnchorDecoder(
        token_dim=model.token_dim,
        pred_len=model.pred_len,
        num_modes=num_modes,
        hidden_dim=64,
        residual_modes=residual_modes,
    )
    cfg.MODEL.DECODER.NUM_MODES = num_modes

