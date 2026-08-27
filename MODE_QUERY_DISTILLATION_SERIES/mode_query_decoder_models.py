"""K20 decoders with explicit geometry queries and mode-specific token retrieval."""

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


class GeometryModeQueryKinematicDecoder(nn.Module):
    """Preserve K20 geometry while conditioning each mode on its fixed anchor."""

    def __init__(
        self,
        token_dim: int,
        pred_len: int,
        num_modes: int = 20,
        hidden_dim: int = 64,
        residual_modes: int = 4,
        use_temporal_cross: bool = False,
        attention_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.pred_len = int(pred_len)
        self.residual_modes = int(residual_modes)
        self.use_temporal_cross = bool(use_temporal_cross)
        self.context = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU()
        )
        self.mode_embedding = nn.Parameter(torch.empty(num_modes, hidden_dim))
        nn.init.normal_(self.mode_embedding, std=0.02)
        self.geometry_mlp = nn.Sequential(
            nn.LayerNorm(4), nn.Linear(4, 32), nn.GELU(), nn.Linear(32, hidden_dim)
        )
        nn.init.zeros_(self.geometry_mlp[-1].weight)
        nn.init.zeros_(self.geometry_mlp[-1].bias)
        if self.use_temporal_cross:
            self.query_norm = nn.LayerNorm(hidden_dim)
            self.cross_q = nn.Linear(hidden_dim, attention_dim)
            self.cross_k = nn.Linear(token_dim, attention_dim)
            self.cross_v = nn.Linear(token_dim, attention_dim)
            self.cross_out = nn.Linear(attention_dim, hidden_dim)
            nn.init.zeros_(self.cross_out.weight)
            nn.init.zeros_(self.cross_out.bias)
            self.cross_gate = nn.Parameter(torch.full((num_modes,), -2.2))
            self.last_temporal_attention = None
        self.mode_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        parameter_dim = 2 + 2 * residual_modes
        self.parameter_head = nn.Linear(hidden_dim, parameter_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + parameter_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
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
        self.last_mode_context = None

    def geometry_features(self) -> torch.Tensor:
        anchors = self.anchors
        radius = torch.linalg.vector_norm(anchors, dim=-1, keepdim=True)
        angle = torch.atan2(anchors[:, 1], anchors[:, 0]).unsqueeze(-1) / math.pi
        return torch.cat(
            (anchors[:, :1] / 1.5, anchors[:, 1:] / 0.8, radius / 1.7, angle),
            dim=-1,
        )

    @staticmethod
    def rotate_local(vectors: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        x, y = vectors[..., 0], vectors[..., 1]
        cos = heading[..., 0].unsqueeze(-1)
        sin = heading[..., 1].unsqueeze(-1)
        return torch.stack((x * cos - y * sin, x * sin + y * cos), dim=-1)

    def forward(self, encoded_tokens: torch.Tensor, past_loc: torch.Tensor):
        pooled = encoded_tokens.mean(dim=-2)
        shared_context = self.context(pooled)
        geometry = self.geometry_mlp(self.geometry_features())
        mode_seed = (
            shared_context[:, :, None, :]
            + self.mode_embedding[None, None]
            + geometry[None, None]
        )
        if self.use_temporal_cross:
            query = self.cross_q(self.query_norm(mode_seed))
            key = self.cross_k(encoded_tokens)
            value = self.cross_v(encoded_tokens)
            scores = torch.einsum("bakd,batd->bakt", query, key)
            scores = scores / math.sqrt(float(query.shape[-1]))
            weights = torch.softmax(scores, dim=-1)
            retrieved = torch.einsum("bakt,batd->bakd", weights, value)
            delta = self.cross_out(retrieved)
            gate = torch.sigmoid(self.cross_gate).view(1, 1, self.num_modes, 1)
            mode_seed = mode_seed + gate * delta
            self.last_temporal_attention = weights.detach()
        self.last_mode_context = mode_seed.detach()
        mode = self.mode_mlp(mode_seed)
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
        "mq1_geometry_k20": False,
        "mq2_temporal_k20": True,
    }
    if name not in settings:
        raise ValueError(f"Unknown mode-query decoder: {name}")
    model.decoder = GeometryModeQueryKinematicDecoder(
        token_dim=model.token_dim,
        pred_len=model.pred_len,
        num_modes=20,
        hidden_dim=64,
        residual_modes=4,
        use_temporal_cross=settings[name],
        attention_dim=32,
    )
    cfg.MODEL.DECODER.NUM_MODES = 20

