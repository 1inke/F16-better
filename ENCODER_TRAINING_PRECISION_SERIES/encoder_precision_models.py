"""Single-variable encoder plugins built on the locked F16 front encoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class StateChannelGate(nn.Module):
    """Recalibrate the existing robust-state token without adding state variables."""

    def __init__(self, base: nn.Module, token_dim: int) -> None:
        super().__init__()
        self.base = base
        feature_dim = int(base.feature_dim)
        hidden = 24
        self.channel_scale = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim),
        )
        nn.init.zeros_(self.channel_scale[-1].weight)
        nn.init.zeros_(self.channel_scale[-1].bias)

    def forward(self, past: Tensor, frequency: float) -> Tensor:
        encoded = self.base(past, frequency)
        features = self.base.features(past, frequency)
        scale = 1.0 + 0.5 * torch.tanh(self.channel_scale(features))
        return encoded * scale


class RelationChannelGate(nn.Module):
    """Recalibrate only the eight established pair-relation channels."""

    def __init__(self, base: nn.Module, token_dim: int) -> None:
        super().__init__()
        self.base = base
        self.channel_scale = nn.Sequential(
            nn.LayerNorm(8), nn.Linear(8, 16), nn.GELU(), nn.Linear(16, token_dim)
        )
        nn.init.zeros_(self.channel_scale[-1].weight)
        nn.init.zeros_(self.channel_scale[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        encoded = self.base(features)
        scale = 1.0 + 0.5 * torch.tanh(self.channel_scale(features))
        return encoded * scale


class ZeroGatedDepthwiseTCN(nn.Module):
    """Two-scale temporal residual that starts exactly on the F16 path."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dilated_1 = nn.Conv1d(
            dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim
        )
        self.dilated_2 = nn.Conv1d(
            dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim
        )
        self.channel_scale = nn.Parameter(torch.ones(dim))
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens: Tensor, past: Tensor, frequency: float) -> Tensor:
        del past, frequency
        batch, agents, steps, dim = tokens.shape
        values = self.norm(tokens).reshape(batch * agents, steps, dim).transpose(1, 2)
        local = F.gelu(self.dilated_1(values))
        multiscale = self.dilated_2(local).transpose(1, 2).reshape(
            batch, agents, steps, dim
        )
        update = multiscale * self.channel_scale.view(1, 1, 1, dim)
        return tokens + torch.tanh(self.residual_gate) * update


class TemporalRoPEEncoder(nn.Module):
    """One norm-first Transformer layer with RoPE applied only to temporal Q/K."""

    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("RoPE temporal dimension must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = dim // heads
        if self.head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout_1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.ffn_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.residual_dropout_2 = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(dim)

    def apply_rope(self, values: Tensor) -> Tensor:
        steps = values.shape[-2]
        half = self.head_dim // 2
        positions = torch.arange(steps, device=values.device, dtype=values.dtype)
        frequencies = torch.arange(half, device=values.device, dtype=values.dtype)
        frequencies = torch.exp(
            -math.log(10000.0) * (2.0 * frequencies / float(self.head_dim))
        )
        angles = positions[:, None] * frequencies[None]
        cosine = angles.cos().view(1, 1, steps, half)
        sine = angles.sin().view(1, 1, steps, half)
        paired = values.reshape(*values.shape[:-1], half, 2)
        even, odd = paired[..., 0], paired[..., 1]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)

    def forward(self, tokens: Tensor) -> Tensor:
        batch_agents, steps, dim = tokens.shape
        normalized = self.norm1(tokens)

        def heads(projection: nn.Module) -> Tensor:
            return projection(normalized).reshape(
                batch_agents, steps, self.heads, self.head_dim
            ).transpose(1, 2)

        query = self.apply_rope(heads(self.q_proj))
        key = self.apply_rope(heads(self.k_proj))
        value = heads(self.v_proj)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            float(self.head_dim)
        )
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch_agents, steps, dim
        )
        tokens = tokens + self.residual_dropout_1(self.out_proj(context))
        feedforward = self.linear2(
            self.ffn_dropout(F.gelu(self.linear1(self.norm2(tokens))))
        )
        tokens = tokens + self.residual_dropout_2(feedforward)
        return self.final_norm(tokens)


def replace_encoder(model, cfg, name: str) -> None:
    encoder = model.agent_aware_trans_encoder
    if name == "state_channel_gate":
        if model.robust_state is None:
            raise ValueError("state_channel_gate requires the F16 robust state branch")
        model.robust_state = StateChannelGate(model.robust_state, model.token_dim)
    elif name == "relation_channel_gate":
        encoder.relation_encoder = RelationChannelGate(
            encoder.relation_encoder, model.token_dim
        )
    elif name == "tcn_residual":
        encoder.multiscale_temporal = ZeroGatedDepthwiseTCN(model.token_dim)
    elif name == "temporal_rope":
        encoder.temporal_encoder = TemporalRoPEEncoder(
            dim=model.token_dim,
            heads=int(cfg.MODEL.NHEAD),
            ffn_dim=int(cfg.MODEL.DIM_FEEDFORWARD),
            dropout=float(cfg.MODEL.DROPOUT),
        )
    else:
        raise ValueError(f"Unknown encoder precision experiment: {name}")
    model.encoder_experiment = name
