"""ETSE-E1: learned relation bias aligned with each neighbour-time token.

This plug-in leaves the sparse F16 graph, temporal encoder and three-anchor
decoder intact.  Only a zero-initialised ``r_ij^t -> per-head bias`` projection
is introduced, so its initial forward pass is exactly the loaded E0 model.
"""
from __future__ import annotations

import math
import types
from typing import Optional

import torch
from torch import Tensor, nn

EXPERIMENTS = {"e1_time_aligned_relation_bias"}


class TimeAlignedRelationBias(nn.Module):
    def __init__(self, heads: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(8), nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, heads)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.enabled = True
        self.last_bias: Optional[Tensor] = None

    def forward(self, features: Tensor) -> Tensor:
        # features: B,N(query),N(key),T,8 -> B,H,N,N,T
        bias = self.net(features).permute(0, 4, 1, 2, 3)
        agents = bias.shape[2]
        eye = torch.eye(agents, dtype=torch.bool, device=bias.device)[None, None, :, :, None]
        bias = bias.masked_fill(eye, 0.0)
        if not self.enabled:
            bias = torch.zeros_like(bias)
        self.last_bias = bias.detach()
        return bias


def _relation_tokens_with_tap(self, past: Tensor):
    features, last_distance, last_closing = self.relation_features(past)
    self.last_etse_relation_features = features
    encoded = self.relation_encoder(features)
    if self.use_multiscale_relation:
        recent2, recent4, full = encoded[:, :, :, -2:].mean(3), encoded[:, :, :, -4:].mean(3), encoded.mean(3)
        scales = torch.stack((recent2, recent4, full), dim=3)
        tokens = (torch.softmax(self.relation_scale_score(scales), dim=3) * scales).sum(3)
    else:
        tokens = (torch.softmax(self.relation_time_score(encoded), dim=3) * encoded).sum(3)
    return tokens, last_distance, last_closing, features[:, :, :, -1, 6] * 10.0 + features[:, :, :, -1, 7]


def _temporal_social_attention_e1(self, temporal: Tensor, agent: Tensor, relation: Tensor, allowed: Tensor, type_features=None, risk_score=None) -> Tensor:
    del type_features, risk_score
    batch, agents, steps, _ = temporal.shape
    features = self.last_etse_relation_features
    if features is None or features.shape[3] != steps:
        raise RuntimeError("ETSE relation-time features are unavailable or misaligned.")
    q = self.q_proj(agent).reshape(batch, agents, self.heads, self.qk_head_dim).permute(0, 2, 1, 3)
    k = self.k_proj(temporal).reshape(batch, agents, steps, self.kv_heads, self.qk_head_dim).permute(0, 3, 1, 2, 4)
    v = self.v_proj(temporal).reshape(batch, agents, steps, self.kv_heads, self.head_dim).permute(0, 3, 1, 2, 4)
    if self.kv_heads != self.heads:
        repeats = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
    scores = torch.einsum("bhid,bhjtd->bhijt", q, k) / math.sqrt(float(self.qk_head_dim))
    scores = scores + self.relation_bias(relation).permute(0, 3, 1, 2)[..., None]
    scores = scores + self.etse_time_bias(features)
    mask = allowed[:, None, :, :, None].expand_as(scores) if allowed.ndim == 3 else allowed[:, :, :, :, None].expand_as(scores)
    weights = torch.softmax(scores.masked_fill(~mask, -1.0e4).flatten(-2), dim=-1).reshape_as(scores)
    edge_value = self.relation_value(relation).reshape(batch, agents, agents, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
    message = v[:, :, None].expand(-1, -1, agents, -1, -1, -1) + edge_value[..., None, :]
    social = (weights[..., None] * message).sum((3, 4)).permute(0, 2, 1, 3).reshape(batch, agents, self.dim)
    return self.out_proj(social)


def replace_encoder(model: nn.Module, cfg, experiment: str) -> None:
    del cfg
    if experiment not in EXPERIMENTS or model.front_variant != "f16_temporal_social_cross":
        raise ValueError("ETSE-E1 requires the locked f16_temporal_social_cross model.")
    base = model.agent_aware_trans_encoder
    forbidden = ("use_pair_deformable_temporal", "use_pair_temporal_soft_prior", "use_scaled_cosine_attention", "use_differential_attention", "use_relation_type_bias", "use_soft_risk_bias", "use_temporal_relative_bias")
    active = [x for x in forbidden if bool(getattr(base, x, False))]
    if active or not base.use_relation_tokens or not base.use_temporal_social_cross:
        raise ValueError(f"ETSE-E1 incompatible F16 state: {active}")
    base.etse_time_bias = TimeAlignedRelationBias(int(base.heads))
    base.last_etse_relation_features = None
    base.relation_tokens = types.MethodType(_relation_tokens_with_tap, base)
    base.temporal_social_attention = types.MethodType(_temporal_social_attention_e1, base)
