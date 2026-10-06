"""Parameter-free edge-first execution adapter for the locked F16 encoder.

This module changes tensor layout and reduction order only.  It intentionally
adds no parameters and is attached after the original checkpoint is loaded.
"""
from __future__ import annotations

import math
import types
from typing import Optional

import torch
from torch import Tensor, nn


def edge_relation_features(base: nn.Module, past: Tensor, edges: Tensor) -> Tensor:
    """Compute the original eight relation channels only for selected edges."""
    b, target, source = edges.unbind(dim=1)
    velocity = torch.zeros_like(past)
    velocity[:, :, 1:] = (past[:, :, 1:] - past[:, :, :-1]) * base.frequency
    velocity[:, :, 0] = velocity[:, :, 1]
    # Preserve the existing dense tensor convention exactly:
    # relation[b, target, source] = state[source] - state[target].
    relative_position = past[b, source] - past[b, target]
    relative_velocity = velocity[b, source] - velocity[b, target]
    distance = torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True)
    dot = (relative_position * relative_velocity).sum(dim=-1, keepdim=True)
    velocity_sq = relative_velocity.square().sum(dim=-1, keepdim=True)
    closing = -dot / distance.clamp_min(1.0e-3)
    ttc = (-dot / velocity_sq.clamp_min(1.0e-4)).clamp(0.0, 10.0)
    closest = relative_position + ttc * relative_velocity
    dca = torch.linalg.vector_norm(closest, dim=-1, keepdim=True)
    return torch.cat(
        (relative_position, relative_velocity, distance, closing, ttc / 10.0, dca),
        dim=-1,
    )


def edge_relation_tokens(base: nn.Module, features: Tensor) -> tuple[Tensor, Tensor]:
    encoded = base.relation_encoder(features)
    if base.use_multiscale_relation:
        recent2 = encoded[:, -2:].mean(dim=1)
        recent4 = encoded[:, -4:].mean(dim=1)
        full = encoded.mean(dim=1)
        scales = torch.stack((recent2, recent4, full), dim=1)
        weights = torch.softmax(base.relation_scale_score(scales), dim=1)
        tokens = (weights * scales).sum(dim=1)
    else:
        weights = torch.softmax(base.relation_time_score(encoded), dim=1)
        tokens = (weights * encoded).sum(dim=1)
    return tokens, encoded


def _edge_attention(
    base: nn.Module,
    temporal: Tensor,
    agent: Tensor,
    edges: Tensor,
    relation: Tensor,
    allowed: Tensor,
) -> Tensor:
    batch, agents, steps, _ = temporal.shape
    b, target, source = edges.unbind(dim=1)
    q = base.q_proj(agent).reshape(batch, agents, base.heads, base.qk_head_dim)
    k = base.k_proj(temporal).reshape(
        batch, agents, steps, base.heads, base.qk_head_dim
    )
    v = base.v_proj(temporal).reshape(
        batch, agents, steps, base.heads, base.head_dim
    )
    scores = torch.einsum("ehd,ethd->eht", q[b, target], k[b, source])
    scores = scores / math.sqrt(float(base.qk_head_dim))
    scores = scores + base.relation_bias(relation)[..., None]

    heads = torch.arange(base.heads, device=temporal.device)[None]
    groups = ((b[:, None] * agents + target[:, None]) * base.heads + heads)
    flat_groups = groups[..., None].expand(-1, -1, steps).reshape(-1)
    flat_scores = scores.reshape(-1)
    group_count = batch * agents * base.heads
    maxima = torch.full(
        (group_count,), -torch.inf, device=scores.device, dtype=scores.dtype
    )
    maxima.scatter_reduce_(
        0, flat_groups, flat_scores, reduce="amax", include_self=True
    )
    exponent = torch.exp(flat_scores - maxima[flat_groups])
    denominator = torch.zeros(group_count, device=scores.device, dtype=scores.dtype)
    denominator.scatter_add_(0, flat_groups, exponent)
    weights = (exponent / denominator[flat_groups].clamp_min(1.0e-12)).reshape_as(scores)

    message = v[b, source] + base.relation_value(relation)[:, None].reshape(
        -1, 1, base.heads, base.head_dim
    )
    edge_output = (weights[..., None] * message.permute(0, 2, 1, 3)).sum(dim=2)
    output = torch.zeros(
        group_count, base.head_dim, device=temporal.device, dtype=temporal.dtype
    )
    output.scatter_add_(
        0,
        groups.reshape(-1, 1).expand(-1, base.head_dim),
        edge_output.reshape(-1, base.head_dim),
    )
    output = output.reshape(batch, agents, base.heads, base.head_dim)

    # The legacy dense path gives padded targets a uniform distribution over
    # every masked source-time slot.  Reproduce that otherwise-observable
    # boundary behavior with virtual fallback pairs only for empty targets.
    empty = ~allowed.any(dim=-1)
    if empty.any():
        empty_bt = empty.nonzero(as_tuple=False)
        source_ids = torch.arange(agents, device=temporal.device)
        virtual = torch.stack(
            (
                empty_bt[:, 0, None].expand(-1, agents),
                empty_bt[:, 1, None].expand(-1, agents),
                source_ids[None].expand(empty_bt.shape[0], -1),
            ),
            dim=-1,
        ).reshape(-1, 3)
        virtual_features = edge_relation_features(base, base._past, virtual)
        virtual_relation, _ = edge_relation_tokens(base, virtual_features)
        relation_fallback = base.relation_value(virtual_relation).reshape(
            empty_bt.shape[0], agents, base.heads, base.head_dim
        ).mean(dim=1)
        value_fallback = v.mean(dim=(1, 2))[empty_bt[:, 0]]
        output[empty_bt[:, 0], empty_bt[:, 1]] = value_fallback + relation_fallback
        base.last_virtual_fallback_pairs = int(virtual.shape[0])
    else:
        base.last_virtual_fallback_pairs = 0
    return base.out_proj(output.reshape(batch, agents, base.dim))


def _edge_first_forward(self: nn.Module, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    batch, _, dim = src.shape
    tokens = src.reshape(batch, self.obs_len, self.agents, dim).permute(0, 2, 1, 3).contiguous()
    temporal = self.temporal_encoder(
        tokens.reshape(batch * self.agents, self.obs_len, dim)
    ).reshape(batch, self.agents, self.obs_len, dim)
    temporal, agent = self.temporal_queries(temporal)
    self.last_temporal_features = temporal
    if self._past is None:
        raise RuntimeError("F16 context was not set")
    if self._num_valid is None:
        valid = torch.ones(batch, self.agents, dtype=torch.bool, device=src.device)
    else:
        ids = torch.arange(self.agents, device=src.device)
        valid = ids[None] < self._num_valid.long()[:, None]
    allowed = self.allowed_from_mask(mask, batch, valid)
    edges = allowed.nonzero(as_tuple=False)
    features = edge_relation_features(self, self._past, edges)
    relation, encoded = edge_relation_tokens(self, features)
    social = _edge_attention(self, temporal, agent, edges, relation, allowed)
    gate = torch.sigmoid(self.social_gate(torch.cat((agent, social), dim=-1)))
    fused_agent = self.social_norm(agent + gate * social)
    self.last_allowed = allowed.detach()
    self.last_edge_index = edges.detach()
    self.last_edge_relation_features = features.detach()
    self.last_edge_relation_encoded = encoded.detach()
    self.last_edge_relation_tokens = relation.detach()
    self.last_agent_features = fused_agent
    enriched = self.output_norm(
        temporal + self.token_feedback(fused_agent)[:, :, None, :]
    )
    return enriched.permute(0, 2, 1, 3).reshape(
        batch, self.obs_len * self.agents, dim
    )


def enable_edge_first(model: nn.Module) -> nn.Module:
    """Attach real sparse execution to a plain locked F16 model."""
    if getattr(model, "front_variant", None) != "f16_temporal_social_cross":
        raise ValueError("edge-first execution requires f16_temporal_social_cross")
    base = model.agent_aware_trans_encoder
    required = (
        base.use_relation_tokens
        and base.use_temporal_social_cross
        and base.use_multiscale_relation
        and not base.use_dual_graph
        and not base.use_dynamic_routing
        and not base.use_grouped_attention
    )
    if not required:
        raise ValueError("locked F16 encoder contract is not satisfied")
    base.forward = types.MethodType(_edge_first_forward, base)
    base.edge_first_execution = True
    return model
