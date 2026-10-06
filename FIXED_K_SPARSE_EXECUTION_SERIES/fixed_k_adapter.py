"""GPU-regular fixed-width sparse execution for the locked formal F16.

The main path contains no ``nonzero`` or scatter reduction.  F16 keeps at most
four social neighbours plus the self edge, so the physical slot width is five.
"""
from __future__ import annotations

import math
import types
from typing import Optional

import torch
from torch import Tensor, nn


def allowed_to_slots(allowed: Tensor, social_k: int) -> tuple[Tensor, Tensor]:
    """Convert BxNxN adjacency to fixed-width neighbour slots."""
    agents = allowed.shape[-1]
    width = min(agents, int(social_k) + 1)  # bounded social slots + self
    values, index = torch.topk(allowed.to(torch.uint8), width, dim=-1)
    return index, values.bool()


def slots_to_allowed(index: Tensor, valid: Tensor, agents: int) -> Tensor:
    """Diagnostic reconstruction without scatter operations."""
    source = torch.arange(agents, device=index.device).view(1, 1, 1, agents)
    return ((index[..., None] == source) & valid[..., None]).any(dim=2)


def fixed_relation_features(base: nn.Module, past: Tensor, index: Tensor) -> Tensor:
    """Original F16 relation channels in BxNxSxTx8 layout."""
    batch, agents, steps, _ = past.shape
    velocity = torch.zeros_like(past)
    velocity[:, :, 1:] = (past[:, :, 1:] - past[:, :, :-1]) * base.frequency
    velocity[:, :, 0] = velocity[:, :, 1]
    batch_index = torch.arange(batch, device=past.device).view(batch, 1, 1)
    source_position = past[batch_index, index]
    source_velocity = velocity[batch_index, index]
    target_position = past[:, :, None]
    target_velocity = velocity[:, :, None]
    relative_position = source_position - target_position
    relative_velocity = source_velocity - target_velocity
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


def fixed_relation_tokens(base: nn.Module, features: Tensor) -> tuple[Tensor, Tensor]:
    encoded = base.relation_encoder(features)
    if not base.use_multiscale_relation:
        weight = torch.softmax(base.relation_time_score(encoded), dim=3)
        return (weight * encoded).sum(dim=3), encoded
    recent2 = encoded[..., -2:, :].mean(dim=3)
    recent4 = encoded[..., -4:, :].mean(dim=3)
    full = encoded.mean(dim=3)
    scales = torch.stack((recent2, recent4, full), dim=3)
    weight = torch.softmax(base.relation_scale_score(scales), dim=3)
    return (weight * scales).sum(dim=3), encoded


def _legacy_empty_target_fallback(
    base: nn.Module, temporal: Tensor, allowed: Tensor, value_all: Tensor
) -> Tensor:
    """Reproduce dense F16's observable uniform result for padded targets only."""
    empty = ~allowed.any(dim=-1)
    fallback = torch.zeros(
        temporal.shape[0], temporal.shape[1], base.heads, base.head_dim,
        device=temporal.device, dtype=temporal.dtype,
    )
    if not empty.any():
        base.last_fixedk_fallback_pairs = 0
        return fallback
    empty_bt = empty.nonzero(as_tuple=False)  # boundary-only, outside main path
    agents = temporal.shape[1]
    source = torch.arange(agents, device=temporal.device)
    virtual = torch.stack(
        (
            empty_bt[:, 0, None].expand(-1, agents),
            empty_bt[:, 1, None].expand(-1, agents),
            source[None].expand(empty_bt.shape[0], -1),
        ), dim=-1,
    ).reshape(-1, 3)
    # Import lazily to keep this adapter usable with the established EXP-197 code.
    from sparse_exec_adapter import edge_relation_features, edge_relation_tokens
    features = edge_relation_features(base, base._past, virtual)
    relation, _ = edge_relation_tokens(base, features)
    relation_value = base.relation_value(relation).reshape(
        empty_bt.shape[0], agents, base.heads, base.head_dim
    ).mean(dim=1)
    fallback[empty_bt[:, 0], empty_bt[:, 1]] = (
        value_all.mean(dim=(1, 2))[empty_bt[:, 0]] + relation_value
    )
    base.last_fixedk_fallback_pairs = int(virtual.shape[0])
    return fallback


def _fixed_attention(
    base: nn.Module,
    temporal: Tensor,
    agent: Tensor,
    relation: Tensor,
    index: Tensor,
    slot_valid: Tensor,
    allowed: Tensor,
) -> Tensor:
    batch, agents, steps, _ = temporal.shape
    slots = index.shape[-1]
    q = base.q_proj(agent).reshape(batch, agents, base.heads, base.qk_head_dim)
    k_all = base.k_proj(temporal).reshape(
        batch, agents, steps, base.heads, base.qk_head_dim
    )
    v_all = base.v_proj(temporal).reshape(
        batch, agents, steps, base.heads, base.head_dim
    )
    batch_index = torch.arange(batch, device=temporal.device).view(batch, 1, 1)
    k = k_all[batch_index, index].permute(0, 4, 1, 2, 3, 5)
    v = v_all[batch_index, index].permute(0, 4, 1, 2, 3, 5)
    q = q.permute(0, 2, 1, 3)
    scores = torch.einsum("bhnd,bhnstd->bhnst", q, k)
    scores = scores / math.sqrt(float(base.qk_head_dim))
    scores = scores + base.relation_bias(relation).permute(0, 3, 1, 2)[..., None]
    mask = slot_valid[:, None, :, :, None]
    weights = torch.softmax(
        scores.masked_fill(~mask, -1.0e4).flatten(-2), dim=-1
    ).reshape(batch, base.heads, agents, slots, steps)
    relation_value = base.relation_value(relation).reshape(
        batch, agents, slots, base.heads, base.head_dim
    ).permute(0, 3, 1, 2, 4)
    message = v + relation_value[..., None, :]
    output = (weights[..., None] * message).sum(dim=(3, 4))
    output = output.permute(0, 2, 1, 3)
    empty = ~slot_valid.any(dim=-1)
    if empty.any():
        fallback = _legacy_empty_target_fallback(base, temporal, allowed, v_all)
        output = torch.where(empty[..., None, None], fallback, output)
    else:
        base.last_fixedk_fallback_pairs = 0
    return base.out_proj(output.reshape(batch, agents, base.dim))


def _fixed_forward(self: nn.Module, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
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
        valid_agent = torch.ones(batch, self.agents, dtype=torch.bool, device=src.device)
    else:
        ids = torch.arange(self.agents, device=src.device)
        valid_agent = ids[None] < self._num_valid.long()[:, None]
    allowed = self.allowed_from_mask(mask, batch, valid_agent)
    index, slot_valid = allowed_to_slots(allowed, self.fixed_k_social_slots)
    reconstructed = slots_to_allowed(index, slot_valid, self.agents)
    if not torch.equal(allowed, reconstructed):
        raise RuntimeError("Fixed-K slots do not reconstruct the F16 graph")
    features = fixed_relation_features(self, self._past, index)
    relation, encoded = fixed_relation_tokens(self, features)
    social = _fixed_attention(
        self, temporal, agent, relation, index, slot_valid, allowed
    )
    gate = torch.sigmoid(self.social_gate(torch.cat((agent, social), dim=-1)))
    fused_agent = self.social_norm(agent + gate * social)
    self.last_allowed = allowed.detach()
    self.last_neighbor_index = index.detach()
    self.last_neighbor_valid = slot_valid.detach()
    self.last_fixed_relation_features = features.detach()
    self.last_fixed_relation_encoded = encoded.detach()
    self.last_fixed_relation_tokens = relation.detach()
    self.last_agent_features = fused_agent
    enriched = self.output_norm(
        temporal + self.token_feedback(fused_agent)[:, :, None, :]
    )
    return enriched.permute(0, 2, 1, 3).reshape(
        batch, self.obs_len * self.agents, dim
    )


def enable_fixed_k(model: nn.Module, social_slots: Optional[int] = None) -> nn.Module:
    if getattr(model, "front_variant", None) != "f16_temporal_social_cross":
        raise ValueError("Fixed-K execution requires f16_temporal_social_cross")
    base = model.agent_aware_trans_encoder
    compatible = (
        base.use_relation_tokens and base.use_temporal_social_cross
        and base.use_multiscale_relation and not base.use_dual_graph
        and not base.use_dynamic_routing and not base.use_grouped_attention
    )
    if not compatible:
        raise ValueError("locked formal F16 encoder contract is not satisfied")
    if social_slots is None:
        graph_cfg = getattr(getattr(model, "cfg", None), "MODEL", None)
        graph_cfg = getattr(graph_cfg, "GRAPH", None)
        mode = str(getattr(graph_cfg, "MODE", ""))
        social_slots = (
            int(getattr(graph_cfg, "K_MAX", base.top_k))
            if mode == "adaptive_kinematic" else int(base.top_k)
        )
    base.forward = types.MethodType(_fixed_forward, base)
    base.fixed_k_execution = True
    base.fixed_k_social_slots = int(social_slots)
    base.fixed_k_total_slots = min(int(base.agents), int(social_slots) + 1)
    return model
