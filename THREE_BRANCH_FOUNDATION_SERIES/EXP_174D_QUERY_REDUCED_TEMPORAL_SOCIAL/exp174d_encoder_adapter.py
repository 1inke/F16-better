from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


EXPERIMENT = "three_representative_temporal_social_kv"


class ThreeRepresentativeMixin:
    """F16 with three deterministic temporal K/V representatives."""

    def temporal_social_attention(
        self,
        temporal: Tensor,
        agent: Tensor,
        relation: Tensor,
        allowed: Tensor,
        type_features: Optional[Tensor] = None,
        risk_score: Optional[Tensor] = None,
    ) -> Tensor:
        steps = temporal.shape[2]
        if steps < 4:
            raise ValueError(f"EXP-174D expects at least four history states, got {steps}")
        midpoint = steps // 2
        long_summary = temporal[:, :, :midpoint].mean(dim=2, keepdim=True)
        short_summary = temporal[:, :, midpoint:-1].mean(dim=2, keepdim=True)
        latest = temporal[:, :, -1:]
        reduced = torch.cat((long_summary, short_summary, latest), dim=2)
        self.last_social_kv_steps = int(reduced.shape[2])
        return super().temporal_social_attention(
            reduced, agent, relation, allowed, type_features, risk_score
        )


def replace_encoder(model, cfg, name: str) -> None:
    if name != EXPERIMENT:
        raise ValueError(f"Unknown EXP-174D encoder experiment: {name}")
    encoder = model.agent_aware_trans_encoder
    if not hasattr(encoder, "temporal_social_attention") or not hasattr(encoder, "use_temporal_social_cross"):
        raise TypeError(f"Expected F16-compatible factorized encoder, got {type(encoder).__name__}")
    if not encoder.use_temporal_social_cross:
        raise RuntimeError("EXP-174D must start from F16 temporal-social cross")
    if any((encoder.use_temporal_half_pool, encoder.use_fixed_half_projection,
            encoder.use_recent_exact_early_pool, encoder.use_motion_event_sampling,
            encoder.use_motion_token_squeeze, bool(encoder.temporal_landmarks))):
        raise RuntimeError("EXP-174D cannot stack another temporal K/V reduction")
    before = set(encoder.state_dict())
    adapted_class = type(
        "ThreeRepresentativeF16Encoder",
        (ThreeRepresentativeMixin, encoder.__class__),
        {"__module__": __name__},
    )
    encoder.__class__ = adapted_class
    after = set(encoder.state_dict())
    if before != after:
        raise RuntimeError("EXP-174D unexpectedly changed the encoder state-dict contract")
    encoder.exp174d_description = "one agent Q; long/short/latest temporal K/V"
