"""Exact focal-agent self-only intervention for the F16 temporal-social layer."""

from __future__ import annotations

import types
from contextlib import contextmanager

import torch


def focal_self_only_allowed(allowed: torch.Tensor) -> torch.Tensor:
    """Keep tensor shape and all non-focal rows; replace focal receive row by self-loop."""
    masked = allowed.clone()
    if masked.ndim == 3:
        masked[:, 0, :] = False
        masked[:, 0, 0] = True
    elif masked.ndim == 4:
        masked[:, :, 0, :] = False
        masked[:, :, 0, 0] = True
    else:
        raise ValueError(f"Expected [B,A,A] or [B,H,A,A], got {tuple(masked.shape)}")
    return masked


@contextmanager
def focal_self_only_intervention(model):
    """Patch only the allowed edges consumed by focal temporal-social attention."""
    encoder = model.agent_aware_trans_encoder
    if not bool(getattr(encoder, "use_temporal_social_cross", False)):
        raise RuntimeError("CSNA requires the F16 temporal-social cross encoder")
    original = encoder.temporal_social_attention

    def wrapped(
        self, temporal, agent, relation, allowed, type_features=None, risk_score=None
    ):
        self.last_csna_original_allowed = allowed.detach().clone()
        masked = focal_self_only_allowed(allowed)
        self.last_csna_masked_allowed = masked.detach().clone()
        return original(temporal, agent, relation, masked, type_features, risk_score)

    encoder.temporal_social_attention = types.MethodType(wrapped, encoder)
    try:
        yield model
    finally:
        encoder.temporal_social_attention = original

