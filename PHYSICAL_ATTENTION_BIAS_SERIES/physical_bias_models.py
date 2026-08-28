"""Time-resolved factorized physical attention bias for the locked F16 encoder.

This module is intentionally implemented as an external encoder plugin. It does
not modify FRONT_ENCODER_SERIES/front_encoder_models.py, so all previous
experiments remain reproducible.

The experiment keeps F16's temporal/social factorization and original sparse
Top-K mask. It changes only how physical pair information enters the
agent-by-neighbour-by-time attention logits.

Variants
--------
pb1_time_bias_add
    Keep the original learned pair-level relation bias and edge value, then add
    a zero-initialized time-resolved physical bias. At initialization it is
    numerically equivalent to F16.

pb2_time_bias_replace
    Replace the original pair-level relation bias with the time-resolved
    physical bias, while keeping the learned edge value message.

pb3_bias_only_interaction
    Remove the learned relation-token encoder/bias/value path and retain only
    content attention plus the low-parameter time-resolved physical bias.
"""

from __future__ import annotations

import math
import types
from typing import Optional

import torch
from torch import Tensor, nn


EXPERIMENTS = {
    "pb1_time_bias_add",
    "pb2_time_bias_replace",
    "pb3_bias_only_interaction",
}


class FactorizedPhysicalTimeBias(nn.Module):
    """Per-head RBF bias from distance, closing speed, TTC and DCA.

    The four physical channels are deliberately kept separate. Each channel is
    expanded by five fixed radial basis functions and each attention head learns
    only 4 x 5 coefficients. With four heads this adds just 80 parameters.

    All coefficients are initialized to zero, so pb1_time_bias_add starts
    exactly from the original F16 attention.
    """

    CHANNELS = ("distance", "closing", "ttc", "dca")

    def __init__(
        self,
        heads: int,
        basis_count: int = 5,
        max_abs_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.basis_count = int(basis_count)
        self.max_abs_bias = float(max_abs_bias)

        self.register_buffer(
            "centers_unsigned", torch.linspace(0.0, 1.0, basis_count)
        )
        self.register_buffer(
            "centers_signed", torch.linspace(-1.0, 1.0, basis_count)
        )
        self.unsigned_sigma = 0.25
        self.signed_sigma = 0.50

        self.weights = nn.Parameter(
            torch.zeros(self.heads, len(self.CHANNELS), self.basis_count)
        )

        self.last_mean_abs_bias: Optional[Tensor] = None
        self.last_max_abs_bias: Optional[Tensor] = None

    @staticmethod
    def _unsigned_scale(value: Tensor, reference: float = 6.0) -> Tensor:
        """Map non-negative metre-like quantities smoothly near [0, 1]."""
        denominator = math.log1p(reference)
        return (torch.log1p(value.clamp_min(0.0)) / denominator).clamp(0.0, 1.5)

    def _rbf_unsigned(self, value: Tensor) -> Tensor:
        return torch.exp(
            -0.5
            * (
                (value[..., None] - self.centers_unsigned)
                / self.unsigned_sigma
            ).square()
        )

    def _rbf_signed(self, value: Tensor) -> Tensor:
        return torch.exp(
            -0.5
            * (
                (value[..., None] - self.centers_signed)
                / self.signed_sigma
            ).square()
        )

    def forward(self, raw_relation_features: Tensor) -> Tensor:
        """Return bias [B,H,N,N,T] from raw F16 relation features.

        raw_relation_features is [B,N,N,T,8] with the existing F16 order:
        dx, dy, dvx, dvy, distance, closing, TTC/10, DCA.
        """
        if raw_relation_features.ndim != 5 or raw_relation_features.shape[-1] != 8:
            raise ValueError(
                "Expected raw relation features [B,N,N,T,8], got "
                f"{tuple(raw_relation_features.shape)}"
            )

        distance = raw_relation_features[..., 4]
        closing = raw_relation_features[..., 5]
        ttc = raw_relation_features[..., 6].clamp(0.0, 1.0)
        dca = raw_relation_features[..., 7]

        distance_n = self._unsigned_scale(distance)
        closing_n = torch.tanh(closing / 1.5)

        approach = torch.sigmoid(closing / 0.20)
        ttc_effective = approach * ttc + (1.0 - approach)
        dca_n = self._unsigned_scale(dca)
        dca_effective = approach * dca_n + (1.0 - approach)

        basis = torch.stack(
            (
                self._rbf_unsigned(distance_n),
                self._rbf_signed(closing_n),
                self._rbf_unsigned(ttc_effective),
                self._rbf_unsigned(dca_effective),
            ),
            dim=-2,
        )
        raw_bias = torch.einsum("bijtcm,hcm->bhijt", basis, self.weights)
        bias = self.max_abs_bias * torch.tanh(
            raw_bias / max(self.max_abs_bias, 1.0e-6)
        )

        agents = bias.shape[2]
        eye = torch.eye(
            agents, dtype=torch.bool, device=bias.device
        ).view(1, 1, agents, agents, 1)
        bias = bias.masked_fill(eye, 0.0)

        self.last_mean_abs_bias = bias.detach().abs().mean()
        self.last_max_abs_bias = bias.detach().abs().amax()
        return bias


def _relation_tokens_with_physical_tap(
    self,
    past: Tensor,
) -> tuple[Optional[Tensor], Tensor, Tensor, Tensor]:
    """Baseline-compatible relation path that also preserves raw time features."""
    features, last_distance, last_closing = self.relation_features(past)
    self.last_raw_relation_features = features

    if self.physical_bias_skip_relation_encoder:
        tokens = None
    else:
        encoded = self.relation_encoder(features)
        if self.use_multiscale_relation:
            recent2 = encoded[:, :, :, -2:].mean(dim=3)
            recent4 = encoded[:, :, :, -4:].mean(dim=3)
            full = encoded.mean(dim=3)
            scales = torch.stack((recent2, recent4, full), dim=3)
            scale_weights = torch.softmax(
                self.relation_scale_score(scales), dim=3
            )
            tokens = (scale_weights * scales).sum(dim=3)
        else:
            weights = torch.softmax(
                self.relation_time_score(encoded), dim=3
            )
            tokens = (weights * encoded).sum(dim=3)

    final_ttc = features[:, :, :, -1, 6] * 10.0
    final_dca = features[:, :, :, -1, 7]
    return tokens, last_distance, last_closing, final_ttc + final_dca


def _temporal_social_attention_with_physical_bias(
    self,
    temporal: Tensor,
    agent: Tensor,
    relation: Optional[Tensor],
    allowed: Tensor,
    type_features: Optional[Tensor] = None,
    risk_score: Optional[Tensor] = None,
) -> Tensor:
    """Specialized F16 attention with a time-resolved physical logit prior."""
    del type_features, risk_score
    batch, agents, steps, _ = temporal.shape

    if self.last_raw_relation_features is None:
        raise RuntimeError(
            "Physical bias requires relation_tokens() to run before attention."
        )
    raw_features = self.last_raw_relation_features
    if raw_features.shape[3] != steps:
        raise RuntimeError(
            "Physical feature time length does not match F16 temporal tokens: "
            f"{raw_features.shape[3]} vs {steps}"
        )

    q = self.q_proj(agent).reshape(
        batch, agents, self.heads, self.qk_head_dim
    ).permute(0, 2, 1, 3)
    k = self.k_proj(temporal).reshape(
        batch, agents, steps, self.kv_heads, self.qk_head_dim
    ).permute(0, 3, 1, 2, 4)
    v = self.v_proj(temporal).reshape(
        batch, agents, steps, self.kv_heads, self.head_dim
    ).permute(0, 3, 1, 2, 4)

    if self.kv_heads != self.heads:
        repeats = self.heads // self.kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    scores = torch.einsum("bhid,bhjtd->bhijt", q, k)
    scores = scores / math.sqrt(float(self.qk_head_dim))

    if self.physical_bias_keep_pair_bias:
        if relation is None:
            raise RuntimeError("Pair-level relation bias requested without relation tokens.")
        pair_bias = self.relation_bias(relation).permute(0, 3, 1, 2)
        scores = scores + pair_bias[..., None]

    physical_bias = self.physical_time_bias(raw_features)
    scores = scores + physical_bias

    if allowed.ndim == 3:
        expanded_mask = allowed[:, None, :, :, None].expand_as(scores)
    elif allowed.ndim == 4:
        expanded_mask = allowed[:, :, :, :, None].expand_as(scores)
    else:
        raise ValueError(f"Unexpected sparse mask shape: {tuple(allowed.shape)}")
    scores = scores.masked_fill(~expanded_mask, -1.0e4)

    weights = torch.softmax(scores.flatten(-2), dim=-1).reshape_as(scores)
    base_message = v[:, :, None].expand(
        -1, -1, agents, -1, -1, -1
    )

    if self.physical_bias_keep_edge_value:
        if relation is None:
            raise RuntimeError("Edge value requested without relation tokens.")
        edge_value = self.relation_value(relation).reshape(
            batch, agents, agents, self.heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)
        message = base_message + edge_value[..., None, :]
    else:
        message = base_message

    social = (weights[..., None] * message).sum(dim=(3, 4))
    social = social.permute(0, 2, 1, 3).reshape(
        batch, agents, self.dim
    )
    return self.out_proj(social)


def _validate_locked_f16(base: nn.Module) -> None:
    """Fail loudly if a future F16 change makes this controlled patch invalid."""
    required_false = (
        "use_pair_deformable_temporal",
        "use_pair_temporal_soft_prior",
        "use_scaled_cosine_attention",
        "use_differential_attention",
        "use_relation_type_bias",
        "use_soft_risk_bias",
        "use_temporal_relative_bias",
        "use_sparsemax",
        "use_entmax15",
        "use_relation_grouped_vector_attention",
        "use_temporal_half_pool",
        "use_fixed_half_projection",
        "use_recent_exact_early_pool",
        "use_motion_event_sampling",
        "use_motion_token_squeeze",
        "use_edge_list_attention",
    )
    active = [name for name in required_false if bool(getattr(base, name, False))]
    if active:
        raise ValueError(
            "Physical-bias series expects the locked plain F16 attention; "
            f"unexpected active flags: {active}"
        )
    if not bool(getattr(base, "use_temporal_social_cross", False)):
        raise ValueError("Physical-bias series requires F16 temporal-social cross attention.")
    if not bool(getattr(base, "use_relation_tokens", False)):
        raise ValueError("Physical-bias series expects the F16 relation-token baseline.")


def replace_encoder(model: nn.Module, cfg, experiment: str) -> None:
    """Install one controlled physical-bias variant into the locked F16 model."""
    del cfg
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown physical-bias experiment: {experiment}")
    if getattr(model, "front_variant", None) != "f16_temporal_social_cross":
        raise ValueError("Physical-bias experiments require f16_temporal_social_cross.")

    base = model.agent_aware_trans_encoder
    _validate_locked_f16(base)

    keep_pair_bias = experiment == "pb1_time_bias_add"
    keep_edge_value = experiment in {
        "pb1_time_bias_add",
        "pb2_time_bias_replace",
    }
    skip_relation_encoder = experiment == "pb3_bias_only_interaction"

    base.physical_time_bias = FactorizedPhysicalTimeBias(
        heads=int(base.heads),
        basis_count=5,
        max_abs_bias=2.0,
    )
    base.physical_bias_keep_pair_bias = keep_pair_bias
    base.physical_bias_keep_edge_value = keep_edge_value
    base.physical_bias_skip_relation_encoder = skip_relation_encoder
    base.last_raw_relation_features = None

    if experiment == "pb2_time_bias_replace":
        base.relation_bias = nn.Identity()

    if experiment == "pb3_bias_only_interaction":
        base.relation_encoder = nn.Identity()
        base.relation_time_score = nn.Identity()
        base.relation_scale_score = nn.Identity()
        base.relation_bias = nn.Identity()
        base.relation_value = nn.Identity()

    base.relation_tokens = types.MethodType(
        _relation_tokens_with_physical_tap, base
    )
    base.temporal_social_attention = types.MethodType(
        _temporal_social_attention_with_physical_bias, base
    )
