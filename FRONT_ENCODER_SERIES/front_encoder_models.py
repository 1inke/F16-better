"""Front-encoder variants for the locked ASTRA Q20 decoder.

The original decoder, ModeQueryLoss and graph/RBF modules are not changed.
Every encoder returns the original time-major shape [B, T*N, D], so the
existing Q20 decoder can be reused without modification.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.astra_model import ASTRA_model


VARIANTS = {
    "f0_baseline",
    "f1_robust_state",
    "f2_temporal_queries",
    "f3_factorized",
    "f4_relation_tokens",
    "f5_dual_graph",
    "f6_light_factorized",
    "f7_ultralight_factorized",
    "f8_heading_aligned_relation",
    "f9_soft_risk_bias",
    "f10_multiscale_relation",
    "f11_motion_auxiliary",
    "f12_focal_conditioned",
    "f13_dynamic_routing",
    "f14_grouped_attention",
    "f15_edge_iteration",
    "f16_temporal_social_cross",
    "f17_sparsemax_attention",
    "f18_relation_qk",
    "f19_entmax15_attention",
    "f20_node_structure_encoding",
    "f21_agent_proxy_attention",
    "f22_temporal_cross_entmax15",
    "f23_dynamic_temporal_cross",
    "f24_temporal_dropkey",
    "f25_temporal_relative_bias",
    "f26_adaptive_dual_social",
    "f27_density_adaptive_social",
    "f28_temporal_landmark4",
    "f29_temporal_landmark2",
    "f30_graph_additive_attention",
    "f31_cascaded_group_attention",
    "f32_grouped_query_attention",
    "f33_multi_query_attention",
    "f34_identity_value_attention",
    "f35_temporal_half_pool",
    "f36_fixed_half_projection",
    "f37_lowdim_qk_attention",
    "f38_recent_exact_early_pool",
    "f39_motion_event_sampling",
    "f40_motion_token_squeeze",
    "f41_edge_list_attention",
    "f42_dual_scale_sparse",
    "f43_relation_type_bias",
    "f44_second_order_context",
    "f45_temporal_risk_bias",
    "f46_multiscale_temporal_residual",
    "f47_multiscale_temporal_replace",
    "f48_ttc_dca_graph",
    "f49_ttc_dca_multiscale_replace",
    "f50_enhanced_kinematic_state",
    "f51_scaled_cosine_attention",
    "f52_pair_deformable_temporal",
    "f53_kdta_recent_exact",
    "f54_kdta_trend_event_recent",
    "f55_pair_temporal_soft_prior",
    "f56_motion_adaptive_key_smoothing",
    "f57_relation_grouped_vector_attention",
    "f58_compute_neutral_differential",
}


class RobustMultiScaleState(nn.Module):
    """Encode short/long motion, smooth acceleration and turning confidence."""

    feature_dim = 11

    def __init__(self, token_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, -0.5)

    @staticmethod
    def features(past: Tensor, frequency: float) -> Tensor:
        # past: [B,N,T,2]
        short_velocity = torch.zeros_like(past)
        short_velocity[:, :, 1:] = (
            past[:, :, 1:] - past[:, :, :-1]
        ) * frequency
        short_velocity[:, :, 0] = short_velocity[:, :, 1]

        lag = min(3, past.shape[2] - 1)
        long_velocity = torch.zeros_like(past)
        long_velocity[:, :, lag:] = (
            past[:, :, lag:] - past[:, :, :-lag]
        ) * (frequency / float(lag))
        long_velocity[:, :, :lag] = long_velocity[:, :, lag : lag + 1]

        smooth_acceleration = (
            short_velocity - long_velocity
        ) * frequency
        speed = torch.linalg.vector_norm(
            short_velocity, dim=-1, keepdim=True
        )
        heading = short_velocity / speed.clamp_min(1.0e-5)
        confidence = torch.sigmoid((speed - 0.05) / 0.05)
        heading = heading * confidence

        previous_heading = torch.cat(
            (heading[:, :, :1], heading[:, :, :-1]), dim=2
        )
        cross = (
            previous_heading[..., 0] * heading[..., 1]
            - previous_heading[..., 1] * heading[..., 0]
        )
        dot = (previous_heading * heading).sum(dim=-1)
        turn_rate = torch.atan2(cross, dot.clamp(-1.0, 1.0)).unsqueeze(-1)
        turn_rate = turn_rate * confidence

        return torch.cat(
            (
                short_velocity / 1.5,
                long_velocity / 1.5,
                smooth_acceleration / 2.0,
                heading,
                speed / 1.5,
                turn_rate / math.pi,
                confidence,
            ),
            dim=-1,
        )

    def forward(self, past: Tensor, frequency: float) -> Tensor:
        features = self.features(past, frequency)
        encoded = torch.tanh(self.encoder(features))
        gate = torch.sigmoid(self.gate(features))
        return encoded * gate


class EnhancedKinematicState(nn.Module):
    """Motion-state embedding with jerk, speed trend and channel-wise confidence."""

    feature_dim = 16

    def __init__(self, token_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.channel_gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, token_dim),
        )
        nn.init.constant_(self.channel_gate[-1].bias, -0.5)

    @staticmethod
    def features(past: Tensor, frequency: float) -> Tensor:
        base = RobustMultiScaleState.features(past, frequency)
        velocity = torch.zeros_like(past)
        velocity[:, :, 1:] = (past[:, :, 1:] - past[:, :, :-1]) * frequency
        velocity[:, :, 0] = velocity[:, :, 1]
        acceleration = torch.zeros_like(velocity)
        acceleration[:, :, 1:] = (velocity[:, :, 1:] - velocity[:, :, :-1]) * frequency
        acceleration[:, :, 0] = acceleration[:, :, 1]
        jerk = torch.zeros_like(acceleration)
        jerk[:, :, 1:] = (acceleration[:, :, 1:] - acceleration[:, :, :-1]) * frequency
        jerk[:, :, 0] = jerk[:, :, 1]
        speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
        acceleration_size = torch.linalg.vector_norm(acceleration, dim=-1, keepdim=True)
        lag = min(3, speed.shape[2] - 1)
        long_speed = torch.zeros_like(speed)
        long_speed[:, :, lag:] = speed[:, :, :-lag]
        long_speed[:, :, :lag] = speed[:, :, lag : lag + 1]
        speed_trend = (speed - long_speed) / 1.5
        stationary = torch.exp(-speed / 0.08)
        return torch.cat(
            (base, jerk / 4.0, acceleration_size / 2.0, speed_trend, stationary),
            dim=-1,
        )

    def forward(self, past: Tensor, frequency: float) -> Tensor:
        features = self.features(past, frequency)
        encoded = torch.tanh(self.encoder(features))
        confidence = torch.sigmoid(self.channel_gate(features))
        return encoded * confidence


class DynamicTemporalQueries(nn.Module):
    """Recent/trend/turn queries with gated feedback to history tokens."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.recent_query = nn.Linear(dim, dim)
        self.trend_query = nn.Linear(dim, dim)
        self.turn_query = nn.Linear(dim, dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.query_gate = nn.Linear(dim, 1)
        self.feedback = nn.Linear(dim, dim)
        self.feedback_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        nn.init.constant_(self.feedback_gate[-1].bias, -2.0)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        # tokens: [B,N,T,D]
        batch, agents, steps, dim = tokens.shape
        flat = tokens.reshape(batch * agents, steps, dim)
        recent = self.recent_query(flat[:, -1])
        trend = self.trend_query(flat.mean(dim=1))
        turn_source = flat[:, -1] - flat[:, -2]
        turn = self.turn_query(turn_source)
        queries = torch.stack((recent, trend, turn), dim=1)
        summaries, _ = self.cross_attention(queries, flat, flat)
        weights = torch.softmax(self.query_gate(summaries), dim=1)
        agent = (weights * summaries).sum(dim=1)
        expanded = agent[:, None].expand(-1, steps, -1)
        feedback_gate = torch.sigmoid(
            self.feedback_gate(torch.cat((flat, expanded), dim=-1))
        )
        enriched = self.norm(
            flat + feedback_gate * self.feedback(expanded)
        )
        return (
            enriched.reshape(batch, agents, steps, dim),
            agent.reshape(batch, agents, dim),
        )


class MultiScaleTemporalMixer(nn.Module):
    """Low-rank multi-dilation motion mixer with a kinematics-conditioned gate."""

    def __init__(self, dim: int, rank: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.branches = nn.ModuleList(
            nn.Conv1d(rank, rank, kernel_size=3, padding=dilation,
                      dilation=dilation, groups=rank)
            for dilation in (1, 2, 3)
        )
        self.motion_gate = nn.Sequential(
            nn.LayerNorm(5), nn.Linear(5, 12), nn.GELU(), nn.Linear(12, 3)
        )
        self.up = nn.Linear(rank, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.residual_gate = nn.Parameter(torch.tensor(-1.5))

    @staticmethod
    def motion_features(past: Tensor, frequency: float) -> Tensor:
        velocity = (past[:, :, 1:] - past[:, :, :-1]) * frequency
        latest = velocity[:, :, -1]
        previous = velocity[:, :, -2] if velocity.shape[2] > 1 else latest
        mean = velocity.mean(dim=2)
        acceleration = (latest - previous) * frequency
        latest_speed = torch.linalg.vector_norm(latest, dim=-1)
        mean_speed = torch.linalg.vector_norm(mean, dim=-1)
        acceleration_size = torch.linalg.vector_norm(acceleration, dim=-1)
        latest_heading = latest / latest_speed[..., None].clamp_min(1.0e-4)
        previous_speed = torch.linalg.vector_norm(previous, dim=-1)
        previous_heading = previous / previous_speed[..., None].clamp_min(1.0e-4)
        heading_cos = (latest_heading * previous_heading).sum(dim=-1).clamp(-1.0, 1.0)
        heading_cross = (
            previous_heading[..., 0] * latest_heading[..., 1]
            - previous_heading[..., 1] * latest_heading[..., 0]
        )
        moving = ((latest_speed > 0.03) & (previous_speed > 0.03)).to(past.dtype)
        return torch.stack(
            (latest_speed / 1.5, mean_speed / 1.5, acceleration_size / 2.0,
             heading_cos * moving, heading_cross * moving), dim=-1
        )

    def forward(self, tokens: Tensor, past: Tensor, frequency: float) -> Tensor:
        batch, agents, steps, _ = tokens.shape
        latent = torch.nn.functional.gelu(self.down(self.norm(tokens)))
        flat = latent.reshape(batch * agents, steps, -1).transpose(1, 2)
        branches = [torch.nn.functional.gelu(layer(flat)).transpose(1, 2)
                    for layer in self.branches]
        stacked = torch.stack(branches, dim=2).reshape(batch, agents, steps, 3, -1)
        weights = torch.softmax(
            self.motion_gate(self.motion_features(past, frequency)), dim=-1
        )[:, :, None, :, None]
        mixed = (weights * stacked).sum(dim=3)
        update = self.up(mixed)
        return self.output_norm(tokens + torch.sigmoid(self.residual_gate) * update)


class LegacyPlusTemporalQueries(nn.Module):
    """F2: preserve the legacy mixed encoder and append query feedback."""

    def __init__(
        self,
        legacy: nn.Module,
        obs_len: int,
        agents: int,
        dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.legacy = legacy
        self.obs_len = obs_len
        self.agents = agents
        self.temporal_queries = DynamicTemporalQueries(dim, heads, dropout)

    def set_context(self, past: Tensor, num_valid: Optional[Tensor]) -> None:
        del past, num_valid

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        encoded = self.legacy(src, mask=mask)
        batch, _, dim = encoded.shape
        tokens = (
            encoded.reshape(batch, self.obs_len, self.agents, dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        enriched, _ = self.temporal_queries(tokens)
        return (
            enriched.permute(0, 2, 1, 3)
            .reshape(batch, self.obs_len * self.agents, dim)
        )


class FactorizedStateRelationEncoder(nn.Module):
    """F3-F5 temporal/social factorization with optional relation tokens."""

    def __init__(
        self,
        obs_len: int,
        agents: int,
        dim: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        use_relation_tokens: bool,
        use_dual_graph: bool,
        top_k: int = 4,
        frequency: float = 2.5,
        heading_aligned_relations: bool = False,
        use_soft_risk_bias: bool = False,
        use_multiscale_relation: bool = False,
        use_focal_conditioning: bool = False,
        use_dynamic_routing: bool = False,
        use_grouped_attention: bool = False,
        use_edge_iteration: bool = False,
        use_temporal_social_cross: bool = False,
        use_sparsemax: bool = False,
        use_relation_qk: bool = False,
        use_entmax15: bool = False,
        use_node_structure: bool = False,
        use_agent_proxies: bool = False,
        use_temporal_dropkey: bool = False,
        use_temporal_relative_bias: bool = False,
        use_adaptive_dual_social: bool = False,
        use_density_adaptive_social: bool = False,
        temporal_landmarks: int = 0,
        use_graph_additive_attention: bool = False,
        use_cascaded_group_attention: bool = False,
        kv_heads: int = 0,
        use_identity_value: bool = False,
        use_temporal_half_pool: bool = False,
        use_fixed_half_projection: bool = False,
        qk_head_dim: int = 0,
        use_recent_exact_early_pool: bool = False,
        use_motion_event_sampling: bool = False,
        use_motion_token_squeeze: bool = False,
        use_edge_list_attention: bool = False,
        use_dual_scale_sparse: bool = False,
        use_relation_type_bias: bool = False,
        use_second_order_context: bool = False,
        use_multiscale_temporal: bool = False,
        replace_temporal_encoder: bool = False,
        use_ttc_dca_graph: bool = False,
        use_scaled_cosine_attention: bool = False,
        use_pair_deformable_temporal: bool = False,
        kdta_recent_exact: bool = False,
        kdta_include_trend: bool = False,
        use_pair_temporal_soft_prior: bool = False,
        use_motion_adaptive_key_smoothing: bool = False,
        use_relation_grouped_vector_attention: bool = False,
        use_differential_attention: bool = False,
    ) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.agents = agents
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.use_relation_tokens = use_relation_tokens
        self.use_dual_graph = use_dual_graph
        self.top_k = top_k
        self.frequency = float(frequency)
        self.heading_aligned_relations = bool(heading_aligned_relations)
        self.use_soft_risk_bias = bool(use_soft_risk_bias)
        self.use_multiscale_relation = bool(use_multiscale_relation)
        self.use_focal_conditioning = bool(use_focal_conditioning)
        self.use_dynamic_routing = bool(use_dynamic_routing)
        self.use_grouped_attention = bool(use_grouped_attention)
        self.use_edge_iteration = bool(use_edge_iteration)
        self.use_temporal_social_cross = bool(use_temporal_social_cross)
        self.use_sparsemax = bool(use_sparsemax)
        self.use_relation_qk = bool(use_relation_qk)
        self.use_entmax15 = bool(use_entmax15)
        self.use_node_structure = bool(use_node_structure)
        self.use_agent_proxies = bool(use_agent_proxies)
        self.use_temporal_dropkey = bool(use_temporal_dropkey)
        self.use_temporal_relative_bias = bool(use_temporal_relative_bias)
        self.use_adaptive_dual_social = bool(use_adaptive_dual_social)
        self.use_density_adaptive_social = bool(use_density_adaptive_social)
        self.temporal_landmarks = int(temporal_landmarks)
        self.use_graph_additive_attention = bool(use_graph_additive_attention)
        self.use_cascaded_group_attention = bool(use_cascaded_group_attention)
        self.kv_heads = int(kv_heads) if kv_heads else heads
        self.use_identity_value = bool(use_identity_value)
        self.use_temporal_half_pool = bool(use_temporal_half_pool)
        self.use_fixed_half_projection = bool(use_fixed_half_projection)
        self.qk_head_dim = int(qk_head_dim) if qk_head_dim else self.head_dim
        self.use_recent_exact_early_pool = bool(use_recent_exact_early_pool)
        self.use_motion_event_sampling = bool(use_motion_event_sampling)
        self.use_motion_token_squeeze = bool(use_motion_token_squeeze)
        self.use_edge_list_attention = bool(use_edge_list_attention)
        self.use_dual_scale_sparse = bool(use_dual_scale_sparse)
        self.use_relation_type_bias = bool(use_relation_type_bias)
        self.use_second_order_context = bool(use_second_order_context)
        self.use_multiscale_temporal = bool(use_multiscale_temporal)
        self.replace_temporal_encoder = bool(replace_temporal_encoder)
        self.use_ttc_dca_graph = bool(use_ttc_dca_graph)
        self.use_scaled_cosine_attention = bool(use_scaled_cosine_attention)
        self.use_pair_deformable_temporal = bool(use_pair_deformable_temporal)
        self.kdta_recent_exact = bool(kdta_recent_exact)
        self.kdta_include_trend = bool(kdta_include_trend)
        self.use_pair_temporal_soft_prior = bool(use_pair_temporal_soft_prior)
        self.use_motion_adaptive_key_smoothing = bool(
            use_motion_adaptive_key_smoothing
        )
        self.use_relation_grouped_vector_attention = bool(
            use_relation_grouped_vector_attention
        )
        self.use_differential_attention = bool(use_differential_attention)
        self.risk_strength = nn.Parameter(torch.zeros(heads))
        if dim % heads:
            raise ValueError("Encoder dimension must be divisible by heads.")

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=1, norm=nn.LayerNorm(dim)
        )
        if self.replace_temporal_encoder:
            self.temporal_encoder = nn.Identity()
        self.multiscale_temporal = (
            MultiScaleTemporalMixer(dim, rank=max(16, dim // 3))
            if self.use_multiscale_temporal else None
        )
        self.temporal_queries = DynamicTemporalQueries(dim, heads, dropout)

        self.relation_encoder = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
        )
        self.relation_time_score = nn.Linear(dim, 1)
        self.relation_scale_score = nn.Linear(dim, 1)
        self.relation_bias = nn.Linear(dim, heads)
        self.relation_value = nn.Linear(dim, dim)

        if self.use_pair_deformable_temporal or self.use_pair_temporal_soft_prior:
            kdta_hidden = max(24, dim // 3)
            offset_count = (
                1 if self.kdta_recent_exact or self.use_pair_temporal_soft_prior
                else 2
            )
            self.kdta_offset = nn.Sequential(
                nn.LayerNorm(dim * 3),
                nn.Linear(dim * 3, kdta_hidden),
                nn.GELU(),
                nn.Linear(kdta_hidden, offset_count),
            )
            # Start exactly from two interpretable history locations.  The
            # model learns bounded pair-specific residuals; the final frame is
            # always appended unchanged.
            nn.init.zeros_(self.kdta_offset[-1].weight)
            nn.init.zeros_(self.kdta_offset[-1].bias)
            if self.use_pair_temporal_soft_prior:
                base_positions = [3.0]
                offset_ranges = [3.0]
            else:
                base_positions = (
                    [1.5] if self.kdta_recent_exact else [1.5, 4.5]
                )
                offset_ranges = (
                    [1.5] if self.kdta_recent_exact else [1.5, 1.5]
                )
            self.register_buffer(
                "kdta_base_positions",
                torch.tensor(base_positions),
                persistent=False,
            )
            self.register_buffer(
                "kdta_offset_ranges",
                torch.tensor(offset_ranges),
                persistent=False,
            )
            self.last_kdta_positions: Optional[Tensor] = None
        if self.use_pair_temporal_soft_prior:
            initial_strength = math.atanh(0.25)
            self.temporal_prior_strength = nn.Parameter(
                torch.full((heads,), initial_strength)
            )
        if self.use_motion_adaptive_key_smoothing:
            self.key_smoothing_logits = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, 2)
            )
            nn.init.zeros_(self.key_smoothing_logits[-1].weight)
            nn.init.zeros_(self.key_smoothing_logits[-1].bias)
            self.last_key_smoothing: Optional[Tensor] = None
        if self.use_relation_grouped_vector_attention:
            self.vector_groups = 4
            if self.head_dim % self.vector_groups:
                raise ValueError("head_dim must be divisible by four vector groups")
            self.relation_group_correction = nn.Linear(
                dim, heads * self.vector_groups
            )
            nn.init.zeros_(self.relation_group_correction.weight)
            nn.init.zeros_(self.relation_group_correction.bias)
            self.group_correction_strength = nn.Parameter(
                torch.full((heads, self.vector_groups), math.atanh(0.10))
            )
        if self.use_differential_attention:
            if self.qk_head_dim % 2:
                raise ValueError("differential attention requires an even Q/K head dim")
            initial_lambda = 0.05
            self.differential_lambda_logit = nn.Parameter(
                torch.full(
                    (heads,), math.log(initial_lambda / (1.0 - initial_lambda))
                )
            )
            self.last_differential_negative_mass: Optional[Tensor] = None

        if self.use_cascaded_group_attention:
            self.group_q_proj = nn.ModuleList(
                nn.Linear(self.head_dim, self.head_dim) for _ in range(heads)
            )
            self.group_k_proj = nn.ModuleList(
                nn.Linear(self.head_dim, self.head_dim) for _ in range(heads)
            )
            self.group_v_proj = nn.ModuleList(
                nn.Linear(self.head_dim, self.head_dim) for _ in range(heads)
            )
        elif self.use_graph_additive_attention:
            self.additive_time_score = nn.Linear(dim, 1)
            self.additive_source_score = nn.Linear(dim, heads)
            self.v_proj = nn.Linear(dim, dim)
        else:
            self.q_proj = nn.Linear(dim, heads * self.qk_head_dim)
            kv_qk_dim = self.kv_heads * self.qk_head_dim
            kv_value_dim = self.kv_heads * self.head_dim
            self.k_proj = nn.Linear(dim, kv_qk_dim)
            if not self.use_identity_value:
                self.v_proj = nn.Linear(dim, kv_value_dim)
        if self.use_scaled_cosine_attention:
            # A per-head log scale preserves the selectivity of dot-product
            # attention while removing feature-norm drift from Q/K similarity.
            # sqrt(d_h) approximately matches the initial score variance of
            # the original QK/sqrt(d_h) formulation.
            self.cosine_logit_scale = nn.Parameter(
                torch.full((heads,), 0.5 * math.log(float(self.qk_head_dim)))
            )
        self.out_proj = nn.Linear(dim, dim)

        if self.temporal_landmarks:
            if not 0 < self.temporal_landmarks <= obs_len:
                raise ValueError("temporal_landmarks must be in [1, obs_len]")
            landmark_logits = torch.full(
                (self.temporal_landmarks, obs_len), -4.0
            )
            boundaries = torch.linspace(
                0, obs_len, self.temporal_landmarks + 1
            ).round().long()
            for landmark in range(self.temporal_landmarks):
                start = int(boundaries[landmark])
                end = max(start + 1, int(boundaries[landmark + 1]))
                landmark_logits[landmark, start:end] = 0.0
            self.temporal_landmark_logits = nn.Parameter(landmark_logits)
        if self.use_fixed_half_projection:
            if obs_len % 2:
                raise ValueError("fixed half projection requires an even obs_len")
            projection = torch.zeros(2, obs_len)
            projection[0, : obs_len // 2] = 2.0 / obs_len
            projection[1, obs_len // 2 :] = 2.0 / obs_len
            self.register_buffer(
                "fixed_half_projection", projection, persistent=False
            )
        self.social_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        nn.init.constant_(self.social_gate[-1].bias, -1.0)
        self.social_norm = nn.LayerNorm(dim)
        self.token_feedback = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim)

        if self.use_relation_type_bias:
            # Explicit soft interaction types complement the continuous edge
            # token.  The zero initialization keeps the initial network exactly
            # on the F16 path and lets training introduce only useful biases.
            self.type_bias = nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, 16),
                nn.GELU(),
                nn.Linear(16, heads),
            )
            nn.init.zeros_(self.type_bias[-1].weight)
            nn.init.zeros_(self.type_bias[-1].bias)

        if self.use_second_order_context:
            # A source agent receives a compact summary of its own sparse
            # neighborhood before it is used as a key/value.  This gives the
            # focal agent two-hop information without a second N x N attention.
            self.second_order_proj = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, dim),
            )
            self.second_order_gate = nn.Parameter(torch.tensor(-2.0))
            self.second_order_norm = nn.LayerNorm(dim)

        if self.use_dynamic_routing:
            route_dim = max(16, dim // 6)
            self.route_query = nn.Linear(dim, route_dim)
            self.route_key = nn.Linear(dim, route_dim)
            self.route_relation = nn.Linear(dim, 1)

        if self.use_edge_iteration:
            edge_hidden = max(16, dim // 3)
            self.edge_refine = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, edge_hidden),
                nn.GELU(),
                nn.Linear(edge_hidden, dim),
            )
            self.edge_refine_strength = nn.Parameter(torch.tensor(-2.0))
            self.second_social_gate = nn.Sequential(
                nn.Linear(dim * 2, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
            )
            nn.init.constant_(self.second_social_gate[-1].bias, -2.0)

        if self.use_relation_qk:
            self.relation_qk = nn.Linear(dim, dim * 2)
            nn.init.zeros_(self.relation_qk.weight)
            nn.init.zeros_(self.relation_qk.bias)

        if self.use_node_structure:
            structure_hidden = max(16, dim // 3)
            self.node_structure_encoder = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, structure_hidden),
                nn.GELU(),
                nn.Linear(structure_hidden, dim),
            )
            self.node_structure_gate = nn.Parameter(torch.tensor(-2.0))
            self.node_structure_norm = nn.LayerNorm(dim)

        if self.use_agent_proxies:
            self.agent_proxies = nn.Parameter(torch.randn(2, dim) * 0.02)
            self.proxy_gate = nn.Sequential(
                nn.Linear(dim * 2, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
            )
            nn.init.constant_(self.proxy_gate[-1].bias, -2.0)
            self.proxy_norm = nn.LayerNorm(dim)

        if self.use_temporal_relative_bias:
            self.temporal_relative_bias = nn.Parameter(
                torch.zeros(heads, obs_len)
            )

        if self.use_adaptive_dual_social:
            self.temporal_mix_gate = nn.Linear(dim * 3, 1)
            nn.init.zeros_(self.temporal_mix_gate.weight)
            nn.init.constant_(self.temporal_mix_gate.bias, 1.0)

        if self.use_focal_conditioning:
            focal_dim = max(16, dim // 3)
            self.focal_query = nn.Linear(dim, focal_dim)
            self.focal_key = nn.Linear(dim, focal_dim)
            self.focal_value = nn.Linear(dim, focal_dim)
            self.focal_out = nn.Linear(focal_dim, dim)
            self.focal_gate = nn.Linear(dim * 2, 1)
            self.focal_norm = nn.LayerNorm(dim)
            nn.init.constant_(self.focal_gate.bias, -2.0)

        self._past: Optional[Tensor] = None
        self._num_valid: Optional[Tensor] = None

    def set_context(self, past: Tensor, num_valid: Optional[Tensor]) -> None:
        self._past = past
        self._num_valid = num_valid

    def relation_features(self, past: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # past [B,N,T,2] -> relation features [B,N,N,T,8]
        frequency = self.frequency
        position = past.permute(0, 2, 1, 3)
        velocity = torch.zeros_like(position)
        velocity[:, 1:] = (
            position[:, 1:] - position[:, :-1]
        ) * frequency
        velocity[:, 0] = velocity[:, 1]
        relative_position = (
            position[:, :, None, :, :] - position[:, :, :, None, :]
        )
        relative_velocity = (
            velocity[:, :, None, :, :] - velocity[:, :, :, None, :]
        )
        if self.heading_aligned_relations:
            receiver_velocity = velocity[:, :, :, None, :]
            receiver_speed = torch.linalg.vector_norm(
                receiver_velocity, dim=-1, keepdim=True
            )
            default_heading = torch.zeros_like(receiver_velocity)
            default_heading[..., 0] = 1.0
            heading = torch.where(
                receiver_speed > 0.02,
                receiver_velocity / receiver_speed.clamp_min(1.0e-6),
                default_heading,
            )
            cos = heading[..., 0]
            sin = heading[..., 1]

            def to_local(vectors: Tensor) -> Tensor:
                x, y = vectors[..., 0], vectors[..., 1]
                return torch.stack(
                    (x * cos + y * sin, -x * sin + y * cos), dim=-1
                )

            relative_position = to_local(relative_position)
            relative_velocity = to_local(relative_velocity)
        distance = torch.linalg.vector_norm(
            relative_position, dim=-1, keepdim=True
        )
        dot = (relative_position * relative_velocity).sum(
            dim=-1, keepdim=True
        )
        velocity_sq = relative_velocity.square().sum(
            dim=-1, keepdim=True
        )
        closing = -dot / distance.clamp_min(1.0e-3)
        ttc = (-dot / velocity_sq.clamp_min(1.0e-4)).clamp(0.0, 10.0)
        closest = relative_position + ttc * relative_velocity
        dca = torch.linalg.vector_norm(closest, dim=-1, keepdim=True)
        features = torch.cat(
            (
                relative_position,
                relative_velocity,
                distance,
                closing,
                ttc / 10.0,
                dca,
            ),
            dim=-1,
        ).permute(0, 2, 3, 1, 4)
        return features, distance[:, -1, :, :, 0], closing[:, -1, :, :, 0]

    def relation_tokens(self, past: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        features, last_distance, last_closing = self.relation_features(past)
        encoded = self.relation_encoder(features)
        if self.use_multiscale_relation:
            recent2 = encoded[:, :, :, -2:].mean(dim=3)
            recent4 = encoded[:, :, :, -4:].mean(dim=3)
            full = encoded.mean(dim=3)
            scales = torch.stack((recent2, recent4, full), dim=3)
            scale_weights = torch.softmax(self.relation_scale_score(scales), dim=3)
            tokens = (scale_weights * scales).sum(dim=3)
        else:
            weights = torch.softmax(self.relation_time_score(encoded), dim=3)
            tokens = (weights * encoded).sum(dim=3)
        # TTC and DCA are the final two scalar features after normalization.
        final_ttc = features[:, :, :, -1, 6] * 10.0
        final_dca = features[:, :, :, -1, 7]
        return tokens, last_distance, last_closing, final_ttc + final_dca

    def allowed_from_mask(
        self,
        mask: Optional[Tensor],
        batch: int,
        valid: Tensor,
    ) -> Tensor:
        if mask is None:
            allowed = valid[:, :, None] & valid[:, None, :]
        elif mask.ndim == 2:
            indices = (self.obs_len - 1) * self.agents + torch.arange(
                self.agents, device=valid.device
            )
            base = torch.isfinite(mask[indices][:, indices])
            allowed = base[None].expand(batch, -1, -1)
            allowed = allowed & valid[:, :, None] & valid[:, None, :]
        else:
            expanded = mask.reshape(
                batch, self.heads, mask.shape[-2], mask.shape[-1]
            )
            indices = (self.obs_len - 1) * self.agents + torch.arange(
                self.agents, device=valid.device
            )
            base = torch.isfinite(
                expanded[:, 0][:, indices][:, :, indices]
            )
            allowed = base & valid[:, :, None] & valid[:, None, :]
        eye = torch.eye(
            self.agents, dtype=torch.bool, device=valid.device
        )[None]
        return allowed | (eye & valid[:, :, None])

    def dual_graph(
        self,
        distance: Tensor,
        closing: Tensor,
        risk_distance: Tensor,
        valid: Tensor,
    ) -> Tensor:
        batch, agents, _ = distance.shape
        pair_valid = valid[:, :, None] & valid[:, None, :]
        eye = torch.eye(agents, dtype=torch.bool, device=distance.device)[None]
        pair_valid = pair_valid & ~eye

        local_score = -distance
        risk_score = (
            1.5 * torch.relu(closing)
            - 0.5 * risk_distance
            - 0.1 * distance
        )
        local_score = local_score.masked_fill(~pair_valid, -1.0e9)
        risk_score = risk_score.masked_fill(~pair_valid, -1.0e9)
        per_channel = max(1, min(self.top_k // 2, agents - 1))
        local_index = local_score.topk(per_channel, dim=-1).indices
        risk_index = risk_score.topk(per_channel, dim=-1).indices
        allowed = torch.zeros_like(pair_valid)
        allowed.scatter_(-1, local_index, True)
        allowed.scatter_(-1, risk_index, True)
        allowed = allowed & pair_valid
        return allowed | (eye & valid[:, :, None])

    def ttc_dca_graph(self, past: Tensor, valid: Tensor) -> Tensor:
        """Sparse local/conflict graph using TTC and DCA as separate quantities."""
        agents = past.shape[1]
        position = past[:, :, -1]
        velocity = (past[:, :, -1] - past[:, :, -2]) * self.frequency
        relative_position = position[:, :, None] - position[:, None, :]
        relative_velocity = velocity[:, :, None] - velocity[:, None, :]
        distance = torch.linalg.vector_norm(relative_position, dim=-1)
        dot = (relative_position * relative_velocity).sum(dim=-1)
        relative_speed_sq = relative_velocity.square().sum(dim=-1)
        raw_ttc = -dot / relative_speed_sq.clamp_min(1.0e-4)
        ttc = raw_ttc.clamp(0.0, 8.0)
        dca = torch.linalg.vector_norm(
            relative_position + ttc[..., None] * relative_velocity, dim=-1
        )
        pair_valid = valid[:, :, None] & valid[:, None, :]
        eye = torch.eye(agents, dtype=torch.bool, device=past.device)[None]
        candidates = pair_valid & ~eye

        local_score = (-distance).masked_fill(~candidates, -1.0e9)
        conflict_valid = candidates & (raw_ttc > 0.0) & (raw_ttc < 6.0) & (dca < 3.0)
        conflict_score = (
            torch.exp(-ttc / 2.5) * torch.exp(-dca / 1.25)
            + 0.10 * torch.exp(-distance / 3.0)
        ).masked_fill(~conflict_valid, -1.0e9)

        local_count = max(1, min(2, agents - 1))
        risk_count = max(1, min(self.top_k - local_count, agents - 1))
        local_values, local_indices = local_score.topk(local_count, dim=-1)
        risk_values, risk_indices = conflict_score.topk(risk_count, dim=-1)
        allowed = torch.zeros_like(candidates)
        allowed.scatter_(-1, local_indices, local_values > -1.0e8)
        allowed.scatter_(-1, risk_indices, risk_values > -1.0e8)
        return (allowed & candidates) | (eye & valid[:, :, None])

    def dynamic_routing(
        self,
        agent: Tensor,
        relation: Tensor,
        valid: Tensor,
    ) -> Tensor:
        """Bi-level-style learned coarse routing followed by sparse attention."""
        batch, agents, _ = agent.shape
        query = self.route_query(agent)
        key = self.route_key(agent)
        scores = torch.einsum("bid,bjd->bij", query, key)
        scores = scores / math.sqrt(float(query.shape[-1]))
        scores = scores + self.route_relation(relation).squeeze(-1)
        pair_valid = valid[:, :, None] & valid[:, None, :]
        eye = torch.eye(agents, dtype=torch.bool, device=agent.device)[None]
        candidates = pair_valid & ~eye
        scores = scores.masked_fill(~candidates, -1.0e4)
        count = max(1, min(self.top_k, agents - 1))
        indices = scores.topk(count, dim=-1).indices
        allowed = torch.zeros_like(pair_valid)
        allowed.scatter_(-1, indices, True)
        allowed = allowed & candidates
        return allowed | (eye & valid[:, :, None])

    def grouped_attention_mask(
        self,
        distance: Tensor,
        closing: Tensor,
        risk_distance: Tensor,
        valid: Tensor,
    ) -> Tensor:
        """Assign local and collision-risk neighborhoods to separate heads."""
        batch, agents, _ = distance.shape
        pair_valid = valid[:, :, None] & valid[:, None, :]
        eye = torch.eye(agents, dtype=torch.bool, device=distance.device)[None]
        candidates = pair_valid & ~eye
        count = max(1, min(self.top_k, agents - 1))
        local_score = (-distance).masked_fill(~candidates, -1.0e4)
        risk_score = (
            1.5 * torch.relu(closing) - 0.5 * risk_distance - 0.1 * distance
        ).masked_fill(~candidates, -1.0e4)
        local = torch.zeros_like(pair_valid)
        risk = torch.zeros_like(pair_valid)
        local.scatter_(-1, local_score.topk(count, dim=-1).indices, True)
        risk.scatter_(-1, risk_score.topk(count, dim=-1).indices, True)
        local = (local & candidates) | (eye & valid[:, :, None])
        risk = (risk & candidates) | (eye & valid[:, :, None])
        split = max(1, self.heads // 2)
        masks = [local if head < split else risk for head in range(self.heads)]
        return torch.stack(masks, dim=1)

    def dual_scale_sparse_mask(
        self,
        base_allowed: Tensor,
        distance: Tensor,
        closing: Tensor,
        risk_distance: Tensor,
        valid: Tensor,
    ) -> Tensor:
        """Keep the local graph and add at most two distant conflict edges.

        The original graph is radius limited.  The extra branch searches beyond
        the immediate 3 m neighborhood and ranks agents by closing motion and
        approximate TTC+DCA risk, rather than by distance alone.
        """
        agents = distance.shape[-1]
        pair_valid = valid[:, :, None] & valid[:, None, :]
        eye = torch.eye(agents, dtype=torch.bool, device=distance.device)[None]
        candidates = pair_valid & ~eye & (distance >= 3.0) & (distance <= 12.0)
        candidates = candidates & ~base_allowed
        far_score = (
            1.75 * torch.relu(closing)
            - 0.30 * risk_distance
            - 0.04 * distance
        ).masked_fill(~candidates, -1.0e9)
        keep = max(1, min(2, agents - 1))
        values, indices = far_score.topk(keep, dim=-1)
        selected = torch.zeros_like(pair_valid)
        selected.scatter_(-1, indices, values > -1.0e8)
        selected = selected & candidates
        return base_allowed | selected | (eye & valid[:, :, None])

    def relation_type_features(self, past: Tensor) -> Tensor:
        """Soft following/opposing/crossing/conflict relation indicators."""
        frequency = self.frequency
        velocity = (past[:, :, -1] - past[:, :, -2]) * frequency
        speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
        heading = velocity / speed.clamp_min(1.0e-4)
        receiver_heading = heading[:, :, None, :]
        source_heading = heading[:, None, :, :]
        direction_cos = (receiver_heading * source_heading).sum(-1).clamp(-1.0, 1.0)
        direction_cross = (
            receiver_heading[..., 0] * source_heading[..., 1]
            - receiver_heading[..., 1] * source_heading[..., 0]
        ).abs()

        relative = past[:, :, -1, None, :] - past[:, None, :, -1, :]
        distance = torch.linalg.vector_norm(relative, dim=-1)
        bearing = relative / distance[..., None].clamp_min(1.0e-4)
        ahead = (bearing * receiver_heading).sum(-1)
        relative_velocity = velocity[:, :, None, :] - velocity[:, None, :, :]
        dot = (relative * relative_velocity).sum(-1)
        rel_speed_sq = relative_velocity.square().sum(-1)
        ttc = (-dot / rel_speed_sq.clamp_min(1.0e-4)).clamp(0.0, 8.0)
        closest = relative + ttc[..., None] * relative_velocity
        dca = torch.linalg.vector_norm(closest, dim=-1)
        moving = (speed[:, :, 0] > 0.05)[:, :, None] & (speed[:, :, 0] > 0.05)[:, None, :]

        following = torch.relu(direction_cos) * torch.sigmoid(3.0 * ahead)
        opposing = torch.relu(-direction_cos)
        crossing = direction_cross
        closing = torch.relu(-dot / distance.clamp_min(1.0e-3))
        conflict = torch.exp(-dca / 1.5) * torch.exp(-ttc / 3.0) * (dot < 0).to(past.dtype)
        range_band = torch.exp(-distance / 4.0)
        features = torch.stack(
            (following, opposing, crossing, closing / 1.5, conflict, range_band),
            dim=-1,
        )
        return features * moving[..., None].to(features.dtype)

    def second_order_enrichment(
        self, temporal: Tensor, agent: Tensor, allowed: Tensor, valid: Tensor
    ) -> Tensor:
        """Inject a sparse one-hop group summary into each source trajectory."""
        base_allowed = allowed.any(dim=1) if allowed.ndim == 4 else allowed
        agents = agent.shape[1]
        eye = torch.eye(agents, dtype=torch.bool, device=agent.device)[None]
        neighbors = base_allowed & ~eye & valid[:, :, None] & valid[:, None, :]
        weights = neighbors.to(agent.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        context = torch.einsum("bij,bjd->bid", weights, agent)
        context = self.second_order_proj(context)
        enriched_agent = self.second_order_norm(
            agent + torch.sigmoid(self.second_order_gate) * context
        )
        delta = enriched_agent - agent
        return temporal + delta[:, :, None, :]

    @staticmethod
    def sparsemax(scores: Tensor, dim: int = -1) -> Tensor:
        shifted = scores - scores.amax(dim=dim, keepdim=True)
        sorted_scores, _ = torch.sort(shifted, dim=dim, descending=True)
        size = scores.shape[dim]
        ranks_shape = [1] * scores.ndim
        ranks_shape[dim] = size
        ranks = torch.arange(
            1, size + 1, device=scores.device, dtype=scores.dtype
        ).reshape(ranks_shape)
        cumulative = sorted_scores.cumsum(dim)
        support = (1.0 + ranks * sorted_scores) > cumulative
        support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
        tau = (
            cumulative.gather(dim, support_size - 1) - 1.0
        ) / support_size.to(scores.dtype)
        return torch.clamp(shifted - tau, min=0.0)

    @staticmethod
    def entmax15(scores: Tensor, dim: int = -1) -> Tensor:
        """Closed-form alpha=1.5 entmax; between softmax and sparsemax."""
        shifted = scores / 2.0
        shifted = shifted - shifted.amax(dim=dim, keepdim=True)
        sorted_scores, _ = torch.sort(shifted, dim=dim, descending=True)
        size = scores.shape[dim]
        ranks_shape = [1] * scores.ndim
        ranks_shape[dim] = size
        ranks = torch.arange(
            1, size + 1, device=scores.device, dtype=scores.dtype
        ).reshape(ranks_shape)
        mean = sorted_scores.cumsum(dim) / ranks
        mean_square = sorted_scores.square().cumsum(dim) / ranks
        variance_sum = ranks * (mean_square - mean.square())
        delta = ((1.0 - variance_sum) / ranks).clamp_min(0.0)
        # A tiny floor avoids infinite sqrt gradients when a large masked
        # support produces an exactly zero discriminant.
        taus = mean - torch.sqrt(delta.clamp_min(1.0e-12))
        support_size = (taus <= sorted_scores).sum(
            dim=dim, keepdim=True
        ).clamp_min(1)
        tau = taus.gather(dim, support_size - 1)
        return torch.clamp(shifted - tau, min=0.0).square()

    def social_attention(
        self,
        agent: Tensor,
        relation: Tensor,
        allowed: Tensor,
        risk_score: Optional[Tensor] = None,
    ) -> Tensor:
        batch, agents, _ = agent.shape
        q = self.q_proj(agent).reshape(
            batch, agents, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        k = self.k_proj(agent).reshape(
            batch, agents, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.v_proj(agent).reshape(
            batch, agents, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        if self.use_relation_qk:
            shifts = self.relation_qk(relation).reshape(
                batch, agents, agents, 2, self.heads, self.head_dim
            ).permute(0, 4, 1, 2, 3, 5)
            q_pair = q[:, :, :, None, :] + shifts[..., 0, :]
            k_pair = k[:, :, None, :, :] + shifts[..., 1, :]
            scores = (q_pair * k_pair).sum(dim=-1)
        else:
            scores = torch.einsum("bhid,bhjd->bhij", q, k)
        scores = scores / math.sqrt(float(self.head_dim))
        if self.use_relation_tokens:
            edge_bias = self.relation_bias(relation).permute(0, 3, 1, 2)
            scores = scores + edge_bias
        if self.use_soft_risk_bias and risk_score is not None:
            strength = torch.tanh(self.risk_strength).view(1, self.heads, 1, 1)
            scores = scores + strength * risk_score[:, None].clamp(-5.0, 5.0)
        attention_mask = allowed if allowed.ndim == 4 else allowed[:, None]
        scores = scores.masked_fill(~attention_mask, -1.0e4)
        if self.use_sparsemax:
            weights = self.sparsemax(scores, dim=-1)
        elif self.use_entmax15:
            weights = self.entmax15(scores, dim=-1)
        else:
            weights = torch.softmax(scores, dim=-1)

        base_message = v[:, :, None].expand(-1, -1, agents, -1, -1)
        if self.use_relation_tokens:
            edge_value = self.relation_value(relation).reshape(
                batch, agents, agents, self.heads, self.head_dim
            ).permute(0, 3, 1, 2, 4)
            message = base_message + edge_value
        else:
            message = base_message
        social = (weights[..., None] * message).sum(dim=3)
        social = social.permute(0, 2, 1, 3).reshape(batch, agents, self.dim)
        return self.out_proj(social)

    def temporal_social_attention(
        self,
        temporal: Tensor,
        agent: Tensor,
        relation: Tensor,
        allowed: Tensor,
        type_features: Optional[Tensor] = None,
        risk_score: Optional[Tensor] = None,
    ) -> Tensor:
        """Let each agent query all observed states of its sparse neighbors."""
        batch, agents, steps, _ = temporal.shape
        if self.use_recent_exact_early_pool:
            if steps < 3:
                raise ValueError("recent-exact pooling requires at least 3 steps")
            temporal = torch.cat(
                (temporal[:, :, :-2].mean(dim=2, keepdim=True), temporal[:, :, -2:]),
                dim=2,
            )
            steps = 3
        elif self.use_motion_event_sampling:
            temporal = self.motion_event_tokens(temporal)
            steps = temporal.shape[2]
        elif self.use_motion_token_squeeze:
            temporal = self.motion_squeezed_tokens(temporal)
            steps = temporal.shape[2]
        elif self.use_fixed_half_projection:
            temporal = torch.einsum(
                "lt,bntd->bnld", self.fixed_half_projection, temporal
            )
            steps = 2
        elif self.use_temporal_half_pool:
            if steps % 2:
                raise ValueError("temporal half pooling requires an even obs_len")
            temporal = temporal.reshape(
                batch, agents, 2, steps // 2, self.dim
            ).mean(dim=3)
            steps = 2
        elif self.temporal_landmarks:
            landmark_weights = torch.softmax(
                self.temporal_landmark_logits, dim=-1
            )
            temporal = torch.einsum(
                "lt,bntd->bnld", landmark_weights, temporal
            )
            steps = self.temporal_landmarks
        if self.use_motion_adaptive_key_smoothing:
            temporal = self.motion_adaptive_key_smoothing(temporal)
        if self.use_graph_additive_attention:
            return self.graph_additive_attention(
                temporal, agent, relation, allowed
            )
        if self.use_cascaded_group_attention:
            return self.cascaded_temporal_social_attention(
                temporal, agent, relation, allowed
            )
        if self.use_edge_list_attention:
            return self.edge_list_temporal_social_attention(
                temporal, agent, relation, allowed
            )
        pair_temporal = None
        if self.use_pair_deformable_temporal:
            pair_temporal = self.pair_deformable_temporal_tokens(
                temporal, agent, relation
            )
            steps = pair_temporal.shape[3]
        q = self.q_proj(agent).reshape(
            batch, agents, self.heads, self.qk_head_dim
        ).permute(0, 2, 1, 3)
        if pair_temporal is not None:
            k = self.k_proj(pair_temporal).reshape(
                batch, agents, agents, steps,
                self.kv_heads, self.qk_head_dim,
            ).permute(0, 4, 1, 2, 3, 5)
            v = self.v_proj(pair_temporal).reshape(
                batch, agents, agents, steps,
                self.kv_heads, self.head_dim,
            ).permute(0, 4, 1, 2, 3, 5)
        else:
            k = self.k_proj(temporal).reshape(
                batch, agents, steps, self.kv_heads, self.qk_head_dim
            ).permute(0, 3, 1, 2, 4)
        if pair_temporal is None and self.use_identity_value:
            v = temporal.reshape(
                batch, agents, steps, self.heads, self.head_dim
            ).permute(0, 3, 1, 2, 4)
        elif pair_temporal is None:
            v = self.v_proj(temporal).reshape(
                batch, agents, steps, self.kv_heads, self.head_dim
            ).permute(0, 3, 1, 2, 4)
        if self.kv_heads != self.heads:
            repeats = self.heads // self.kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        secondary_scores = None
        if self.use_differential_attention:
            if pair_temporal is not None:
                raise ValueError("differential attention does not support pair K/V")
            half = self.qk_head_dim // 2
            scores = torch.einsum(
                "bhid,bhjtd->bhijt", q[..., :half], k[..., :half]
            ) / math.sqrt(float(half))
            secondary_scores = torch.einsum(
                "bhid,bhjtd->bhijt", q[..., half:], k[..., half:]
            ) / math.sqrt(float(half))
        elif self.use_scaled_cosine_attention:
            q_similarity = F.normalize(q, dim=-1, eps=1.0e-6)
            k_similarity = F.normalize(k, dim=-1, eps=1.0e-6)
            if pair_temporal is not None:
                scores = torch.einsum(
                    "bhid,bhijtd->bhijt", q_similarity, k_similarity
                )
            else:
                scores = torch.einsum(
                    "bhid,bhjtd->bhijt", q_similarity, k_similarity
                )
            logit_scale = self.cosine_logit_scale.clamp(max=math.log(100.0))
            scores = scores * logit_scale.exp().view(1, self.heads, 1, 1, 1)
        else:
            if pair_temporal is not None:
                scores = torch.einsum("bhid,bhijtd->bhijt", q, k)
            else:
                scores = torch.einsum("bhid,bhjtd->bhijt", q, k)
            scores = scores / math.sqrt(float(self.qk_head_dim))
        if self.use_pair_temporal_soft_prior:
            target = agent[:, :, None, :].expand(-1, -1, agents, -1)
            source = agent[:, None, :, :].expand(-1, agents, -1, -1)
            residual = torch.tanh(
                self.kdta_offset(torch.cat((target, source, relation), dim=-1))
            )
            base_position = self.kdta_base_positions.to(
                device=temporal.device, dtype=temporal.dtype
            )
            offset_range = self.kdta_offset_ranges.to(
                device=temporal.device, dtype=temporal.dtype
            )
            event_position = (
                base_position.view(1, 1, 1, 1)
                + residual * offset_range.view(1, 1, 1, 1)
            ).clamp(0.0, float(steps - 2))
            self.last_kdta_positions = event_position.detach()
            time_grid = torch.arange(
                steps, device=temporal.device, dtype=temporal.dtype
            ).view(1, 1, 1, steps)
            temporal_prior = torch.exp(
                -0.5 * ((time_grid - event_position) / 1.5).square()
            )
            prior_strength = torch.tanh(self.temporal_prior_strength).view(
                1, self.heads, 1, 1, 1
            )
            scores = scores + prior_strength * temporal_prior[:, None]
            if secondary_scores is not None:
                secondary_scores = secondary_scores + prior_strength * temporal_prior[:, None]
        if self.use_relation_tokens:
            edge_bias = self.relation_bias(relation).permute(0, 3, 1, 2)
            scores = scores + edge_bias[..., None]
            if secondary_scores is not None:
                secondary_scores = secondary_scores + edge_bias[..., None]
        if self.use_relation_type_bias and type_features is not None:
            explicit_bias = self.type_bias(type_features).permute(0, 3, 1, 2)
            scores = scores + explicit_bias[..., None]
            if secondary_scores is not None:
                secondary_scores = secondary_scores + explicit_bias[..., None]
        if self.use_soft_risk_bias and risk_score is not None:
            strength = torch.tanh(self.risk_strength).view(1, self.heads, 1, 1, 1)
            scores = scores + strength * risk_score[:, None, :, :, None].clamp(-5.0, 5.0)
            if secondary_scores is not None:
                secondary_scores = secondary_scores + strength * risk_score[:, None, :, :, None].clamp(-5.0, 5.0)
        if self.use_temporal_relative_bias:
            scores = scores + self.temporal_relative_bias[None, :, None, None, :]
            if secondary_scores is not None:
                secondary_scores = secondary_scores + self.temporal_relative_bias[None, :, None, None, :]
        expanded_mask = allowed[:, None, :, :, None].expand_as(scores)
        if self.use_temporal_dropkey and self.training:
            drop = torch.rand_like(scores) < 0.10
            drop[..., -1] = False
            expanded_mask = expanded_mask & ~drop
        scores = scores.masked_fill(~expanded_mask, -1.0e4)
        if secondary_scores is not None:
            secondary_scores = secondary_scores.masked_fill(~expanded_mask, -1.0e4)
            first = torch.softmax(scores.flatten(-2), dim=-1).reshape_as(scores)
            second = torch.softmax(
                secondary_scores.flatten(-2), dim=-1
            ).reshape_as(scores)
            differential_lambda = torch.sigmoid(
                self.differential_lambda_logit
            ).clamp(max=0.20).view(1, self.heads, 1, 1, 1)
            weights = (first - differential_lambda * second) / (
                1.0 - differential_lambda
            )
            self.last_differential_negative_mass = (
                (-weights).clamp_min(0.0).sum(dim=(3, 4)).mean().detach()
            )
        elif self.use_relation_grouped_vector_attention:
            groups = self.vector_groups
            group_dim = self.qk_head_dim // groups
            q_group = q.reshape(batch, self.heads, agents, groups, group_dim)
            k_group = k.reshape(
                batch, self.heads, agents, steps, groups, group_dim
            )
            group_signal = torch.einsum(
                "bhigd,bhjtgd->bhijtg", q_group, k_group
            ) / math.sqrt(float(group_dim))
            group_signal = group_signal - group_signal.mean(dim=-1, keepdim=True)
            relation_correction = self.relation_group_correction(relation).reshape(
                batch, agents, agents, self.heads, groups
            ).permute(0, 3, 1, 2, 4)
            correction_strength = torch.tanh(
                self.group_correction_strength
            ).view(1, self.heads, 1, 1, 1, groups)
            group_scores = scores[..., None] + correction_strength * (
                group_signal + relation_correction[..., None, :]
            )
            group_mask = expanded_mask[..., None].expand_as(group_scores)
            group_scores = group_scores.masked_fill(~group_mask, -1.0e4)
            group_flat = group_scores.permute(0, 1, 2, 5, 3, 4).flatten(-2)
            group_weights = torch.softmax(group_flat, dim=-1)
            weights = group_weights.reshape(
                batch, self.heads, agents, groups, agents, steps
            ).permute(0, 1, 2, 4, 5, 3)
        else:
            flat_scores = scores.flatten(-2)
            if self.use_sparsemax:
                flat_weights = self.sparsemax(flat_scores, dim=-1)
            elif self.use_entmax15:
                flat_weights = self.entmax15(flat_scores, dim=-1)
            else:
                flat_weights = torch.softmax(flat_scores, dim=-1)
            weights = flat_weights.reshape_as(scores)
        if pair_temporal is not None:
            base = v
        else:
            base = v[:, :, None].expand(-1, -1, agents, -1, -1, -1)
        if self.use_relation_tokens:
            edge_value = self.relation_value(relation).reshape(
                batch, agents, agents, self.heads, self.head_dim
            ).permute(0, 3, 1, 2, 4)
            message = base + edge_value[..., None, :]
        else:
            message = base
        if self.use_relation_grouped_vector_attention:
            group_dim = self.head_dim // self.vector_groups
            grouped_message = message.reshape(
                batch, self.heads, agents, agents, steps,
                self.vector_groups, group_dim,
            )
            social = (weights[..., None] * grouped_message).sum(dim=(3, 4))
            social = social.reshape(batch, self.heads, agents, self.head_dim)
        else:
            social = (weights[..., None] * message).sum(dim=(3, 4))
        social = social.permute(0, 2, 1, 3).reshape(batch, agents, self.dim)
        return self.out_proj(social)

    def motion_adaptive_key_smoothing(self, temporal: Tensor) -> Tensor:
        """Denoise quiet motion locally while protecting turns and the last frame."""
        if self._past is None:
            raise RuntimeError("Motion-adaptive smoothing requires trajectory context.")
        steps = temporal.shape[2]
        if steps < 3:
            return temporal
        previous = torch.cat((temporal[:, :, :1], temporal[:, :, :-1]), dim=2)
        following = torch.cat((temporal[:, :, 1:], temporal[:, :, -1:]), dim=2)
        logits = self.key_smoothing_logits(temporal)

        velocity = self._past[:, :, 1:] - self._past[:, :, :-1]
        acceleration = torch.linalg.vector_norm(
            velocity[:, :, 1:] - velocity[:, :, :-1], dim=-1
        )
        acceleration = F.pad(acceleration, (1, 1), mode="replicate")
        quiet_motion = 1.0 / (1.0 + 4.0 * self.frequency * acceleration)
        boundary = torch.ones_like(quiet_motion)
        boundary[..., 0] = 0.5
        boundary[..., -1] = 0.25
        maximum_mix = 0.20 * quiet_motion * boundary
        alpha = torch.sigmoid(logits) * maximum_mix[..., None]
        alpha_left, alpha_right = alpha.unbind(dim=-1)
        left_mask = torch.ones(steps, device=temporal.device, dtype=temporal.dtype)
        right_mask = torch.ones_like(left_mask)
        left_mask[0] = 0.0
        right_mask[-1] = 0.0
        alpha_left = alpha_left * left_mask
        alpha_right = alpha_right * right_mask
        self.last_key_smoothing = torch.stack(
            (alpha_left, alpha_right), dim=-1
        ).detach()
        return (
            temporal
            + alpha_left[..., None] * (previous - temporal)
            + alpha_right[..., None] * (following - temporal)
        )

    def pair_deformable_temporal_tokens(
        self, temporal: Tensor, agent: Tensor, relation: Tensor
    ) -> Tensor:
        """Sample pair-conditioned history tokens on the temporal axis.

        Agent indices remain discrete and are never interpolated.  For every
        target-source pair, two bounded continuous history locations are
        predicted and linearly interpolated; the latest observation is always
        retained as the third token.
        """
        batch, agents, steps, dim = temporal.shape
        target = agent[:, :, None, :].expand(-1, -1, agents, -1)
        source = agent[:, None, :, :].expand(-1, agents, -1, -1)
        offset_input = torch.cat((target, source, relation), dim=-1)
        residual = torch.tanh(self.kdta_offset(offset_input))
        base = self.kdta_base_positions.to(
            device=temporal.device, dtype=temporal.dtype
        )
        ranges = self.kdta_offset_ranges.to(
            device=temporal.device, dtype=temporal.dtype
        )
        count = int(base.numel())
        positions = (
            base.view(1, 1, 1, count)
            + residual * ranges.view(1, 1, 1, count)
        )
        positions = positions.clamp(0.0, float(steps - 1))
        if self.kdta_recent_exact:
            recent = torch.stack(
                (
                    torch.full_like(positions[..., 0], float(steps - 2)),
                    torch.full_like(positions[..., 0], float(steps - 1)),
                ),
                dim=-1,
            )
            positions = torch.cat((positions, recent), dim=-1)
        else:
            latest = torch.full_like(positions[..., :1], float(steps - 1))
            positions = torch.cat((positions, latest), dim=-1)
        self.last_kdta_positions = positions.detach()

        left = positions.floor().long()
        right = (left + 1).clamp(max=steps - 1)
        alpha = (positions - left.to(positions.dtype))[..., None]
        pair_history = temporal[:, None].expand(-1, agents, -1, -1, -1)
        gather_shape = (*left.shape, dim)
        left_value = torch.gather(
            pair_history, 3,
            left[..., None].expand(gather_shape),
        )
        right_value = torch.gather(
            pair_history, 3,
            right[..., None].expand(gather_shape),
        )
        sampled = left_value + alpha * (right_value - left_value)
        if self.kdta_include_trend:
            # A stable early-history summary restores path-shape context that
            # one deformable point alone cannot represent.  It is deliberately
            # parameter-free and complements, rather than replaces, the
            # pair-conditioned event token.
            trend = temporal[:, :, :-2].mean(dim=2)
            trend = trend[:, None, :, None, :].expand(
                -1, agents, -1, -1, -1
            )
            sampled = torch.cat((trend, sampled), dim=3)
        return sampled

    def motion_event_tokens(self, temporal: Tensor) -> Tensor:
        """Keep long-term context, the strongest motion event, and the last state."""
        if self._past is None:
            raise RuntimeError("Motion-event sampling requires trajectory context.")
        steps = temporal.shape[2]
        if steps < 4:
            return temporal
        velocity = self._past[:, :, 1:] - self._past[:, :, :-1]
        acceleration = velocity[:, :, 1:] - velocity[:, :, :-1]
        event_score = torch.linalg.vector_norm(acceleration, dim=-1)
        # The acceleration at index k corresponds to temporal state k+2.
        event_index = event_score.argmax(dim=-1) + 2
        event_index = event_index.clamp(max=steps - 2)
        gather_index = event_index[..., None, None].expand(
            -1, -1, 1, temporal.shape[-1]
        )
        event = temporal.gather(2, gather_index)
        early = temporal[:, :, :-2].mean(dim=2, keepdim=True)
        latest = temporal[:, :, -1:]
        return torch.cat((early, event, latest), dim=2)

    def motion_squeezed_tokens(self, temporal: Tensor) -> Tensor:
        """Merge eight history tokens into four motion-consistent clusters."""
        if self._past is None:
            raise RuntimeError("Motion token squeezing requires trajectory context.")
        batch, agents, steps, dim = temporal.shape
        if steps <= 4:
            return temporal
        velocity = torch.zeros_like(self._past)
        velocity[:, :, 1:] = self._past[:, :, 1:] - self._past[:, :, :-1]
        velocity[:, :, 0] = velocity[:, :, 1]
        acceleration = torch.linalg.vector_norm(
            velocity[:, :, 1:] - velocity[:, :, :-1], dim=-1
        )
        interior_score = acceleration[:, :, 1:-1]
        event_indices = interior_score.topk(2, dim=-1).indices + 2
        endpoints = torch.empty(
            batch, agents, 2, dtype=torch.long, device=temporal.device
        )
        endpoints[..., 0] = 0
        endpoints[..., 1] = steps - 1
        centers = torch.cat((endpoints, event_indices), dim=-1).sort(dim=-1).values
        center_velocity = velocity.gather(
            2, centers[..., None].expand(-1, -1, -1, velocity.shape[-1])
        )
        distance = torch.linalg.vector_norm(
            velocity[:, :, :, None, :] - center_velocity[:, :, None, :, :],
            dim=-1,
        )
        assignment = distance.argmin(dim=-1)
        membership = F.one_hot(assignment, num_classes=4).to(temporal.dtype)
        counts = membership.sum(dim=2).clamp_min(1.0)
        merged = torch.einsum("bntk,bntd->bnkd", membership, temporal)
        return merged / counts[..., None]

    def edge_list_temporal_social_attention(
        self,
        temporal: Tensor,
        agent: Tensor,
        relation: Tensor,
        allowed: Tensor,
    ) -> Tensor:
        """Exact temporal-social attention evaluated only on valid graph edges."""
        batch, agents, steps, _ = temporal.shape
        q = self.q_proj(agent).reshape(
            batch, agents, self.heads, self.qk_head_dim
        )
        k = self.k_proj(temporal).reshape(
            batch, agents, steps, self.heads, self.qk_head_dim
        )
        v = self.v_proj(temporal).reshape(
            batch, agents, steps, self.heads, self.head_dim
        )
        edges = allowed.nonzero(as_tuple=False)
        batch_index, target_index, source_index = edges.unbind(dim=1)
        edge_q = q[batch_index, target_index]
        edge_k = k[batch_index, source_index]
        scores = torch.einsum("ehd,ethd->eht", edge_q, edge_k)
        scores = scores / math.sqrt(float(self.qk_head_dim))
        if self.use_relation_tokens:
            edge_bias = self.relation_bias(
                relation[batch_index, target_index, source_index]
            )
            scores = scores + edge_bias[..., None]

        group = (
            (batch_index[:, None] * agents + target_index[:, None]) * self.heads
            + torch.arange(self.heads, device=temporal.device)[None]
        )
        flat_group = group[..., None].expand(-1, -1, steps).reshape(-1)
        flat_scores = scores.reshape(-1)
        group_count = batch * agents * self.heads
        maxima = torch.full(
            (group_count,), -torch.inf, dtype=scores.dtype, device=scores.device
        )
        maxima.scatter_reduce_(0, flat_group, flat_scores, reduce="amax", include_self=True)
        exponent = torch.exp(flat_scores - maxima[flat_group])
        denominator = torch.zeros(
            group_count, dtype=scores.dtype, device=scores.device
        )
        denominator.scatter_add_(0, flat_group, exponent)
        weights = (exponent / denominator[flat_group].clamp_min(1.0e-12)).reshape_as(scores)

        message = v[batch_index, source_index]
        if self.use_relation_tokens:
            edge_value = self.relation_value(
                relation[batch_index, target_index, source_index]
            ).reshape(-1, self.heads, self.head_dim)
            message = message + edge_value[:, None]
        edge_output = (weights[..., None] * message.permute(0, 2, 1, 3)).sum(dim=2)
        output = torch.zeros(
            batch * agents * self.heads,
            self.head_dim,
            dtype=temporal.dtype,
            device=temporal.device,
        )
        output.scatter_add_(
            0,
            group.reshape(-1, 1).expand(-1, self.head_dim),
            edge_output.reshape(-1, self.head_dim),
        )
        output = output.reshape(batch, agents, self.heads, self.head_dim)
        # Dense F16 assigns a uniform distribution when a padded target has no
        # valid edges because every masked score is the same (-1e4).  Preserve
        # that boundary behavior so this variant changes execution only.
        empty_target = ~allowed.any(dim=-1)
        if empty_target.any():
            fallback = v.mean(dim=(1, 2))[:, None].expand(-1, agents, -1, -1)
            if self.use_relation_tokens:
                relation_fallback = self.relation_value(relation).reshape(
                    batch, agents, agents, self.heads, self.head_dim
                ).mean(dim=2)
                fallback = fallback + relation_fallback
            output = torch.where(empty_target[..., None, None], fallback, output)
        return self.out_proj(output.reshape(batch, agents, self.dim))

    def graph_additive_attention(
        self,
        temporal: Tensor,
        agent: Tensor,
        relation: Tensor,
        allowed: Tensor,
    ) -> Tensor:
        """SwiftFormer-style additive mixing under the kinematic graph.

        Source salience is computed once per agent rather than by every
        query-key pair.  The graph and relation bias still make each target's
        received context different.
        """
        batch, agents, steps, _ = temporal.shape
        time_logits = self.additive_time_score(temporal).squeeze(-1)
        source = (
            torch.softmax(time_logits, dim=-1)[..., None] * temporal
        ).sum(dim=2)
        source_scores = self.additive_source_score(agent).permute(0, 2, 1)
        scores = source_scores[:, :, None, :].expand(
            -1, -1, agents, -1
        )
        if self.use_relation_tokens:
            scores = scores + self.relation_bias(relation).permute(0, 3, 1, 2)
        attention_mask = allowed if allowed.ndim == 4 else allowed[:, None]
        weights = torch.softmax(
            scores.masked_fill(~attention_mask, -1.0e4), dim=-1
        )
        value = self.v_proj(source).reshape(
            batch, agents, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        base = value[:, :, None].expand(-1, -1, agents, -1, -1)
        if self.use_relation_tokens:
            edge = self.relation_value(relation).reshape(
                batch, agents, agents, self.heads, self.head_dim
            ).permute(0, 3, 1, 2, 4)
            message = base + edge
        else:
            message = base
        social = (weights[..., None] * message).sum(dim=3)
        social = social.permute(0, 2, 1, 3).reshape(
            batch, agents, self.dim
        )
        return self.out_proj(social)

    def cascaded_temporal_social_attention(
        self,
        temporal: Tensor,
        agent: Tensor,
        relation: Tensor,
        allowed: Tensor,
    ) -> Tensor:
        """Channel-grouped cross-attention with cascaded head features."""
        batch, agents, steps, _ = temporal.shape
        agent_groups = agent.reshape(
            batch, agents, self.heads, self.head_dim
        )
        temporal_groups = temporal.reshape(
            batch, agents, steps, self.heads, self.head_dim
        )
        relation_bias = self.relation_bias(relation).permute(0, 3, 1, 2)
        relation_value = self.relation_value(relation).reshape(
            batch, agents, agents, self.heads, self.head_dim
        )
        outputs = []
        previous: Optional[Tensor] = None
        for head in range(self.heads):
            query_input = agent_groups[:, :, head]
            key_value_input = temporal_groups[:, :, :, head]
            if previous is not None:
                query_input = query_input + previous
                key_value_input = key_value_input + previous[:, :, None]
            q = self.group_q_proj[head](query_input)
            k = self.group_k_proj[head](key_value_input)
            v = self.group_v_proj[head](key_value_input)
            scores = torch.einsum("bid,bjtd->bijt", q, k)
            scores = scores / math.sqrt(float(self.head_dim))
            scores = scores + relation_bias[:, head, :, :, None]
            expanded_mask = allowed[:, :, :, None].expand_as(scores)
            weights = torch.softmax(
                scores.masked_fill(~expanded_mask, -1.0e4).flatten(-2),
                dim=-1,
            ).reshape_as(scores)
            edge = relation_value[:, :, :, head]
            message = v[:, None] + edge[:, :, :, None]
            previous = (weights[..., None] * message).sum(dim=(2, 3))
            outputs.append(previous)
        return self.out_proj(torch.cat(outputs, dim=-1))

    def node_structure_features(
        self,
        distance: Tensor,
        closing: Tensor,
        risk_distance: Tensor,
        allowed: Tensor,
        valid: Tensor,
    ) -> Tensor:
        agents = distance.shape[-1]
        eye = torch.eye(agents, dtype=torch.bool, device=distance.device)[None]
        neighbors = allowed & ~eye & valid[:, :, None] & valid[:, None, :]
        count = neighbors.sum(dim=-1, keepdim=True)
        denominator = count.clamp_min(1).to(distance.dtype)
        degree = count.to(distance.dtype) / max(1.0, float(agents - 1))
        mean_distance = (
            distance * neighbors.to(distance.dtype)
        ).sum(dim=-1, keepdim=True) / denominator
        mean_risk_distance = (
            risk_distance * neighbors.to(distance.dtype)
        ).sum(dim=-1, keepdim=True) / denominator
        max_closing = closing.masked_fill(~neighbors, -1.0e4).amax(
            dim=-1, keepdim=True
        )
        max_closing = torch.where(count > 0, max_closing, torch.zeros_like(max_closing))
        return torch.cat(
            (degree, mean_distance, mean_risk_distance, max_closing), dim=-1
        )

    def proxy_attention(self, agent: Tensor, valid: Tensor) -> Tensor:
        batch, agents, _ = agent.shape
        proxies = self.agent_proxies[None].expand(batch, -1, -1)
        proxy_count = proxies.shape[1]
        proxy_q = self.q_proj(proxies).reshape(
            batch, proxy_count, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        agent_k = self.k_proj(agent).reshape(
            batch, agents, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        agent_v = self.v_proj(agent).reshape(
            batch, agents, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        collect_scores = torch.einsum("bhpd,bhjd->bhpj", proxy_q, agent_k)
        collect_scores = collect_scores / math.sqrt(float(self.head_dim))
        collect_scores = collect_scores.masked_fill(~valid[:, None, None], -1.0e4)
        collected = torch.einsum(
            "bhpj,bhjd->bhpd", torch.softmax(collect_scores, dim=-1), agent_v
        ).permute(0, 2, 1, 3).reshape(batch, proxy_count, self.dim)

        query = self.q_proj(agent).reshape(
            batch, agents, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        proxy_k = self.k_proj(collected).reshape(
            batch, proxy_count, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        proxy_v = self.v_proj(collected).reshape(
            batch, proxy_count, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        broadcast_scores = torch.einsum("bhid,bhpd->bhip", query, proxy_k)
        broadcast_scores = broadcast_scores / math.sqrt(float(self.head_dim))
        context = torch.einsum(
            "bhip,bhpd->bhid", torch.softmax(broadcast_scores, dim=-1), proxy_v
        ).permute(0, 2, 1, 3).reshape(batch, agents, self.dim)
        return self.out_proj(context)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch, _, dim = src.shape
        tokens = (
            src.reshape(batch, self.obs_len, self.agents, dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        flat = tokens.reshape(batch * self.agents, self.obs_len, dim)
        temporal = self.temporal_encoder(flat).reshape(
            batch, self.agents, self.obs_len, dim
        )
        if self.multiscale_temporal is not None:
            if self._past is None:
                raise RuntimeError("Multi-scale temporal mixer requires trajectory context.")
            temporal = self.multiscale_temporal(temporal, self._past, self.frequency)
        temporal, agent = self.temporal_queries(temporal)
        self.last_temporal_features = temporal

        if self._past is None:
            raise RuntimeError("Factorized encoder context was not set.")
        if self._num_valid is None:
            valid = torch.ones(
                batch, self.agents, dtype=torch.bool, device=src.device
            )
        else:
            ids = torch.arange(self.agents, device=src.device)
            valid = ids[None] < self._num_valid.long()[:, None]

        relation, distance, closing, risk_distance = self.relation_tokens(
            self._past
        )
        if self.use_ttc_dca_graph:
            allowed = self.ttc_dca_graph(self._past, valid)
        elif self.use_dynamic_routing:
            allowed = self.dynamic_routing(agent, relation, valid)
        elif self.use_grouped_attention:
            allowed = self.grouped_attention_mask(
                distance, closing, risk_distance, valid
            )
        elif self.use_dual_graph:
            allowed = self.dual_graph(
                distance, closing, risk_distance, valid
            )
        else:
            allowed = self.allowed_from_mask(mask, batch, valid)
        if self.use_dual_scale_sparse:
            allowed = self.dual_scale_sparse_mask(
                allowed, distance, closing, risk_distance, valid
            )
        self.last_allowed = allowed.detach()
        type_features = (
            self.relation_type_features(self._past)
            if self.use_relation_type_bias else None
        )
        if self.use_second_order_context:
            temporal = self.second_order_enrichment(temporal, agent, allowed, valid)
        risk_score = torch.relu(closing) - 0.15 * distance - 0.10 * risk_distance
        if self.use_node_structure:
            structure = self.node_structure_features(
                distance, closing, risk_distance, allowed, valid
            )
            agent = self.node_structure_norm(
                agent + torch.sigmoid(self.node_structure_gate)
                * self.node_structure_encoder(structure)
            )
        if self.use_density_adaptive_social:
            summary_social = self.social_attention(
                agent, relation, allowed, risk_score
            )
            temporal_social = self.temporal_social_attention(
                temporal, agent, relation, allowed
            )
            use_temporal = (valid.sum(dim=-1) > 2)[:, None, None]
            social = torch.where(
                use_temporal, temporal_social, summary_social
            )
        elif self.use_adaptive_dual_social:
            summary_social = self.social_attention(
                agent, relation, allowed, risk_score
            )
            temporal_social = self.temporal_social_attention(
                temporal, agent, relation, allowed
            )
            temporal_mix = torch.sigmoid(
                self.temporal_mix_gate(
                    torch.cat((agent, summary_social, temporal_social), dim=-1)
                )
            )
            social = (
                (1.0 - temporal_mix) * summary_social
                + temporal_mix * temporal_social
            )
        elif self.use_temporal_social_cross:
            social = self.temporal_social_attention(
                temporal, agent, relation, allowed, type_features, risk_score
            )
        else:
            social = self.social_attention(agent, relation, allowed, risk_score)
        gate = torch.sigmoid(
            self.social_gate(torch.cat((agent, social), dim=-1))
        )
        fused_agent = self.social_norm(agent + gate * social)
        if self.use_edge_iteration:
            pair_difference = (
                fused_agent[:, :, None, :] - fused_agent[:, None, :, :]
            )
            edge_delta = self.edge_refine(
                torch.cat((relation, pair_difference), dim=-1)
            )
            refined_relation = relation + torch.sigmoid(
                self.edge_refine_strength
            ) * edge_delta
            refined_social = self.social_attention(
                fused_agent, refined_relation, allowed, risk_score
            )
            second_gate = torch.sigmoid(
                self.second_social_gate(
                    torch.cat((fused_agent, refined_social), dim=-1)
                )
            )
            fused_agent = self.social_norm(
                fused_agent + second_gate * refined_social
            )
        if self.use_focal_conditioning:
            focal = fused_agent[:, :1]
            neighbor_mask = allowed[:, 0].clone()
            neighbor_mask[:, 0] = False
            has_neighbor = neighbor_mask.any(dim=-1, keepdim=True)
            query = self.focal_query(focal)
            keys = self.focal_key(fused_agent)
            scores = torch.einsum("bid,bjd->bij", query, keys)
            scores = scores / math.sqrt(float(query.shape[-1]))
            scores = scores.masked_fill(~neighbor_mask[:, None], -1.0e4)
            weights = torch.softmax(scores, dim=-1)
            weights = weights * neighbor_mask[:, None].to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            context = torch.einsum(
                "bij,bjd->bid", weights, self.focal_value(fused_agent)
            )
            context = self.focal_out(context)
            focal_gate = torch.sigmoid(
                self.focal_gate(torch.cat((focal, context), dim=-1))
            )
            conditioned = self.focal_norm(focal + focal_gate * context)
            conditioned = torch.where(has_neighbor[:, :, None], conditioned, focal)
            fused_agent = torch.cat((conditioned, fused_agent[:, 1:]), dim=1)
        if self.use_agent_proxies:
            proxy_context = self.proxy_attention(fused_agent, valid)
            proxy_gate = torch.sigmoid(
                self.proxy_gate(torch.cat((fused_agent, proxy_context), dim=-1))
            )
            fused_agent = self.proxy_norm(
                fused_agent + proxy_gate * proxy_context
            )
        self.last_agent_features = fused_agent
        enriched = self.output_norm(
            temporal + self.token_feedback(fused_agent)[:, :, None, :]
        )
        return (
            enriched.permute(0, 2, 1, 3)
            .reshape(batch, self.obs_len * self.agents, dim)
        )


class FrontEncoderASTRA(ASTRA_model):
    """ASTRA with a selectable front encoder and the untouched Q20 decoder."""

    def __init__(self, cfg, front_variant: str) -> None:
        if front_variant not in VARIANTS:
            raise ValueError(f"Unknown front variant: {front_variant}")
        self.front_variant = front_variant
        super().__init__(cfg)
        order = sorted(VARIANTS)
        variant_rank = order.index(front_variant)
        del variant_rank  # names below are explicit to avoid ordering mistakes

        if front_variant != "f0_baseline":
            state_class = (
                EnhancedKinematicState
                if front_variant == "f50_enhanced_kinematic_state"
                else RobustMultiScaleState
            )
            self.robust_state = state_class(self.token_dim, hidden_dim=64)
            # The robust state module has its own sample/time gate.  Keep the
            # legacy scalar near one so it does not suppress the new branch.
            self.self_state_gate = nn.Parameter(
                torch.tensor(2.65), requires_grad=False
            )
        else:
            self.robust_state = None

        if front_variant == "f2_temporal_queries":
            self.agent_aware_trans_encoder = LegacyPlusTemporalQueries(
                legacy=self.agent_aware_trans_encoder,
                obs_len=self.obs_len,
                agents=self.num_pedestrians,
                dim=self.token_dim,
                heads=int(self.cfg.MODEL.NHEAD),
                dropout=float(self.cfg.MODEL.DROPOUT),
            )
        elif front_variant in {
            "f3_factorized",
            "f4_relation_tokens",
            "f5_dual_graph",
            "f6_light_factorized",
            "f7_ultralight_factorized",
            "f8_heading_aligned_relation",
            "f9_soft_risk_bias",
            "f10_multiscale_relation",
            "f11_motion_auxiliary",
            "f12_focal_conditioned",
            "f13_dynamic_routing",
            "f14_grouped_attention",
            "f15_edge_iteration",
            "f16_temporal_social_cross",
            "f17_sparsemax_attention",
            "f18_relation_qk",
            "f19_entmax15_attention",
            "f20_node_structure_encoding",
            "f21_agent_proxy_attention",
            "f22_temporal_cross_entmax15",
            "f23_dynamic_temporal_cross",
            "f24_temporal_dropkey",
            "f25_temporal_relative_bias",
            "f26_adaptive_dual_social",
            "f27_density_adaptive_social",
            "f28_temporal_landmark4",
            "f29_temporal_landmark2",
            "f30_graph_additive_attention",
            "f31_cascaded_group_attention",
            "f32_grouped_query_attention",
            "f33_multi_query_attention",
            "f34_identity_value_attention",
            "f35_temporal_half_pool",
            "f36_fixed_half_projection",
            "f37_lowdim_qk_attention",
            "f38_recent_exact_early_pool",
            "f39_motion_event_sampling",
            "f40_motion_token_squeeze",
            "f41_edge_list_attention",
            "f42_dual_scale_sparse",
            "f43_relation_type_bias",
            "f44_second_order_context",
            "f45_temporal_risk_bias",
            "f46_multiscale_temporal_residual",
            "f47_multiscale_temporal_replace",
            "f48_ttc_dca_graph",
            "f49_ttc_dca_multiscale_replace",
            "f50_enhanced_kinematic_state",
            "f51_scaled_cosine_attention",
            "f52_pair_deformable_temporal",
            "f53_kdta_recent_exact",
            "f54_kdta_trend_event_recent",
            "f55_pair_temporal_soft_prior",
            "f56_motion_adaptive_key_smoothing",
            "f57_relation_grouped_vector_attention",
            "f58_compute_neutral_differential",
        }:
            self.agent_aware_trans_encoder = FactorizedStateRelationEncoder(
                obs_len=self.obs_len,
                agents=self.num_pedestrians,
                dim=self.token_dim,
                heads=int(self.cfg.MODEL.NHEAD),
                ffn_dim=(
                    (self.token_dim // 2 if front_variant == "f7_ultralight_factorized" else self.token_dim)
                    if front_variant in {"f6_light_factorized", "f7_ultralight_factorized"}
                    else int(self.cfg.MODEL.DIM_FEEDFORWARD)
                ),
                dropout=float(self.cfg.MODEL.DROPOUT),
                use_relation_tokens=front_variant in {
                    "f4_relation_tokens", "f5_dual_graph", "f6_light_factorized",
                    "f7_ultralight_factorized", "f8_heading_aligned_relation",
                    "f9_soft_risk_bias"
                    , "f10_multiscale_relation", "f11_motion_auxiliary",
                    "f12_focal_conditioned"
                    , "f13_dynamic_routing", "f14_grouped_attention",
                    "f15_edge_iteration", "f16_temporal_social_cross",
                    "f17_sparsemax_attention", "f18_relation_qk"
                    , "f19_entmax15_attention"
                    , "f20_node_structure_encoding", "f21_agent_proxy_attention"
                    , "f22_temporal_cross_entmax15", "f23_dynamic_temporal_cross"
                    , "f24_temporal_dropkey", "f25_temporal_relative_bias"
                    , "f26_adaptive_dual_social"
                    , "f27_density_adaptive_social"
                    , "f28_temporal_landmark4", "f29_temporal_landmark2"
                    , "f30_graph_additive_attention"
                    , "f31_cascaded_group_attention"
                    , "f32_grouped_query_attention"
                    , "f33_multi_query_attention"
                    , "f34_identity_value_attention"
                    , "f35_temporal_half_pool"
                    , "f36_fixed_half_projection"
                    , "f37_lowdim_qk_attention"
                    , "f38_recent_exact_early_pool"
                    , "f39_motion_event_sampling"
                    , "f40_motion_token_squeeze"
                    , "f41_edge_list_attention"
                    , "f42_dual_scale_sparse"
                    , "f43_relation_type_bias"
                    , "f44_second_order_context"
                    , "f45_temporal_risk_bias"
                    , "f46_multiscale_temporal_residual"
                    , "f47_multiscale_temporal_replace"
                    , "f48_ttc_dca_graph"
                    , "f49_ttc_dca_multiscale_replace"
                    , "f50_enhanced_kinematic_state"
                    , "f51_scaled_cosine_attention"
                    , "f52_pair_deformable_temporal"
                    , "f53_kdta_recent_exact"
                    , "f54_kdta_trend_event_recent"
                    , "f55_pair_temporal_soft_prior"
                    , "f56_motion_adaptive_key_smoothing"
                    , "f57_relation_grouped_vector_attention"
                    , "f58_compute_neutral_differential"
                },
                use_dual_graph=front_variant == "f5_dual_graph",
                top_k=int(getattr(self.cfg.MODEL.GRAPH, "TOP_K", 4)),
                frequency=float(self.cfg.DATA.FREQUENCY),
                heading_aligned_relations=(
                    front_variant == "f8_heading_aligned_relation"
                ),
                use_soft_risk_bias=(front_variant in {
                    "f9_soft_risk_bias", "f45_temporal_risk_bias"
                }),
                use_multiscale_relation=(front_variant in {
                    "f10_multiscale_relation", "f11_motion_auxiliary",
                    "f12_focal_conditioned"
                    , "f13_dynamic_routing", "f14_grouped_attention",
                    "f15_edge_iteration", "f16_temporal_social_cross",
                    "f17_sparsemax_attention", "f18_relation_qk"
                    , "f19_entmax15_attention"
                    , "f20_node_structure_encoding", "f21_agent_proxy_attention"
                    , "f22_temporal_cross_entmax15", "f23_dynamic_temporal_cross"
                    , "f24_temporal_dropkey", "f25_temporal_relative_bias"
                    , "f26_adaptive_dual_social"
                    , "f27_density_adaptive_social"
                    , "f28_temporal_landmark4", "f29_temporal_landmark2"
                    , "f30_graph_additive_attention"
                    , "f31_cascaded_group_attention"
                    , "f32_grouped_query_attention"
                    , "f33_multi_query_attention"
                    , "f34_identity_value_attention"
                    , "f35_temporal_half_pool"
                    , "f36_fixed_half_projection"
                    , "f37_lowdim_qk_attention"
                    , "f38_recent_exact_early_pool"
                    , "f39_motion_event_sampling"
                    , "f40_motion_token_squeeze"
                    , "f41_edge_list_attention"
                    , "f42_dual_scale_sparse"
                    , "f43_relation_type_bias"
                    , "f44_second_order_context"
                    , "f45_temporal_risk_bias"
                    , "f46_multiscale_temporal_residual"
                    , "f47_multiscale_temporal_replace"
                    , "f48_ttc_dca_graph"
                    , "f49_ttc_dca_multiscale_replace"
                    , "f50_enhanced_kinematic_state"
                    , "f51_scaled_cosine_attention"
                    , "f52_pair_deformable_temporal"
                    , "f53_kdta_recent_exact"
                    , "f54_kdta_trend_event_recent"
                    , "f55_pair_temporal_soft_prior"
                    , "f56_motion_adaptive_key_smoothing"
                    , "f57_relation_grouped_vector_attention"
                    , "f58_compute_neutral_differential"
                }),
                use_focal_conditioning=(
                    front_variant == "f12_focal_conditioned"
                ),
                use_dynamic_routing=(front_variant in {
                    "f13_dynamic_routing", "f23_dynamic_temporal_cross"
                }),
                use_grouped_attention=(front_variant == "f14_grouped_attention"),
                use_edge_iteration=(front_variant == "f15_edge_iteration"),
                use_temporal_social_cross=(front_variant in {
                    "f16_temporal_social_cross", "f22_temporal_cross_entmax15",
                    "f23_dynamic_temporal_cross"
                    , "f24_temporal_dropkey", "f25_temporal_relative_bias"
                    , "f28_temporal_landmark4", "f29_temporal_landmark2"
                    , "f30_graph_additive_attention"
                    , "f31_cascaded_group_attention"
                    , "f32_grouped_query_attention"
                    , "f33_multi_query_attention"
                    , "f34_identity_value_attention"
                    , "f35_temporal_half_pool"
                    , "f36_fixed_half_projection"
                    , "f37_lowdim_qk_attention"
                    , "f38_recent_exact_early_pool"
                    , "f39_motion_event_sampling"
                    , "f40_motion_token_squeeze"
                    , "f41_edge_list_attention"
                    , "f42_dual_scale_sparse"
                    , "f43_relation_type_bias"
                    , "f44_second_order_context"
                    , "f45_temporal_risk_bias"
                    , "f46_multiscale_temporal_residual"
                    , "f47_multiscale_temporal_replace"
                    , "f48_ttc_dca_graph"
                    , "f49_ttc_dca_multiscale_replace"
                    , "f50_enhanced_kinematic_state"
                    , "f51_scaled_cosine_attention"
                    , "f52_pair_deformable_temporal"
                    , "f53_kdta_recent_exact"
                    , "f54_kdta_trend_event_recent"
                    , "f55_pair_temporal_soft_prior"
                    , "f56_motion_adaptive_key_smoothing"
                    , "f57_relation_grouped_vector_attention"
                    , "f58_compute_neutral_differential"
                }),
                use_sparsemax=(front_variant == "f17_sparsemax_attention"),
                use_relation_qk=(front_variant == "f18_relation_qk"),
                use_entmax15=(front_variant in {
                    "f19_entmax15_attention", "f22_temporal_cross_entmax15"
                }),
                use_node_structure=(
                    front_variant == "f20_node_structure_encoding"
                ),
                use_agent_proxies=(
                    front_variant == "f21_agent_proxy_attention"
                ),
                use_temporal_dropkey=(front_variant == "f24_temporal_dropkey"),
                use_temporal_relative_bias=(
                    front_variant == "f25_temporal_relative_bias"
                ),
                use_adaptive_dual_social=(
                    front_variant == "f26_adaptive_dual_social"
                ),
                use_density_adaptive_social=(
                    front_variant == "f27_density_adaptive_social"
                ),
                temporal_landmarks=(
                    4 if front_variant == "f28_temporal_landmark4"
                    else 2 if front_variant == "f29_temporal_landmark2"
                    else 0
                ),
                use_graph_additive_attention=(
                    front_variant == "f30_graph_additive_attention"
                ),
                use_cascaded_group_attention=(
                    front_variant == "f31_cascaded_group_attention"
                ),
                kv_heads=(
                    2 if front_variant == "f32_grouped_query_attention"
                    else 1 if front_variant == "f33_multi_query_attention"
                    else 0
                ),
                use_identity_value=(
                    front_variant == "f34_identity_value_attention"
                ),
                use_temporal_half_pool=(
                    front_variant == "f35_temporal_half_pool"
                ),
                use_fixed_half_projection=(
                    front_variant == "f36_fixed_half_projection"
                ),
                qk_head_dim=(
                    8 if front_variant == "f37_lowdim_qk_attention" else 0
                ),
                use_recent_exact_early_pool=(
                    front_variant == "f38_recent_exact_early_pool"
                ),
                use_motion_event_sampling=(
                    front_variant == "f39_motion_event_sampling"
                ),
                use_motion_token_squeeze=(
                    front_variant == "f40_motion_token_squeeze"
                ),
                use_edge_list_attention=(
                    front_variant == "f41_edge_list_attention"
                ),
                use_dual_scale_sparse=(
                    front_variant == "f42_dual_scale_sparse"
                ),
                use_relation_type_bias=(
                    front_variant == "f43_relation_type_bias"
                ),
                use_second_order_context=(
                    front_variant == "f44_second_order_context"
                ),
                use_multiscale_temporal=(front_variant in {
                    "f46_multiscale_temporal_residual",
                    "f47_multiscale_temporal_replace",
                    "f49_ttc_dca_multiscale_replace",
                }),
                replace_temporal_encoder=(
                    front_variant in {
                        "f47_multiscale_temporal_replace",
                        "f49_ttc_dca_multiscale_replace",
                    }
                ),
                use_ttc_dca_graph=(front_variant in {
                    "f48_ttc_dca_graph", "f49_ttc_dca_multiscale_replace"
                }),
                use_scaled_cosine_attention=(
                    front_variant == "f51_scaled_cosine_attention"
                ),
                use_pair_deformable_temporal=(
                    front_variant in {
                        "f52_pair_deformable_temporal", "f53_kdta_recent_exact",
                        "f54_kdta_trend_event_recent",
                    }
                ),
                kdta_recent_exact=(front_variant in {
                    "f53_kdta_recent_exact", "f54_kdta_trend_event_recent"
                }),
                kdta_include_trend=(
                    front_variant == "f54_kdta_trend_event_recent"
                ),
                use_pair_temporal_soft_prior=(
                    front_variant == "f55_pair_temporal_soft_prior"
                ),
                use_motion_adaptive_key_smoothing=(
                    front_variant == "f56_motion_adaptive_key_smoothing"
                ),
                use_relation_grouped_vector_attention=(
                    front_variant == "f57_relation_grouped_vector_attention"
                ),
                use_differential_attention=(
                    front_variant == "f58_compute_neutral_differential"
                ),
            )
        if front_variant == "f11_motion_auxiliary":
            self.motion_aux_head = nn.Linear(self.token_dim, 4)

    def self_state_tokens(self, past_loc: Tensor) -> Tensor:
        if self.robust_state is None:
            return super().self_state_tokens(past_loc)
        encoded = self.robust_state(
            past_loc, float(self.cfg.DATA.FREQUENCY)
        )
        return (
            encoded.permute(0, 2, 1, 3)
            .reshape(
                past_loc.shape[0],
                self.obs_len * self.num_pedestrians,
                self.token_dim,
            )
        )

    def forward(
        self,
        past_loc: Tensor,
        fut_loc: Optional[Tensor],
        unet_features,
        mode: str = "train",
        num_valid: Optional[Tensor] = None,
        graph_loc: Optional[Tensor] = None,
    ):
        encoder = self.agent_aware_trans_encoder
        if hasattr(encoder, "set_context"):
            encoder.set_context(past_loc, num_valid)
        return super().forward(
            past_loc,
            fut_loc,
            unet_features,
            mode=mode,
            num_valid=num_valid,
            graph_loc=graph_loc,
        )
