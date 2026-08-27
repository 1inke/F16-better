"""CV-free hierarchical proposal/select/refine decoders for EXP-139.

All reference points are learned relative to the last observed position.  The
last velocity is supplied only as a feature; it is never multiplied by the
prediction horizon and never used as a candidate center.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EXPERIMENTS = {
    "h1_endpoint_proposal",
    "h2_endpoint_select_refine",
    "h3_anchor_proposal",
    "h4_anchor_select_refine",
}


def endpoint_template_initialization(num_modes: int) -> Tensor:
    """Diverse trainable initialization, relative to the last position."""
    points = [(x, 0.0) for x in (0.10, 0.50, 1.00, 1.80, 2.80)]
    points += [
        (x, y)
        for x in (0.50, 1.20, 2.00)
        for y in (-1.00, -0.45, 0.45, 1.00)
    ]
    points += [(0.15, -0.80), (0.15, 0.80), (-0.20, 0.0)]
    if num_modes > len(points):
        generator = torch.Generator().manual_seed(139)
        extra = torch.randn(num_modes - len(points), 2, generator=generator) * 0.30
        points.extend((float(x), float(y)) for x, y in extra)
    return torch.tensor(points[:num_modes], dtype=torch.float32)


def interpolation_weights(pred_len: int, anchor_steps: tuple[int, ...]) -> Tensor:
    """Linear interpolation matrix from relative anchors to every future step."""
    weights = torch.zeros(pred_len, len(anchor_steps), dtype=torch.float32)
    previous_step = 0
    for segment, step in enumerate(anchor_steps):
        for time in range(previous_step + 1, step + 1):
            alpha = float(time - previous_step) / float(step - previous_step)
            weights[time - 1, segment] = alpha
            if segment:
                weights[time - 1, segment - 1] = 1.0 - alpha
        previous_step = step
    if anchor_steps[-1] < pred_len:
        weights[anchor_steps[-1]:, -1] = 1.0
    return weights


class AnchorConditionedRefiner(nn.Module):
    """Twelve time queries cross-attend to selected endpoint/anchor tokens."""

    def __init__(
        self, hidden_dim: int, pred_len: int, anchor_count: int,
        heads: int = 4, max_offset: float = 0.55,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.anchor_count = int(anchor_count)
        self.max_offset = float(max_offset)
        self.time_queries = nn.Parameter(torch.empty(pred_len, hidden_dim))
        self.anchor_times = nn.Parameter(torch.empty(anchor_count, hidden_dim))
        nn.init.normal_(self.time_queries, std=0.02)
        nn.init.normal_(self.anchor_times, std=0.02)
        self.coarse_projection = nn.Linear(2, hidden_dim)
        self.anchor_projection = nn.Linear(2, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.residual_head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        alpha = torch.arange(1, pred_len + 1, dtype=torch.float32) / float(pred_len)
        envelope = torch.sin(math.pi * alpha)
        envelope[-1] = 0.0
        self.register_buffer("endpoint_envelope", envelope)

    def forward(
        self, mode: Tensor, coarse_relative: Tensor, anchors_relative: Tensor
    ) -> Tensor:
        # Inputs are already gathered to [B*A*M,...].
        query = (
            mode[:, None] + self.time_queries[None]
            + self.coarse_projection(coarse_relative)
        )
        anchor_tokens = (
            mode[:, None] + self.anchor_times[None]
            + self.anchor_projection(anchors_relative)
        )
        attended, _ = self.cross_attention(
            query, anchor_tokens, anchor_tokens, need_weights=False
        )
        hidden = self.attention_norm(query + attended)
        hidden = self.output_norm(hidden + self.ffn(hidden))
        residual = self.max_offset * torch.tanh(self.residual_head(hidden))
        return residual * self.endpoint_envelope[None, :, None]


class HierarchicalLearnedAnchorDecoder(nn.Module):
    """Learned endpoints/anchors followed by optional Top-M refinement."""

    def __init__(
        self,
        token_dim: int,
        pred_len: int,
        num_modes: int = 20,
        anchor_steps: tuple[int, ...] = (12,),
        use_refiner: bool = False,
        top_m: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.pred_len = int(pred_len)
        self.anchor_steps = tuple(int(step) for step in anchor_steps)
        self.anchor_count = len(self.anchor_steps)
        self.use_refiner = bool(use_refiner)
        self.top_m = min(int(top_m), self.num_modes)
        if self.anchor_steps[-1] != self.pred_len:
            raise ValueError("The final learned anchor must be the endpoint")

        context_dim = token_dim * 2 + 7
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.mode_embedding = nn.Parameter(torch.empty(num_modes, hidden_dim))
        nn.init.normal_(self.mode_embedding, std=0.02)
        self.mode_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.proposal_scale = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.proposal_scale.weight)
        nn.init.constant_(self.proposal_scale.bias, math.log(math.exp(0.75) - 1.0))
        self.proposal_delta = nn.Linear(hidden_dim, self.anchor_count * 2)
        nn.init.zeros_(self.proposal_delta.weight)
        nn.init.zeros_(self.proposal_delta.bias)

        endpoints = endpoint_template_initialization(num_modes)
        fractions = torch.tensor(self.anchor_steps, dtype=torch.float32) / float(pred_len)
        template = fractions[:, None, None] * endpoints[None]
        self.anchor_templates = nn.Parameter(template.permute(1, 0, 2).contiguous())
        self.register_buffer(
            "interpolation",
            interpolation_weights(pred_len, self.anchor_steps),
        )
        score_input = hidden_dim + self.anchor_count * 2
        self.score_head = nn.Sequential(
            nn.LayerNorm(score_input), nn.Linear(score_input, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)
        self.refiner = (
            AnchorConditionedRefiner(
                hidden_dim, pred_len, self.anchor_count, heads=4
            ) if self.use_refiner else None
        )
        self.last_coarse_trajectories: Tensor | None = None
        self.last_selected_indices: Tensor | None = None
        self.last_anchor_proposals: Tensor | None = None
        self._training_target: Tensor | None = None

    def set_training_target(self, target: Tensor | None) -> None:
        """Teacher-forced selection only; target never enters proposal features."""
        self._training_target = target

    @staticmethod
    def motion_features(past_loc: Tensor) -> Tensor:
        short = past_loc[:, :, -1] - past_loc[:, :, -2]
        lag = min(3, past_loc.shape[2] - 1)
        long = (past_loc[:, :, -1] - past_loc[:, :, -1 - lag]) / float(lag)
        acceleration = short - long
        speed = torch.linalg.vector_norm(short, dim=-1, keepdim=True)
        return torch.cat((short, long, acceleration, speed), dim=-1)

    @staticmethod
    def gather_modes(values: Tensor, indices: Tensor) -> Tensor:
        suffix = values.shape[3:]
        view = indices.shape + (1,) * len(suffix)
        expand = indices.reshape(view).expand(*indices.shape, *suffix)
        return torch.gather(values, 2, expand)

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context_input = torch.cat(
            (recent, pooled, self.motion_features(past_loc)), dim=-1
        )
        context = self.context(context_input)
        mode = self.mode_mlp(
            context[:, :, None] + self.mode_embedding[None, None]
        )
        scale = F.softplus(self.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(
            *mode.shape[:-1], self.anchor_count, 2
        )
        relative_anchors = scale[:, :, None, None] * (
            self.anchor_templates[None, None] + delta
        )
        coarse_relative = torch.einsum(
            "tp,bakpc->baktc", self.interpolation, relative_anchors
        )
        last = past_loc[:, :, None, -1:, :]
        coarse = last + coarse_relative
        logits = self.score_head(
            torch.cat((mode, relative_anchors.flatten(-2)), dim=-1)
        ).squeeze(-1)

        self.last_anchor_proposals = (
            past_loc[:, :, None, None, -1, :] + relative_anchors
        ).detach()
        self.last_coarse_trajectories = coarse.detach()
        self.last_selected_indices = None
        if not self.use_refiner:
            return coarse, logits

        selection_scores = logits
        if self.training:
            uniform = torch.rand_like(selection_scores).clamp(1.0e-5, 1.0 - 1.0e-5)
            gumbel = -torch.log(-torch.log(uniform))
            selection_scores = selection_scores + 0.25 * gumbel
        selected = selection_scores.topk(self.top_m, dim=-1).indices
        if self.training and self._training_target is not None:
            target = self._training_target
            coarse_quality = torch.linalg.vector_norm(
                coarse - target[:, :, None], dim=-1
            ).mean(dim=-1)
            oracle = coarse_quality.argmin(dim=-1)
            already_selected = (selected == oracle[..., None]).any(dim=-1)
            replacement = torch.cat((selected[..., :-1], oracle[..., None]), dim=-1)
            selected = torch.where(
                already_selected[..., None], selected, replacement
            )
        self._training_target = None
        selected_mode = self.gather_modes(mode, selected)
        selected_coarse = self.gather_modes(coarse_relative, selected)
        selected_anchors = self.gather_modes(relative_anchors, selected)
        flat_mode = selected_mode.reshape(-1, mode.shape[-1])
        flat_coarse = selected_coarse.reshape(-1, self.pred_len, 2)
        flat_anchors = selected_anchors.reshape(-1, self.anchor_count, 2)
        refined_relative = flat_coarse + self.refiner(
            flat_mode, flat_coarse, flat_anchors
        )
        refined = refined_relative.reshape(
            *selected.shape, self.pred_len, 2
        ) + past_loc[:, :, None, -1:, :]
        selected_mask = torch.zeros_like(logits).scatter(
            -1, selected, torch.ones_like(selected, dtype=logits.dtype)
        )
        refined_full = torch.zeros_like(coarse).scatter(
            2,
            selected[..., None, None].expand(
                *selected.shape, self.pred_len, 2
            ),
            refined,
        )
        trajectories = (
            coarse * (1.0 - selected_mask[..., None, None]) + refined_full
        )
        self.last_selected_indices = selected.detach()
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    settings = {
        "h1_endpoint_proposal": ((model.pred_len,), False),
        "h2_endpoint_select_refine": ((model.pred_len,), True),
        "h3_anchor_proposal": ((4, 8, model.pred_len), False),
        "h4_anchor_select_refine": ((4, 8, model.pred_len), True),
    }
    if name not in settings:
        raise ValueError(f"Unknown EXP-139 decoder: {name}")
    anchor_steps, use_refiner = settings[name]
    model.decoder = HierarchicalLearnedAnchorDecoder(
        token_dim=model.token_dim,
        pred_len=model.pred_len,
        num_modes=20,
        anchor_steps=anchor_steps,
        use_refiner=use_refiner,
        top_m=8,
        hidden_dim=64,
    )
    cfg.MODEL.DECODER.NUM_MODES = 20
