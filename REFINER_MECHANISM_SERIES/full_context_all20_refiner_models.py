"""Frozen EXP-139C proposal/scorer with full-context all-mode refinement."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
SOURCE_MODULE = ROOT.parent / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"
EXPERIMENTS = {"r2_frozen_full_context_all20"}


def _load_source_module():
    spec = importlib.util.spec_from_file_location("exp139_full_context_source", SOURCE_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source_module()


class FullContextAnchorRefiner(nn.Module):
    """Mode-specific future queries attend to history tokens and learned anchors."""

    def __init__(
        self, token_dim: int, hidden_dim: int, pred_len: int, anchor_count: int,
        heads: int = 4, max_offset: float = 0.55, max_obs_len: int = 16,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.max_offset = float(max_offset)
        self.time_queries = nn.Parameter(torch.empty(pred_len, hidden_dim))
        self.history_times = nn.Parameter(torch.empty(max_obs_len, hidden_dim))
        self.anchor_times = nn.Parameter(torch.empty(anchor_count, hidden_dim))
        nn.init.normal_(self.time_queries, std=0.02)
        nn.init.normal_(self.history_times, std=0.02)
        nn.init.normal_(self.anchor_times, std=0.02)
        self.history_projection = nn.Linear(token_dim, hidden_dim)
        self.coarse_projection = nn.Linear(2, hidden_dim)
        self.anchor_projection = nn.Linear(2, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.residual_gate = nn.Linear(hidden_dim, 1)
        self.residual_head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, 1.0)
        alpha = torch.arange(1, pred_len + 1, dtype=torch.float32) / float(pred_len)
        endpoint_envelope = torch.sin(math.pi * alpha)
        endpoint_envelope[-1] = 0.0
        self.register_buffer("endpoint_envelope", endpoint_envelope)

    def forward(
        self, mode: Tensor, coarse_relative: Tensor, anchors_relative: Tensor,
        encoded_history: Tensor,
    ) -> Tensor:
        obs_len = encoded_history.shape[-2]
        if obs_len > self.history_times.shape[0]:
            raise ValueError(f"Observed history length {obs_len} exceeds supported length")
        query = (
            mode[:, None] + self.time_queries[None]
            + self.coarse_projection(coarse_relative)
        )
        history_tokens = (
            self.history_projection(encoded_history)
            + self.history_times[None, :obs_len]
        )
        anchor_tokens = (
            mode[:, None] + self.anchor_times[None]
            + self.anchor_projection(anchors_relative)
        )
        memory = torch.cat((history_tokens, anchor_tokens), dim=1)
        attended, _ = self.cross_attention(query, memory, memory, need_weights=False)
        hidden = self.attention_norm(query + attended)
        hidden = self.output_norm(hidden + self.ffn(hidden))
        gate = torch.sigmoid(self.residual_gate(hidden))
        residual = self.max_offset * gate * torch.tanh(self.residual_head(hidden))
        return residual * self.endpoint_envelope[None, :, None]


class FrozenFullContextAll20Decoder(SOURCE.HierarchicalLearnedAnchorDecoder):
    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(
            token_dim=token_dim, pred_len=pred_len, num_modes=20,
            anchor_steps=(4, 8, pred_len), use_refiner=True, top_m=20,
            hidden_dim=64,
        )
        self.refiner = FullContextAnchorRefiner(
            token_dim=token_dim, hidden_dim=64, pred_len=pred_len,
            anchor_count=self.anchor_count,
        )

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.eval()
        self.refiner.train(True)

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
        scale = torch.nn.functional.softplus(self.proposal_scale(context)) + 0.25
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
        batch, agents, modes = mode.shape[:3]
        selected = torch.arange(modes, device=mode.device).view(1, 1, modes)
        selected = selected.expand(batch, agents, modes)
        self.last_selected_indices = selected.detach()
        self._training_target = None

        flat_mode = mode.reshape(-1, mode.shape[-1])
        flat_coarse = coarse_relative.reshape(-1, self.pred_len, 2)
        flat_anchors = relative_anchors.reshape(-1, self.anchor_count, 2)
        history = encoded_tokens[:, :, None].expand(
            batch, agents, modes, encoded_tokens.shape[2], encoded_tokens.shape[3]
        ).reshape(-1, encoded_tokens.shape[2], encoded_tokens.shape[3])
        refined_relative = flat_coarse + self.refiner(
            flat_mode, flat_coarse, flat_anchors, history
        )
        trajectories = refined_relative.reshape(
            batch, agents, modes, self.pred_len, 2
        ) + last
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown EXP-140 decoder: {name}")
    checkpoint_text = os.environ.get("EXP140_SOURCE_CHECKPOINT", "").strip()
    if not checkpoint_text:
        raise RuntimeError("EXP140_SOURCE_CHECKPOINT must point to EXP-139C best_selected.pth")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model.decoder = FrozenFullContextAll20Decoder(model.token_dim, model.pred_len)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    expected_missing = {
        key for key in model.state_dict() if key.startswith("decoder.refiner.")
    }
    if missing != expected_missing:
        raise RuntimeError(f"Unexpected missing keys: {sorted(missing)}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected source keys: {incompatible.unexpected_keys}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.refiner.parameters():
        parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.frozen_source = True
    cfg.MODEL.DECODER.NUM_MODES = 20

