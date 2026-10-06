"""Regime-aware Gated Delta additions for the locked EXP-173/C00 decoder.

The module is deliberately incremental:

* the C00 generator and native score path are retained;
* both generation and selection additions are exact identity maps at init;
* selection reads the *corrected* V2 anchors, so its score is not stale;
* inference inputs are observed history, encoded context, and candidates only;
* future targets are accepted only by the optional training-only auxiliary loss.

No experiment is launched by importing or testing this module.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
BASE_ADAPTER_PATH = (
    HERE.parent / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
)


def _load_base_adapter():
    spec = importlib.util.spec_from_file_location("rgdm_exp173_base", BASE_ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_adapter()


VARIANTS: Dict[str, Dict[str, bool]] = {
    # Selection ablation sequence from the supplied design document.
    "rgdm_b1_history_gate": dict(selection=True, memory_score=False, candidate_query=False,
                                  history_gate=True, generation=False),
    "rgdm_b2_memory": dict(selection=True, memory_score=True, candidate_query=False,
                            history_gate=False, generation=False),
    "rgdm_b3_candidate_query": dict(selection=True, memory_score=True, candidate_query=True,
                                     history_gate=False, generation=False),
    "rgdm_b4_full": dict(selection=True, memory_score=True, candidate_query=True,
                          history_gate=True, generation=False),
    # Generation and final joint routes are implemented now but should only be
    # trained after the selection-stage review stop.
    "rgdm_g4_generation": dict(selection=False, memory_score=False, candidate_query=False,
                                history_gate=False, generation=True),
    "rgdm_ab_joint": dict(selection=True, memory_score=True, candidate_query=True,
                           history_gate=True, generation=True),
}


def observation_motion_evidence(past_loc: Tensor, eps: float = 1.0e-6) -> Tensor:
    """Observation-only change evidence [vx,vy,ax,ay,speed,dspeed,sin_turn,1-cos_turn]."""
    if past_loc.ndim != 4 or past_loc.shape[-1] != 2:
        raise ValueError("past_loc must have shape [B,A,T,2]")
    if past_loc.shape[-2] < 2:
        raise ValueError("at least two observed positions are required")
    velocity = torch.cat(
        (torch.zeros_like(past_loc[..., :1, :]), past_loc[..., 1:, :] - past_loc[..., :-1, :]),
        dim=-2,
    )
    previous_velocity = torch.cat((velocity[..., :1, :], velocity[..., :-1, :]), dim=-2)
    acceleration = velocity - previous_velocity
    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    previous_speed = torch.linalg.vector_norm(previous_velocity, dim=-1, keepdim=True)
    denominator = (speed * previous_speed).clamp_min(eps)
    cosine = (velocity * previous_velocity).sum(dim=-1, keepdim=True) / denominator
    sine = (
        velocity[..., 0:1] * previous_velocity[..., 1:2]
        - velocity[..., 1:2] * previous_velocity[..., 0:1]
    ) / denominator
    both_moving = (speed * previous_speed) > eps
    cosine = torch.where(both_moving, cosine.clamp(-1.0, 1.0), torch.ones_like(cosine))
    sine = torch.where(both_moving, sine.clamp(-1.0, 1.0), torch.zeros_like(sine))
    return torch.cat(
        (velocity, acceleration, speed, speed - previous_speed, sine, 1.0 - cosine), dim=-1
    )


def gated_delta_step(state: Tensor, key: Tensor, value: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """One exact Gated Delta update.

    Implements S_t = alpha S_(t-1) (I - beta k k^T) + beta v k^T
    without materialising the identity matrix. State is [..., value_dim, key_dim].
    """
    old_value = torch.einsum("...vd,...d->...v", state, key)
    replacement = value - alpha[..., None] * old_value
    return (
        alpha[..., None, None] * state
        + beta[..., None, None] * torch.einsum("...v,...d->...vd", replacement, key)
    )


class _CausalDepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(channels, channels, self.kernel_size, groups=channels, bias=True)

    def forward(self, values: Tensor) -> Tensor:
        # values: [B*A,T,C]. Left padding makes the operation strictly causal.
        values = F.pad(values.transpose(1, 2), (self.kernel_size - 1, 0))
        return self.conv(values).transpose(1, 2)


class RegimeAwareGatedDeltaMemory(nn.Module):
    """Small multi-head Gated Delta state over encoded observation tokens."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 32,
        heads: int = 4,
        key_dim: int = 8,
        value_dim: int = 8,
    ) -> None:
        super().__init__()
        if hidden_dim != heads * value_dim:
            raise ValueError("hidden_dim must equal heads * value_dim")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        source_dim = self.token_dim + 8
        self.source_norm = nn.LayerNorm(source_dim)
        self.q_projection = nn.Linear(source_dim, heads * key_dim)
        self.k_projection = nn.Linear(source_dim, heads * key_dim)
        self.v_projection = nn.Linear(source_dim, heads * value_dim)
        self.q_conv = _CausalDepthwiseConv(heads * key_dim)
        self.k_conv = _CausalDepthwiseConv(heads * key_dim)
        self.v_conv = _CausalDepthwiseConv(heads * value_dim)
        self.alpha_beta = nn.Linear(source_dim, 2 * heads)
        nn.init.zeros_(self.alpha_beta.weight)
        alpha0, beta0 = 0.98, 0.10
        with torch.no_grad():
            self.alpha_beta.bias[:heads].fill_(math.log(alpha0 / (1.0 - alpha0)))
            self.alpha_beta.bias[heads:].fill_(math.log(beta0 / (1.0 - beta0)))
        self.output = nn.Linear(heads * value_dim, hidden_dim)

    def _reshape_key(self, values: Tensor, batch: int, agents: int, steps: int) -> Tensor:
        return values.reshape(batch, agents, steps, self.heads, self.key_dim)

    def _reshape_value(self, values: Tensor, batch: int, agents: int, steps: int) -> Tensor:
        return values.reshape(batch, agents, steps, self.heads, self.value_dim)

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        if encoded_tokens.ndim != 4:
            raise ValueError("encoded_tokens must have shape [B,A,T,D]")
        if encoded_tokens.shape[:-1] != past_loc.shape[:-1]:
            raise ValueError("encoded_tokens and past_loc batch/agent/time axes must match")
        if encoded_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"expected token dim {self.token_dim}, got {encoded_tokens.shape[-1]}")
        batch, agents, steps, _ = encoded_tokens.shape
        evidence = observation_motion_evidence(past_loc)
        source = self.source_norm(torch.cat((encoded_tokens, evidence), dim=-1))
        flat = source.reshape(batch * agents, steps, -1)
        query = self._reshape_key(
            F.silu(self.q_conv(self.q_projection(flat))), batch, agents, steps
        )
        key = self._reshape_key(
            F.silu(self.k_conv(self.k_projection(flat))), batch, agents, steps
        )
        value = self._reshape_value(
            F.silu(self.v_conv(self.v_projection(flat))), batch, agents, steps
        )
        query = F.normalize(query, dim=-1, eps=1.0e-6)
        key = F.normalize(key, dim=-1, eps=1.0e-6)
        gates = torch.sigmoid(self.alpha_beta(source)).reshape(batch, agents, steps, 2, self.heads)
        alpha, beta = gates[..., 0, :], gates[..., 1, :]
        state = encoded_tokens.new_zeros(batch, agents, self.heads, self.value_dim, self.key_dim)
        outputs = []
        for time_index in range(steps):
            state = gated_delta_step(
                state, key[:, :, time_index], value[:, :, time_index],
                alpha[:, :, time_index], beta[:, :, time_index],
            )
            read = torch.einsum("bahvd,bahd->bahv", state, query[:, :, time_index])
            outputs.append(read.reshape(batch, agents, self.hidden_dim))
        sequence = self.output(torch.stack(outputs, dim=-2))
        diagnostics = {
            "alpha": alpha,
            "beta": beta,
            "change_probability": (1.0 - alpha).mean(dim=(-1, -2)),
        }
        return sequence, state, diagnostics


class GenerationSideRGDM(nn.Module):
    """Observation-conditioned residual update to the existing decoder context."""

    def __init__(self, token_dim: int, context_dim: int = 64, memory_dim: int = 32) -> None:
        super().__init__()
        self.memory = RegimeAwareGatedDeltaMemory(token_dim, hidden_dim=memory_dim)
        self.residual = nn.Sequential(
            nn.LayerNorm(context_dim + memory_dim),
            nn.Linear(context_dim + memory_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.lambda_gate = nn.Linear(context_dim + memory_dim, 1)
        self.fusion_strength = nn.Parameter(torch.zeros(()))

    def forward(self, context: Tensor, encoded_tokens: Tensor, past_loc: Tensor):
        sequence, _, diagnostics = self.memory(encoded_tokens, past_loc)
        joint = torch.cat((context, sequence[:, :, -1]), dim=-1)
        local_lambda = torch.sigmoid(self.lambda_gate(joint))
        # Exact identity at init with a nonzero derivative in both directions.
        # The residual network absorbs the sign; tanh prevents an unbounded edit.
        strength = torch.tanh(self.fusion_strength)
        updated = context + strength * local_lambda * self.residual(joint)
        diagnostics = dict(diagnostics, fusion_strength=strength, local_lambda=local_lambda.squeeze(-1))
        return updated, diagnostics


class SelectionSideRGDM(nn.Module):
    """Candidate-conditioned memory readout and change-aware native-logit calibration."""

    def __init__(self, token_dim: int, mode_dim: int = 64, anchor_count: int = 3,
                 hidden_dim: int = 64, memory_dim: int = 32) -> None:
        super().__init__()
        self.memory = RegimeAwareGatedDeltaMemory(token_dim, hidden_dim=memory_dim)
        candidate_input = mode_dim + anchor_count * 2
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(candidate_input), nn.Linear(candidate_input, hidden_dim), nn.GELU(),
        )
        self.context_projection = nn.Linear(token_dim, hidden_dim)
        self.candidate_query = nn.Linear(hidden_dim, self.memory.heads * self.memory.key_dim)
        self.shared_query = nn.Linear(memory_dim, self.memory.heads * self.memory.key_dim)
        self.memory_projection = nn.Linear(self.memory.heads * self.memory.value_dim, hidden_dim)
        score_dim = hidden_dim * 4 + 1
        self.score_residual = nn.Sequential(
            nn.LayerNorm(score_dim), nn.Linear(score_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score_residual[-1].weight)
        nn.init.zeros_(self.score_residual[-1].bias)
        self.history_calibration_strength = nn.Parameter(torch.zeros(()))

    def _read(self, state: Tensor, query: Tensor) -> Tensor:
        query = F.normalize(query, dim=-1, eps=1.0e-6)
        read = torch.einsum("bahvd,bamhd->bamhv", state, query)
        return self.memory_projection(read.flatten(-2))

    def forward(
        self,
        native_logits: Tensor,
        mode: Tensor,
        corrected_anchors: Tensor,
        encoded_tokens: Tensor,
        past_loc: Tensor,
        *,
        memory_score: bool,
        candidate_query: bool,
        history_gate: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        sequence, state, diagnostics = self.memory(encoded_tokens, past_loc)
        candidate = self.candidate_encoder(torch.cat((mode, corrected_anchors.flatten(-2)), dim=-1))
        batch, agents, modes, _ = candidate.shape
        if candidate_query:
            query = self.candidate_query(candidate).reshape(
                batch, agents, modes, self.memory.heads, self.memory.key_dim
            )
        else:
            shared = self.shared_query(sequence[:, :, -1]).reshape(
                batch, agents, 1, self.memory.heads, self.memory.key_dim
            )
            query = shared.expand(-1, -1, modes, -1, -1)
        memory = self._read(state, query)
        context = self.context_projection(encoded_tokens.mean(dim=-2))[:, :, None].expand_as(candidate)
        change = diagnostics["change_probability"]
        score_features = torch.cat(
            (candidate, memory, candidate * memory, context, change[:, :, None, None].expand(-1, -1, modes, 1)),
            dim=-1,
        )
        residual = self.score_residual(score_features).squeeze(-1)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        if not memory_score:
            residual = torch.zeros_like(residual)
        centered = native_logits - native_logits.mean(dim=-1, keepdim=True)
        if history_gate:
            calibration = 0.90 * torch.tanh(self.history_calibration_strength)
            confidence = 1.0 - calibration * change
            # Written as an additive correction so calibration==0 preserves
            # the native tensor without a subtract/reconstruct round-off.
            logits = native_logits - calibration * change[:, :, None] * centered
        else:
            calibration = native_logits.new_zeros(())
            confidence = native_logits.new_ones(native_logits.shape[:2])
            logits = native_logits
        logits = logits + residual
        diagnostics = dict(
            diagnostics,
            history_confidence=confidence,
            history_calibration_strength=calibration,
            score_residual=residual,
        )
        return logits, diagnostics


class RGDMSupplementDecoder(BASE.JointTrainableV2Decoder):
    """C00 decoder plus independently switchable RGDM-G and RGDM-S branches."""

    def __init__(self, token_dim: int, pred_len: int, variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown RGDM variant: {variant}")
        super().__init__(token_dim, pred_len)
        self.rgdm_variant = variant
        self.rgdm_spec = dict(VARIANTS[variant])
        self.generation_rgdm = GenerationSideRGDM(token_dim)
        self.selection_rgdm = SelectionSideRGDM(token_dim, anchor_count=self.anchor_count)
        self.gate_loss_weight = float(os.environ.get("RGDM_GATE_LOSS_WEIGHT", "0"))
        self.rank_loss_weight = float(os.environ.get("RGDM_RANK_LOSS_WEIGHT", "0"))
        self.rank_margin = float(os.environ.get("RGDM_RANK_MARGIN", "0.2"))
        self.last_native_logits: Tensor | None = None
        self.last_rgdm_diagnostics: Dict[str, Dict[str, Tensor]] = {}
        self._last_past: Tensor | None = None
        self._last_change_probability: Tensor | None = None

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.eval()
        if self.rgdm_spec["generation"]:
            self.generation_rgdm.train(True)
        if self.rgdm_spec["selection"]:
            self.selection_rgdm.train(True)

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context_input = torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1)
        context = self.context(context_input)
        diagnostics: Dict[str, Dict[str, Tensor]] = {}
        if self.rgdm_spec["generation"]:
            context, diagnostics["generation"] = self.generation_rgdm(context, encoded_tokens, past_loc)
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        scale = F.softplus(self.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(
            *mode.shape[:-1], self.anchor_count, 2
        )
        source_anchors = scale[:, :, None, None] * (self.anchor_templates[None, None] + delta)
        source_relative = torch.einsum("tp,bakpc->baktc", self.interpolation, source_anchors)
        last = past_loc[:, :, None, -1:, :]
        coarse = last + source_relative
        native_logits = self.score_head(
            torch.cat((mode, source_anchors.flatten(-2)), dim=-1)
        ).squeeze(-1)

        correction = self.refiner(mode, source_anchors)
        corrected_anchors = source_anchors + correction
        corrected_relative = torch.einsum("tp,bakpc->baktc", self.interpolation, corrected_anchors)
        trajectories = last + corrected_relative
        if self.rgdm_spec["selection"]:
            logits, diagnostics["selection"] = self.selection_rgdm(
                native_logits, mode, corrected_anchors, encoded_tokens, past_loc,
                memory_score=self.rgdm_spec["memory_score"],
                candidate_query=self.rgdm_spec["candidate_query"],
                history_gate=self.rgdm_spec["history_gate"],
            )
            self._last_change_probability = diagnostics["selection"]["change_probability"]
        else:
            logits = native_logits
            self._last_change_probability = diagnostics["generation"]["change_probability"]

        batch, agents, modes = logits.shape
        selected = torch.arange(modes, device=logits.device).view(1, 1, modes)
        self.last_selected_indices = selected.expand(batch, agents, modes).detach()
        self.last_coarse_trajectories = coarse.detach()
        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + source_anchors).detach()
        self.last_corrected_anchors = (past_loc[:, :, None, None, -1, :] + corrected_anchors).detach()
        self.last_native_logits = native_logits.detach()
        self.last_rgdm_diagnostics = {
            branch: {key: value.detach() for key, value in values.items()}
            for branch, values in diagnostics.items()
        }
        self._last_past = past_loc.detach()
        self._training_target = None
        return trajectories, logits

    @staticmethod
    def _future_change_target(past: Tensor, target: Tensor, eps: float = 1.0e-6) -> Tensor:
        observed = past[..., -1, :] - past[..., -2, :]
        future = (target[..., -1, :] - past[..., -1, :]) / float(target.shape[-2])
        observed_speed = torch.linalg.vector_norm(observed, dim=-1)
        future_speed = torch.linalg.vector_norm(future, dim=-1)
        denominator = (observed_speed * future_speed).clamp_min(eps)
        cosine = (observed * future).sum(dim=-1) / denominator
        moving = (observed_speed * future_speed) > eps
        cosine = torch.where(moving, cosine.clamp(-1.0, 1.0), torch.ones_like(cosine))
        turn = ((1.0 - cosine) / 2.0).clamp(0.0, 1.0)
        log_speed_ratio = torch.abs(torch.log((future_speed + 0.05) / (observed_speed + 0.05)))
        speed_change = (log_speed_ratio / math.log(2.0)).clamp(0.0, 1.0)
        return torch.maximum(turn, speed_change).detach()

    def auxiliary_training_loss(self, predictions: Tensor, target: Tensor, logits: Tensor) -> Tensor:
        """Optional train-only gate/rank losses; defaults are both zero."""
        total = logits.new_zeros(())
        if self.gate_loss_weight > 0.0:
            if self._last_past is None or self._last_change_probability is None:
                raise RuntimeError("RGDM forward must precede its auxiliary loss")
            change_target = self._future_change_target(self._last_past[:, :1], target)
            change_probability = self._last_change_probability[:, :1].clamp(1.0e-5, 1.0 - 1.0e-5)
            total = total + self.gate_loss_weight * F.binary_cross_entropy(
                change_probability, change_target
            )
        if self.rank_loss_weight > 0.0:
            distances = torch.linalg.vector_norm(predictions - target.unsqueeze(-3), dim=-1)
            quality = distances.mean(dim=-1) + 0.5 * distances[..., -1]
            winner = quality.detach().argmin(dim=-1)
            winner_logit = logits.gather(-1, winner[..., None]).squeeze(-1)
            negative = logits.masked_fill(
                F.one_hot(winner, num_classes=logits.shape[-1]).bool(), float("-inf")
            ).max(dim=-1).values
            rank = F.relu(self.rank_margin - winner_logit + negative).mean()
            total = total + self.rank_loss_weight * rank
        return total


def replace_decoder(model: nn.Module, cfg, name: str) -> None:
    """Install RGDM on a full C00 checkpoint and freeze the locked baseline."""
    if name not in VARIANTS:
        raise ValueError(f"unknown RGDM experiment: {name}")
    checkpoint_text = os.environ.get("RGDM_SOURCE_CHECKPOINT", "").strip()
    if not checkpoint_text:
        raise RuntimeError("RGDM_SOURCE_CHECKPOINT must point to the locked C00 best_selected.pth")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model.decoder = RGDMSupplementDecoder(model.token_dim, model.pred_len, name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_prefixes = ("decoder.generation_rgdm.", "decoder.selection_rgdm.")
    unexpected_missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)]
    if unexpected_missing:
        raise RuntimeError(f"unexpected C00 missing keys: {unexpected_missing}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected C00 keys: {incompatible.unexpected_keys}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    spec = VARIANTS[name]
    if spec["generation"]:
        for parameter in model.decoder.generation_rgdm.parameters():
            parameter.requires_grad_(True)
    if spec["selection"]:
        selection = model.decoder.selection_rgdm
        # Do not count dormant ablation components as trainable capacity.
        for parameter in selection.memory.parameters():
            parameter.requires_grad_(True)
        if spec["history_gate"]:
            selection.history_calibration_strength.requires_grad_(True)
        if spec["memory_score"]:
            for component in (
                selection.candidate_encoder,
                selection.context_projection,
                selection.memory_projection,
                selection.score_residual,
            ):
                for parameter in component.parameters():
                    parameter.requires_grad_(True)
            query_component = (
                selection.candidate_query if spec["candidate_query"] else selection.shared_query
            )
            for parameter in query_component.parameters():
                parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint.resolve())
    model.decoder.training_policy_name = "frozen_c00_incremental_rgdm"
    model.decoder.test_accessed = False
    cfg.MODEL.DECODER.NUM_MODES = 20
