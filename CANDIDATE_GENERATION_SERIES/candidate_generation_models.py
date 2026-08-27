"""EXP-144 structural candidate-generation decoders built on EXP-143B."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
CONTROLLED_PATH = ROOT.parent / "CONTROLLED_ANCHOR_CORRECTION_SERIES" / "controlled_anchor_correction_models.py"
EXPERIMENTS = {
    "g1_set_coordinated_generation": (True, False),
    "g2_dynamic_progress_generation": (False, True),
    "g3_set_progress_generation": (True, True),
}


def _load_controlled():
    spec = importlib.util.spec_from_file_location("exp144_controlled_source", CONTROLLED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(CONTROLLED_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


CONTROLLED = _load_controlled()


class CandidateSetGenerator(nn.Module):
    """Coordinate all 20 modes before producing a second anchor proposal."""

    def __init__(self, hidden_dim: int = 64, anchor_count: int = 3) -> None:
        super().__init__()
        input_dim = hidden_dim + anchor_count * 2
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
        )
        self.set_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.anchor_head = nn.Linear(hidden_dim, anchor_count * 2)
        nn.init.zeros_(self.anchor_head.weight); nn.init.zeros_(self.anchor_head.bias)
        self.register_buffer("max_offsets", torch.tensor([0.12, 0.16, 0.25], dtype=torch.float32))

    def forward(self, mode: Tensor, anchors: Tensor) -> Tensor:
        shape = mode.shape
        features = torch.cat((mode, anchors.flatten(-2)), dim=-1)
        hidden = self.input_projection(features).reshape(-1, shape[-2], shape[-1])
        attended, _ = self.set_attention(hidden, hidden, hidden, need_weights=False)
        hidden = self.attention_norm(hidden + attended)
        hidden = self.output_norm(hidden + self.ffn(hidden))
        raw = self.anchor_head(hidden).reshape(*anchors.shape)
        return torch.tanh(raw) * self.max_offsets.view(1, 1, 1, -1, 1)


class DynamicProgressInterpolator(nn.Module):
    """Predict monotonic segment durations instead of fixed 4/8/12 timing."""

    def __init__(self, hidden_dim: int = 64, anchor_count: int = 3, pred_len: int = 12) -> None:
        super().__init__()
        self.anchor_count = int(anchor_count); self.pred_len = int(pred_len)
        input_dim = hidden_dim + anchor_count * 2
        self.duration_head = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, anchor_count),
        )
        nn.init.zeros_(self.duration_head[-1].weight)
        nn.init.zeros_(self.duration_head[-1].bias)
        times = torch.arange(1, pred_len + 1, dtype=torch.float32) / float(pred_len)
        self.register_buffer("query_times", times)

    def forward(self, mode: Tensor, anchors: Tensor) -> tuple[Tensor, Tensor]:
        features = torch.cat((mode, anchors.flatten(-2)), dim=-1)
        # Bounded logits prevent a segment from collapsing to nearly zero.
        duration_logits = 0.80 * torch.tanh(self.duration_head(features))
        durations = torch.softmax(duration_logits, dim=-1)
        starts = torch.cumsum(durations, dim=-1) - durations
        previous = torch.cat((torch.zeros_like(anchors[..., :1, :]), anchors[..., :-1, :]), dim=-2)
        delta = anchors - previous
        slopes = delta / durations[..., None].clamp_min(1.0e-4)
        time = self.query_times.view(*((1,) * (anchors.ndim - 2)), self.pred_len)
        elapsed = torch.minimum(
            torch.relu(time[..., :, None] - starts[..., None, :]),
            durations[..., None, :],
        )
        trajectory = torch.einsum("...tp,...pc->...tc", elapsed, slopes)
        knots = torch.cumsum(durations, dim=-1)
        return trajectory, knots


class StructuralCandidateDecoder(CONTROLLED.FrozenControlledAnchorDecoder):
    """Keep EXP-143B frozen and learn a coordinated structural second stage."""

    def __init__(self, token_dim: int, pred_len: int, use_set: bool, use_progress: bool) -> None:
        super().__init__(token_dim, pred_len)
        self.refiner.max_offsets.copy_(torch.tensor([0.20, 0.25, 0.50]))
        self.set_generator = CandidateSetGenerator(64, 3) if use_set else None
        self.progress_interpolator = DynamicProgressInterpolator(64, 3, pred_len) if use_progress else None
        self.use_set_generation = bool(use_set)
        self.use_dynamic_progress = bool(use_progress)
        self.last_progress_knots: Tensor | None = None

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.eval()
        if self.set_generator is not None:
            self.set_generator.train(True)
        if self.progress_interpolator is not None:
            self.progress_interpolator.train(True)

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]; pooled = encoded_tokens.mean(dim=2)
        context = self.context(torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1))
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        scale = torch.nn.functional.softplus(self.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(*mode.shape[:-1], 3, 2)
        source_anchors = scale[:, :, None, None] * (self.anchor_templates[None, None] + delta)
        base_correction = self.refiner(mode, source_anchors)
        base_anchors = source_anchors + base_correction
        final_anchors = base_anchors
        if self.set_generator is not None:
            # The locked protocol supervises/evaluates agent 0.  Avoid paying
            # for a 20-mode set generator on padded/context-only agents.
            focal_anchors = base_anchors[:, :1] + self.set_generator(
                mode[:, :1], base_anchors[:, :1]
            )
            final_anchors = torch.cat((focal_anchors, base_anchors[:, 1:]), dim=1)

        base_relative = torch.einsum("tp,bakpc->baktc", self.interpolation, base_anchors)
        fixed_relative = torch.einsum("tp,bakpc->baktc", self.interpolation, final_anchors)
        if self.progress_interpolator is None:
            final_relative = fixed_relative
            knots = torch.tensor([1 / 3, 2 / 3, 1.0], device=mode.device, dtype=mode.dtype)
            knots = knots.view(1, 1, 1, 3).expand(*mode.shape[:-1], 3)
        else:
            focal_relative, focal_knots = self.progress_interpolator(
                mode[:, :1], final_anchors[:, :1]
            )
            final_relative = torch.cat((focal_relative, fixed_relative[:, 1:]), dim=1)
            fixed_knots = torch.tensor(
                [1 / 3, 2 / 3, 1.0], device=mode.device, dtype=mode.dtype
            ).view(1, 1, 1, 3).expand(mode.shape[0], mode.shape[1] - 1, mode.shape[2], 3)
            knots = torch.cat((focal_knots, fixed_knots), dim=1)
        last = past_loc[:, :, None, -1:, :]
        trajectories = last + final_relative
        logits = self.score_head(torch.cat((mode, source_anchors.flatten(-2)), dim=-1)).squeeze(-1)

        batch, agents, modes = logits.shape
        selected = torch.arange(modes, device=logits.device).view(1, 1, modes)
        self.last_selected_indices = selected.expand(batch, agents, modes).detach()
        self.last_coarse_trajectories = (last + base_relative).detach()
        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + base_anchors).detach()
        self.last_corrected_anchors = (past_loc[:, :, None, None, -1, :] + final_anchors).detach()
        self.last_progress_knots = knots.detach()
        self._training_target = None
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown EXP-144 decoder: {name}")
    checkpoint_text = os.environ.get("EXP144_SOURCE_CHECKPOINT", "").strip()
    if not checkpoint_text:
        raise RuntimeError("EXP144_SOURCE_CHECKPOINT is required")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    use_set, use_progress = EXPERIMENTS[name]
    model.decoder = StructuralCandidateDecoder(model.token_dim, model.pred_len, use_set, use_progress)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected = {
        key for key in model.state_dict()
        if key.startswith("decoder.set_generator.") or key.startswith("decoder.progress_interpolator.")
    }
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefixes = []
    if use_set:
        prefixes.append("decoder.set_generator.")
    if use_progress:
        prefixes.append("decoder.progress_interpolator.")
    for parameter_name, parameter in model.named_parameters():
        if any(parameter_name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.structural_mechanism = name
    cfg.MODEL.DECODER.NUM_MODES = 20
