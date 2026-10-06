"""PIF1: scene-free conditional flow over motion-normalized physical intent.

The frozen F16/C00 representation conditions a small RealNVP-style flow.  The
flow generates twenty deterministic low-discrepancy intent samples and supplies
their exact density as the candidate score.  There is deliberately no separate
score head and no static-scene input.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
CONTROLLED_PATH = (
    HERE.parent / "CONTROLLED_ANCHOR_CORRECTION_SERIES"
    / "controlled_anchor_correction_models.py"
)
EXPERIMENT = "pif1_physical_intent_flow"
CONTROL_DIM = 6
ANCHOR_INDICES = (3, 7, 11)
SEGMENT_STEPS = 4.0


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROLLED = _load_module("pif_controlled_source", CONTROLLED_PATH)


def wrap_angle(value: Tensor) -> Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def reference_frame(past: Tensor, speed_floor: float) -> tuple[Tensor, Tensor, Tensor]:
    """Return forward, left and reference speed for [..., T, 2] histories."""
    short = past[..., -1, :] - past[..., -2, :]
    lag = min(3, past.shape[-2] - 1)
    long = (past[..., -1, :] - past[..., -1 - lag, :]) / float(lag)
    short_speed = torch.linalg.vector_norm(short, dim=-1, keepdim=True)
    long_speed = torch.linalg.vector_norm(long, dim=-1, keepdim=True)
    heading = torch.where(short_speed > float(speed_floor), short, long)
    heading_speed = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    fallback = torch.zeros_like(heading)
    fallback[..., 0] = 1.0
    forward = torch.where(
        heading_speed > 1.0e-8,
        heading / heading_speed.clamp_min(1.0e-8),
        fallback,
    )
    left = torch.stack((-forward[..., 1], forward[..., 0]), dim=-1)
    speed = torch.maximum(short_speed, long_speed).clamp_min(float(speed_floor))
    return forward, left, speed


def anchors_to_controls(
    anchors_relative: Tensor,
    past: Tensor,
    speed_floor: float,
) -> Tensor:
    """Encode three relative anchors as log-speed ratios and turn increments."""
    forward, left, speed = reference_frame(past, speed_floor)
    previous = torch.cat((torch.zeros_like(anchors_relative[..., :1, :]), anchors_relative[..., :-1, :]), dim=-2)
    segments = anchors_relative - previous
    local_x = (segments * forward.unsqueeze(-2)).sum(dim=-1)
    local_y = (segments * left.unsqueeze(-2)).sum(dim=-1)
    lengths = torch.linalg.vector_norm(segments, dim=-1).clamp_min(1.0e-6)
    base = (speed.squeeze(-1) * SEGMENT_STEPS).unsqueeze(-1).clamp_min(1.0e-6)
    log_ratio = torch.log(lengths / base)
    angles = torch.atan2(local_y, local_x)
    turns = torch.cat((angles[..., :1], wrap_angle(angles[..., 1:] - angles[..., :-1])), dim=-1)
    return torch.stack(
        (log_ratio[..., 0], turns[..., 0], log_ratio[..., 1], turns[..., 1], log_ratio[..., 2], turns[..., 2]),
        dim=-1,
    )


def controls_to_anchors(
    controls: Tensor,
    past: Tensor,
    speed_floor: float,
) -> Tensor:
    """Decode [..., K, 6] controls into [..., K, 3, 2] relative anchors."""
    forward, left, speed = reference_frame(past, speed_floor)
    log_ratio = controls[..., (0, 2, 4)]
    turn = controls[..., (1, 3, 5)]
    angle = torch.cumsum(turn, dim=-1)
    base = speed.squeeze(-1).unsqueeze(-1).unsqueeze(-1) * SEGMENT_STEPS
    length = base * torch.exp(log_ratio)
    local_x = length * torch.cos(angle)
    local_y = length * torch.sin(angle)
    world_segments = (
        local_x[..., None] * forward.unsqueeze(-2).unsqueeze(-2)
        + local_y[..., None] * left.unsqueeze(-2).unsqueeze(-2)
    )
    return torch.cumsum(world_segments, dim=-2)


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden_dim: int, mask: Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask.reshape(dim).float())
        self.net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _scale_shift(self, masked: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        raw_scale, shift = self.net(torch.cat((masked, context), dim=-1)).chunk(2, dim=-1)
        active = 1.0 - self.mask
        scale = 0.75 * torch.tanh(raw_scale) * active
        return scale, shift * active

    def to_base(self, value: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        masked = value * self.mask
        scale, shift = self._scale_shift(masked, context)
        base = masked + (1.0 - self.mask) * (value - shift) * torch.exp(-scale)
        return base, -scale.sum(dim=-1)

    def from_base(self, base: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        masked = base * self.mask
        scale, shift = self._scale_shift(masked, context)
        value = masked + (1.0 - self.mask) * (base * torch.exp(scale) + shift)
        return value, scale.sum(dim=-1)


class ConditionalIntentFlow(nn.Module):
    def __init__(self, context_dim: int = 64, hidden_dim: int = 96, layers: int = 6) -> None:
        super().__init__()
        masks = []
        for layer in range(int(layers)):
            offset = layer % 2
            masks.append(torch.tensor([(index + offset) % 2 for index in range(CONTROL_DIM)]))
        self.layers = nn.ModuleList(
            ConditionalAffineCoupling(CONTROL_DIM, context_dim, hidden_dim, mask)
            for mask in masks
        )

    @staticmethod
    def base_log_prob(value: Tensor) -> Tensor:
        return -0.5 * (value.square() + math.log(2.0 * math.pi)).sum(dim=-1)

    def log_prob(self, value: Tensor, context: Tensor) -> Tensor:
        base, log_det = value, torch.zeros(value.shape[:-1], device=value.device, dtype=value.dtype)
        for layer in self.layers:
            base, delta = layer.to_base(base, context)
            log_det = log_det + delta
        return self.base_log_prob(base) + log_det

    def sample(self, context: Tensor, latent: Tensor) -> tuple[Tensor, Tensor]:
        value = latent
        inverse_log_det = torch.zeros(latent.shape[:-1], device=latent.device, dtype=latent.dtype)
        for layer in reversed(self.layers):
            value, delta = layer.from_base(value, context)
            inverse_log_det = inverse_log_det + delta
        return value, self.base_log_prob(latent) - inverse_log_det


def fixed_latent_points(count: int = 20, dim: int = CONTROL_DIM, seed: int = 20260923) -> Tensor:
    if count < 2:
        raise ValueError("At least two candidate points are required")
    engine = torch.quasirandom.SobolEngine(dim, scramble=True, seed=int(seed))
    uniform = engine.draw(count - 1).clamp(1.0e-4, 1.0 - 1.0e-4)
    normal = math.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)
    return torch.cat((torch.zeros(1, dim), normal), dim=0)


class PhysicalIntentFlowDecoder(CONTROLLED.FrozenControlledAnchorDecoder):
    def __init__(self, token_dim: int, pred_len: int, stats: dict) -> None:
        super().__init__(token_dim, pred_len)
        if pred_len != 12:
            raise ValueError(f"PIF1 is preregistered for 12 future steps, got {pred_len}")
        mean = torch.tensor(stats["control_mean"], dtype=torch.float32)
        std = torch.tensor(stats["control_std"], dtype=torch.float32)
        if mean.shape != (CONTROL_DIM,) or std.shape != (CONTROL_DIM,) or torch.any(std <= 0):
            raise ValueError("Invalid physical-intent calibration statistics")
        self.flow = ConditionalIntentFlow(
            context_dim=64,
            hidden_dim=int(os.environ.get("PIF_HIDDEN_DIM", "96")),
            layers=int(os.environ.get("PIF_FLOW_LAYERS", "6")),
        )
        self.register_buffer("control_mean", mean)
        self.register_buffer("control_std", std)
        self.register_buffer("latent_points", fixed_latent_points())
        self.speed_floor = float(stats["speed_floor"])
        self.nll_weight = float(os.environ.get("PIF_NLL_WEIGHT", "0.10"))
        self.last_target_nll: Tensor | None = None
        self.last_generated_controls: Tensor | None = None
        self.last_flow_log_prob: Tensor | None = None

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.eval()
        self.flow.train(True)

    def auxiliary_training_loss(self, prediction: Tensor, target: Tensor, logits: Tensor) -> Tensor:
        del target, logits
        if self.last_target_nll is None:
            return prediction.sum() * 0.0
        return self.nll_weight * self.last_target_nll

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context = self.context(
            torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1)
        ).detach()
        batch, agents, _ = context.shape
        modes = self.latent_points.shape[0]
        expanded_context = context[:, :, None, :].expand(batch, agents, modes, -1)
        latent = self.latent_points.view(1, 1, modes, CONTROL_DIM).expand(batch, agents, -1, -1)
        standardized, logits = self.flow.sample(expanded_context, latent)
        controls = standardized * self.control_std + self.control_mean
        relative_anchors = controls_to_anchors(controls, past_loc, self.speed_floor)
        relative = torch.einsum("tp,bakpc->baktc", self.interpolation, relative_anchors)
        last = past_loc[:, :, None, -1:, :]
        trajectories = last + relative

        target = self._training_target
        self._training_target = None
        self.last_target_nll = None
        if target is not None:
            target_relative_anchors = target[:, :1, list(ANCHOR_INDICES), :] - past_loc[:, :1, -1:, :]
            target_controls = anchors_to_controls(target_relative_anchors, past_loc[:, :1], self.speed_floor)
            target_standardized = (target_controls - self.control_mean) / self.control_std
            target_log_prob = self.flow.log_prob(target_standardized, context[:, :1])
            self.last_target_nll = (-target_log_prob).mean() / float(CONTROL_DIM)

        selected = torch.arange(modes, device=logits.device).view(1, 1, modes)
        self.last_selected_indices = selected.expand(batch, agents, modes).detach()
        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + relative_anchors).detach()
        self.last_corrected_anchors = self.last_anchor_proposals
        self.last_coarse_trajectories = trajectories.detach()
        self.last_generated_controls = controls.detach()
        self.last_flow_log_prob = logits.detach()
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name != EXPERIMENT:
        raise ValueError(name)
    source = Path(os.environ.get("PIF_SOURCE_CHECKPOINT", ""))
    stats_path = Path(os.environ.get("PIF_STATS_JSON", ""))
    if not source.is_file() or not stats_path.is_file():
        raise FileNotFoundError((source, stats_path))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    model.decoder = PhysicalIntentFlowDecoder(model.token_dim, model.pred_len, stats)
    source_state = torch.load(source, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(source_state, strict=False)
    expected_missing = {
        key for key in model.state_dict()
        if key.startswith("decoder.flow.")
        or key in {
            "decoder.control_mean", "decoder.control_std", "decoder.latent_points"
        }
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch; missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.flow.parameters():
        parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(source)
    model.decoder.stats_path = str(stats_path)
    model.decoder.training_policy_name = "frozen_f16_c00_context_train_flow_only"
    cfg.MODEL.DECODER.NUM_MODES = 20
    # Candidate spread is learned by likelihood and winner/coverage objectives,
    # never by the explicit pairwise diversity hinge.
    cfg.MODEL.DECODER.DIVERSITY_WEIGHT = 0.0

