"""Small motion-conditioned adapters on the frozen official H3+V2 decoder."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
V2_MODULE = WORKSPACE / "ANCHOR_CORRECTION_VALIDATION_SERIES" / "anchor_correction_validation_models.py"
EXPERIMENTS = {
    "p1_horizon_motion_proposal_residual": "proposal",
    "c1_horizon_motion_correction_residual": "correction",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def motion_descriptor(past: Tensor) -> Tensor:
    """Nine continuous, mostly dimensionless observation-only motion channels."""
    short = past[:, :, -1] - past[:, :, -2]
    lag = min(3, past.shape[2] - 1)
    long = (past[:, :, -1] - past[:, :, -1 - lag]) / float(lag)
    short_speed = torch.linalg.vector_norm(short, dim=-1, keepdim=True)
    long_speed = torch.linalg.vector_norm(long, dim=-1, keepdim=True)
    short_confidence = torch.sigmoid((short_speed - 0.02) / 0.02)
    long_confidence = torch.sigmoid((long_speed - 0.02) / 0.02)
    short_unit = short / short_speed.clamp_min(1e-5) * short_confidence
    long_unit = long / long_speed.clamp_min(1e-5) * long_confidence
    scale = short_speed + long_speed + 0.02
    acceleration_ratio = torch.tanh(2.0 * (short - long) / scale)
    speed_ratio = torch.log((short_speed + 0.02) / (long_speed + 0.02)).clamp(-2.0, 2.0)
    turn_cross = (long_unit[..., 0] * short_unit[..., 1] - long_unit[..., 1] * short_unit[..., 0]).unsqueeze(-1)
    acceleration_size = (torch.linalg.vector_norm(short - long, dim=-1, keepdim=True) / scale).clamp(0.0, 2.0)
    return torch.cat((short_unit, long_unit, acceleration_ratio, speed_ratio, turn_cross, acceleration_size), dim=-1)


class HorizonMotionResidual(nn.Module):
    def __init__(self, kind: str, hidden_dim: int = 64, anchor_count: int = 3) -> None:
        super().__init__()
        self.kind = kind
        self.anchor_count = anchor_count
        self.motion_encoder = nn.Sequential(
            nn.LayerNorm(9), nn.Linear(9, 32), nn.GELU(), nn.Linear(32, 32),
        )
        input_dim = hidden_dim + 32 + anchor_count * 2
        self.residual = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, anchor_count * 2),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        caps = (0.08, 0.12, 0.20) if kind == "proposal" else (0.05, 0.08, 0.15)
        self.register_buffer("adapter_caps", torch.tensor(caps, dtype=torch.float32))

    def forward(self, mode: Tensor, anchors: Tensor, motion: Tensor) -> Tensor:
        embedded = self.motion_encoder(motion)[:, :, None].expand(-1, -1, mode.shape[2], -1)
        features = torch.cat((mode, embedded, anchors.flatten(-2)), dim=-1)
        raw = self.residual(features).reshape(*anchors.shape)
        return torch.tanh(raw) * self.adapter_caps.view(*((1,) * (raw.ndim - 2)), -1, 1)


class MotionConditionedV2Decoder(nn.Module):
    def __init__(self, base: nn.Module, kind: str) -> None:
        super().__init__()
        self.base = base
        self.kind = kind
        self.adapter = HorizonMotionResidual(kind, hidden_dim=64, anchor_count=3)
        self.num_modes = base.num_modes
        self.pred_len = base.pred_len
        self.anchor_steps = base.anchor_steps
        self.anchor_count = base.anchor_count
        self.top_m = base.top_m
        self.use_refiner = True
        self.last_coarse_trajectories = None
        self.last_selected_indices = None
        self.last_anchor_proposals = None
        self.last_adapter_anchors = None
        self.last_corrected_anchors = None
        self.last_adapter_delta = None
        self.last_base_correction = None
        self.last_final_correction = None

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.eval()
        self.adapter.train(True)

    def set_training_target(self, target: Tensor | None) -> None:
        self.base.set_training_target(target)

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context_input = torch.cat((recent, pooled, self.base.motion_features(past_loc)), dim=-1)
        context = self.base.context(context_input)
        mode = self.base.mode_mlp(context[:, :, None] + self.base.mode_embedding[None, None])
        scale = torch.nn.functional.softplus(self.base.proposal_scale(context)) + 0.25
        proposal_delta = 0.75 * torch.tanh(self.base.proposal_delta(mode)).reshape(
            *mode.shape[:-1], self.anchor_count, 2
        )
        source_anchors = scale[:, :, None, None] * (self.base.anchor_templates[None, None] + proposal_delta)
        source_relative = torch.einsum("tp,bakpc->baktc", self.base.interpolation, source_anchors)
        last = past_loc[:, :, None, -1:, :]
        coarse = last + source_relative
        logits = self.base.score_head(torch.cat((mode, source_anchors.flatten(-2)), dim=-1)).squeeze(-1)

        motion = motion_descriptor(past_loc)
        adapter_delta = self.adapter(mode, source_anchors, motion)
        if self.kind == "proposal":
            adapted_anchors = source_anchors + adapter_delta
            base_correction = self.base.refiner(mode, adapted_anchors)
            final_correction = base_correction
            corrected_anchors = adapted_anchors + final_correction
        else:
            adapted_anchors = source_anchors
            base_correction = self.base.refiner(mode, source_anchors)
            correction_delta = adapter_delta
            caps = self.base.refiner.max_offsets.view(*((1,) * (base_correction.ndim - 2)), -1, 1)
            final_correction = torch.maximum(torch.minimum(base_correction + correction_delta, caps), -caps)
            corrected_anchors = source_anchors + final_correction

        corrected_relative = torch.einsum("tp,bakpc->baktc", self.base.interpolation, corrected_anchors)
        trajectories = last + corrected_relative
        batch, agents, modes = logits.shape
        selected = torch.arange(modes, device=logits.device).view(1, 1, modes).expand(batch, agents, modes)
        self.last_selected_indices = selected.detach()
        self.last_coarse_trajectories = coarse.detach()
        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + source_anchors).detach()
        self.last_adapter_anchors = (past_loc[:, :, None, None, -1, :] + adapted_anchors).detach()
        self.last_corrected_anchors = (past_loc[:, :, None, None, -1, :] + corrected_anchors).detach()
        self.last_adapter_delta = adapter_delta.detach()
        self.last_base_correction = base_correction.detach()
        self.last_final_correction = final_correction.detach()
        self.base._training_target = None
        return trajectories, logits


def replace_decoder(model: nn.Module, cfg, name: str) -> None:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown motion-conditioned decoder: {name}")
    h3_text = os.environ.get("HRMC_H3_CHECKPOINT", "").strip()
    v2_text = os.environ.get("HRMC_V2_CHECKPOINT", "").strip()
    if not h3_text or not v2_text:
        raise RuntimeError("HRMC_H3_CHECKPOINT and HRMC_V2_CHECKPOINT are required")
    h3, v2 = Path(h3_text), Path(v2_text)
    if not h3.is_file() or not v2.is_file():
        raise FileNotFoundError((h3, v2))
    os.environ["EXP143_SOURCE_CHECKPOINT"] = str(h3)
    v2_module = load_module(f"hrmc_v2_{name}", V2_MODULE)
    v2_module.replace_decoder(model, cfg, "v2_wide_endpoint_correction")
    model.load_state_dict(torch.load(v2, map_location="cpu", weights_only=True), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.decoder = MotionConditionedV2Decoder(model.decoder, EXPERIMENTS[name])
    for parameter in model.decoder.adapter.parameters():
        parameter.requires_grad_(True)
    model.decoder.source_h3_checkpoint = str(h3)
    model.decoder.source_v2_checkpoint = str(v2)
    cfg.MODEL.DECODER.NUM_MODES = 20
