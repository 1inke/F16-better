"""EXP-168A joint K20 generator: 10 structured V2 modes + 10 free modes.

Both families consume the same F16 encoder tokens and are optimized together.
There is no post-hoc K40 compression and no family-balancing loss.
"""

from __future__ import annotations

import atexit
import csv
import importlib.util
import json
import os
from collections import deque
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HIERARCHICAL = _load(
    "exp168_hierarchical_source",
    WORKSPACE / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py",
)
CONTROLLED = _load(
    "exp168_controlled_source",
    WORKSPACE / "CONTROLLED_ANCHOR_CORRECTION_SERIES" / "controlled_anchor_correction_models.py",
)
KINEMATIC = _load(
    "exp168_free_source",
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py",
)


class JointStructuredV2Decoder(HIERARCHICAL.HierarchicalLearnedAnchorDecoder):
    """Ten learned three-anchor proposals with jointly trained V2 correction."""

    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(
            token_dim=token_dim,
            pred_len=pred_len,
            num_modes=10,
            anchor_steps=(4, 8, pred_len),
            use_refiner=False,
            top_m=10,
            hidden_dim=64,
        )
        self.refiner = CONTROLLED.BoundedAnchorCorrection(hidden_dim=64, anchor_count=3)
        # Reuse the accepted V2 wide-endpoint bounds.
        self.refiner.max_offsets.copy_(torch.tensor([0.20, 0.25, 0.50]))
        self.use_refiner = True
        self.last_corrected_anchors: Tensor | None = None

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context = self.context(torch.cat(
            (recent, pooled, self.motion_features(past_loc)), dim=-1
        ))
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        scale = F.softplus(self.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(
            *mode.shape[:-1], self.anchor_count, 2
        )
        source_anchors = scale[:, :, None, None] * (
            self.anchor_templates[None, None] + delta
        )
        correction = self.refiner(mode, source_anchors)
        corrected_anchors = source_anchors + correction
        relative = torch.einsum("tp,bakpc->baktc", self.interpolation, corrected_anchors)
        trajectories = past_loc[:, :, None, -1:, :] + relative
        logits = self.score_head(
            torch.cat((mode, corrected_anchors.flatten(-2)), dim=-1)
        ).squeeze(-1)
        self.last_anchor_proposals = (
            past_loc[:, :, None, None, -1, :] + source_anchors
        ).detach()
        self.last_corrected_anchors = (
            past_loc[:, :, None, None, -1, :] + corrected_anchors
        ).detach()
        self.last_coarse_trajectories = trajectories.detach()
        self._training_target = None
        return trajectories, logits


class HybridK20Decoder(nn.Module):
    """A fixed K20 decoder containing two jointly optimized 10-mode families."""

    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__()
        self.num_modes = 20
        self.anchor_modes = 10
        self.free_modes = 10
        self.pred_len = int(pred_len)
        self.anchor_steps = (4, 8, pred_len)
        self.use_refiner = True
        self.structured = JointStructuredV2Decoder(token_dim, pred_len)
        self.free = KINEMATIC.KinematicAnchorDecoder(
            token_dim=token_dim,
            pred_len=pred_len,
            num_modes=10,
            hidden_dim=64,
            residual_modes=4,
        )
        self._training_target: Tensor | None = None
        self._steps = 0
        self._winner_anchor = 0.0
        self._winner_total = 0.0
        self._soft_anchor = 0.0
        self._soft_free = 0.0
        self._diag_rows: list[dict] = []
        self._tail = deque(maxlen=100)
        self._active_anchor: set[int] = set()
        self._active_free: set[int] = set()
        self._grad_sq = {"anchor": 0.0, "free": 0.0}
        self._grad_calls = {"anchor": 0, "free": 0}
        self._diag_path = os.environ.get("EXP168_DIAGNOSTICS_PATH", "").strip()
        self._register_gradient_hooks()
        if self._diag_path:
            atexit.register(self._write_diagnostics)

    def _register_gradient_hooks(self) -> None:
        def hook(family: str):
            def accumulate(gradient: Tensor) -> Tensor:
                self._grad_sq[family] += float(gradient.detach().float().square().sum().cpu())
                self._grad_calls[family] += 1
                return gradient
            return accumulate

        for parameter in self.structured.parameters():
            if parameter.requires_grad:
                parameter.register_hook(hook("anchor"))
        for parameter in self.free.parameters():
            if parameter.requires_grad:
                parameter.register_hook(hook("free"))

    def set_training_target(self, target: Tensor | None) -> None:
        self._training_target = target

    @staticmethod
    def _mean_pairwise(trajectories: Tensor) -> float:
        # Diagnostics use focal-agent mean path distance.
        values = trajectories[:, 0].flatten(-2)
        distances = torch.cdist(values, values) / (trajectories.shape[-2] ** 0.5)
        mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        return float(distances[mask].mean().detach().cpu()) if mask.any() else 0.0

    @staticmethod
    def _cross_distance(anchor: Tensor, free: Tensor) -> float:
        left = anchor[:, 0].flatten(-2)
        right = free[:, 0].flatten(-2)
        return float((torch.cdist(left, right) / (anchor.shape[-2] ** 0.5)).mean().detach().cpu())

    @torch.no_grad()
    def _record_diagnostics(self, trajectories: Tensor, target: Tensor) -> None:
        focal = trajectories[:, 0]
        truth = target[:, 0, None]
        distance = torch.linalg.vector_norm(focal - truth, dim=-1)
        quality = distance.mean(dim=-1) + 0.25 * distance[..., -1]
        winner = quality.argmin(dim=-1)
        gap = quality - quality.min(dim=-1, keepdim=True).values
        scale = gap.mean(dim=-1, keepdim=True).clamp_min(0.05)
        responsibility = torch.softmax(-gap / (0.10 * scale), dim=-1)
        winner_anchor = (winner < self.anchor_modes).float().mean()
        soft_anchor = responsibility[:, :self.anchor_modes].sum(dim=-1).mean()
        soft_free = responsibility[:, self.anchor_modes:].sum(dim=-1).mean()
        anchor_div = self._mean_pairwise(focal[:, None, :self.anchor_modes])
        free_div = self._mean_pairwise(focal[:, None, self.anchor_modes:])
        cross_div = self._cross_distance(
            focal[:, None, :self.anchor_modes], focal[:, None, self.anchor_modes:]
        )
        self._steps += 1
        batch = float(winner.numel())
        self._winner_anchor += float((winner < self.anchor_modes).sum().cpu())
        self._winner_total += batch
        self._soft_anchor += float(soft_anchor.cpu())
        self._soft_free += float(soft_free.cpu())
        self._active_anchor.update(int(x) for x in winner[winner < 10].cpu().tolist())
        self._active_free.update(int(x - 10) for x in winner[winner >= 10].cpu().tolist())
        row = {
            "step": self._steps,
            "winner_anchor": float(winner_anchor.cpu()),
            "winner_free": 1.0 - float(winner_anchor.cpu()),
            "soft_anchor": float(soft_anchor.cpu()),
            "soft_free": float(soft_free.cpu()),
            "within_anchor_diversity": anchor_div,
            "within_free_diversity": free_div,
            "cross_family_distance": cross_div,
        }
        self._diag_rows.append(row)
        self._tail.append(row)

    def _write_diagnostics(self) -> None:
        if not self._diag_path or self._steps == 0:
            return
        path = Path(self._diag_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tail = list(self._tail)
        summary = {
            "training_batches": self._steps,
            "winner_family_fraction": {
                "anchor": self._winner_anchor / max(self._winner_total, 1.0),
                "free": 1.0 - self._winner_anchor / max(self._winner_total, 1.0),
            },
            "soft_responsibility": {
                "anchor": self._soft_anchor / self._steps,
                "free": self._soft_free / self._steps,
            },
            "tail_100_soft_responsibility": {
                "anchor": sum(x["soft_anchor"] for x in tail) / max(len(tail), 1),
                "free": sum(x["soft_free"] for x in tail) / max(len(tail), 1),
            },
            "gradient_norm_proxy": {
                family: (value / max(self._steps, 1)) ** 0.5
                for family, value in self._grad_sq.items()
            },
            "gradient_hook_calls": self._grad_calls,
            "active_modes_by_family": {
                "anchor": len(self._active_anchor), "free": len(self._active_free)
            },
            "mean_diversity": {
                key: sum(x[key] for x in self._diag_rows) / self._steps
                for key in ("within_anchor_diversity", "within_free_diversity", "cross_family_distance")
            },
            "family_collapse_tail_lt_10pct": (
                min(
                    sum(x["soft_anchor"] for x in tail) / max(len(tail), 1),
                    sum(x["soft_free"] for x in tail) / max(len(tail), 1),
                ) < 0.10
            ),
        }
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self._diag_rows[0]))
            writer.writeheader()
            writer.writerows(self._diag_rows)

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        target = self._training_target
        self.structured.set_training_target(target)
        anchor_trajectories, anchor_logits = self.structured(encoded_tokens, past_loc)
        free_trajectories, free_logits = self.free(encoded_tokens, past_loc)
        trajectories = torch.cat((anchor_trajectories, free_trajectories), dim=2)
        logits = torch.cat((anchor_logits, free_logits), dim=-1)
        if self.training and target is not None and self._diag_path:
            self._record_diagnostics(trajectories, target)
        self._training_target = None
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name != "hybrid_k20_10_10":
        raise ValueError(f"Unknown EXP-168 decoder: {name}")
    model.decoder = HybridK20Decoder(model.token_dim, model.pred_len)
    cfg.MODEL.DECODER.NUM_MODES = 20

