"""EXP-145 balanced intent-query allocation for first-stage proposals."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
STRUCTURAL_PATH = ROOT.parent / "CANDIDATE_GENERATION_SERIES" / "candidate_generation_models.py"
EXPERIMENTS = {
    "i1_balanced_intent_allocation": (False, 0.0),
    "i2_alternate_intent_allocation": (False, 0.35),
    "i3_balanced_intent_dynamic_progress": (True, 0.0),
}


def _load_structural():
    spec = importlib.util.spec_from_file_location("exp145_structural_source", STRUCTURAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(STRUCTURAL_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


STRUCTURAL = _load_structural()


class BalancedIntentDecoder(STRUCTURAL.StructuralCandidateDecoder):
    """Train source intent queries with batch-balanced optimal transport."""

    def __init__(
        self, token_dim: int, pred_len: int, use_progress: bool,
        winner_dropout: float = 0.0,
    ) -> None:
        super().__init__(token_dim, pred_len, use_set=False, use_progress=use_progress)
        self.winner_dropout = float(winner_dropout)
        self.transport_temperature = 0.08
        self.transport_iterations = 30
        self.allocation_weight = 0.35
        self.classification_weight = 0.15
        self.endpoint_weight = 0.50
        self.last_transport_usage: Tensor | None = None

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.eval()
        trainable_modules = (
            self.mode_mlp, self.proposal_scale, self.proposal_delta, self.score_head,
        )
        for module in trainable_modules:
            module.train(True)
        if self.progress_interpolator is not None:
            self.progress_interpolator.train(True)

    def _sinkhorn(self, cost: Tensor) -> Tensor:
        """Return [batch,modes] assignments with balanced batch usage."""
        shifted = cost - cost.min(dim=-1, keepdim=True).values
        q = torch.exp(-shifted / self.transport_temperature).t().clamp_min(1.0e-12)
        q = q / q.sum()
        modes, batch = q.shape
        for _ in range(self.transport_iterations):
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
            q = q / float(modes)
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1.0e-12)
            q = q / float(batch)
        return (q * float(batch)).t()

    def auxiliary_training_loss(
        self, prediction: Tensor, target: Tensor, logits: Tensor
    ) -> Tensor:
        # The locked task supervises the focal agent only.
        candidate = prediction[:, 0]
        truth = target[:, 0]
        distance = torch.linalg.vector_norm(candidate - truth[:, None], dim=-1)
        cost = distance.mean(dim=-1) + self.endpoint_weight * distance[..., -1]
        assignment_cost = cost.detach().clone()
        if self.training and self.winner_dropout > 0.0:
            drop = torch.rand(cost.shape[0], device=cost.device) < self.winner_dropout
            best = assignment_cost.argmin(dim=-1)
            penalty = torch.zeros_like(assignment_cost)
            penalty[torch.arange(cost.shape[0], device=cost.device), best] = drop.to(cost.dtype) * 1.0e3
            assignment_cost = assignment_cost + penalty
        assignment = self._sinkhorn(assignment_cost).detach()
        self.last_transport_usage = assignment.mean(dim=0).detach()
        regression = (assignment * cost).sum(dim=-1).mean()
        classification = -(assignment * F.log_softmax(logits[:, 0], dim=-1)).sum(dim=-1).mean()
        return self.allocation_weight * regression + self.classification_weight * classification


def replace_decoder(model, cfg, name: str) -> None:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown EXP-145 decoder: {name}")
    checkpoint_text = os.environ.get("EXP145_SOURCE_CHECKPOINT", "").strip()
    if not checkpoint_text:
        raise RuntimeError("EXP145_SOURCE_CHECKPOINT is required")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    use_progress, winner_dropout = EXPERIMENTS[name]
    model.decoder = BalancedIntentDecoder(
        model.token_dim, model.pred_len, use_progress, winner_dropout
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected = {
        key for key in model.state_dict()
        if key.startswith("decoder.progress_interpolator.")
    }
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_prefixes = (
        "decoder.mode_embedding", "decoder.anchor_templates",
        "decoder.mode_mlp.", "decoder.proposal_scale.",
        "decoder.proposal_delta.", "decoder.score_head.",
    )
    for parameter_name, parameter in model.named_parameters():
        if parameter_name.startswith(trainable_prefixes):
            parameter.requires_grad_(True)
        if use_progress and parameter_name.startswith("decoder.progress_interpolator."):
            parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.intent_allocation_mechanism = name
    cfg.MODEL.DECODER.NUM_MODES = 20
