"""IRA1: distributional intent supervision without changing C00 inference.

Both registered arms retain the exact C00 proposal, correction and score path.
They also train an identical conditional flow to model the six physical intent
controls used by PIF1.  The only intervention is whether the auxiliary NLL may
back-propagate through the shared context representation.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONTROLLED_PATH = (
    REPO / "CONTROLLED_ANCHOR_CORRECTION_SERIES"
    / "controlled_anchor_correction_models.py"
)
PIF_PATH = REPO / "PHYSICAL_INTENT_FLOW_SERIES" / "physical_intent_flow_models.py"
EXPERIMENTS = {
    "ira0_detached_intent_aux": False,
    "ira1_joint_intent_aux": True,
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROLLED = _load_module("ira_controlled_source", CONTROLLED_PATH)
PIF = _load_module("ira_pif_source", PIF_PATH)


class IntentAuxiliaryDecoder(CONTROLLED.FrozenControlledAnchorDecoder):
    """C00 decoder plus a training-only conditional intent likelihood head."""

    def __init__(
        self,
        token_dim: int,
        pred_len: int,
        stats: dict,
        joint_intent_gradient: bool,
    ) -> None:
        super().__init__(token_dim, pred_len)
        if pred_len != 12:
            raise ValueError(f"IRA1 is preregistered for 12 future steps, got {pred_len}")
        mean = torch.tensor(stats["control_mean"], dtype=torch.float32)
        std = torch.tensor(stats["control_std"], dtype=torch.float32)
        if (
            mean.shape != (PIF.CONTROL_DIM,)
            or std.shape != (PIF.CONTROL_DIM,)
            or torch.any(std <= 0)
        ):
            raise ValueError("Invalid physical-intent calibration statistics")

        self.intent_flow = PIF.ConditionalIntentFlow(
            context_dim=64,
            hidden_dim=int(os.environ.get("IRA_FLOW_HIDDEN_DIM", "96")),
            layers=int(os.environ.get("IRA_FLOW_LAYERS", "6")),
        )
        self.register_buffer("intent_control_mean", mean)
        self.register_buffer("intent_control_std", std)
        self.speed_floor = float(stats["speed_floor"])
        self.intent_nll_weight = float(os.environ.get("IRA_NLL_WEIGHT", "0.05"))
        self.joint_intent_gradient = bool(joint_intent_gradient)
        self.last_intent_nll: Tensor | None = None

    def apply_training_policy(self, owner: nn.Module) -> None:
        # This is a continuation of the jointly trainable C00 model.  The
        # detached arm blocks only the auxiliary gradient at `context` below.
        owner.train(True)

    def auxiliary_training_loss(
        self, prediction: Tensor, target: Tensor, logits: Tensor
    ) -> Tensor:
        del target, logits
        if self.last_intent_nll is None:
            return prediction.sum() * 0.0
        return self.intent_nll_weight * self.last_intent_nll

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        # This block intentionally mirrors FrozenControlledAnchorDecoder/C00.
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
        source_anchors = scale[:, :, None, None] * (
            self.anchor_templates[None, None] + delta
        )
        source_relative = torch.einsum(
            "tp,bakpc->baktc", self.interpolation, source_anchors
        )
        last = past_loc[:, :, None, -1:, :]
        coarse = last + source_relative
        logits = self.score_head(
            torch.cat((mode, source_anchors.flatten(-2)), dim=-1)
        ).squeeze(-1)

        correction = self.refiner(mode, source_anchors)
        corrected_anchors = source_anchors + correction
        corrected_relative = torch.einsum(
            "tp,bakpc->baktc", self.interpolation, corrected_anchors
        )
        trajectories = last + corrected_relative

        target = self._training_target
        self._training_target = None
        self.last_intent_nll = None
        if target is not None:
            target_anchors = (
                target[:, :1, list(PIF.ANCHOR_INDICES), :]
                - past_loc[:, :1, -1:, :]
            )
            controls = PIF.anchors_to_controls(
                target_anchors, past_loc[:, :1], self.speed_floor
            )
            standardized = (
                controls - self.intent_control_mean
            ) / self.intent_control_std
            flow_context = context[:, :1]
            if not self.joint_intent_gradient:
                flow_context = flow_context.detach()
            log_prob = self.intent_flow.log_prob(standardized, flow_context)
            self.last_intent_nll = (-log_prob).mean() / float(PIF.CONTROL_DIM)

        batch, agents, modes = logits.shape
        selected = torch.arange(modes, device=logits.device).view(1, 1, modes)
        self.last_selected_indices = selected.expand(batch, agents, modes).detach()
        self.last_coarse_trajectories = coarse.detach()
        self.last_anchor_proposals = (
            past_loc[:, :, None, None, -1, :] + source_anchors
        ).detach()
        self.last_corrected_anchors = (
            past_loc[:, :, None, None, -1, :] + corrected_anchors
        ).detach()
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown IRA1 experiment: {name}")
    source = Path(os.environ.get("IRA_SOURCE_CHECKPOINT", ""))
    stats_path = Path(os.environ.get("IRA_STATS_JSON", ""))
    if not source.is_file() or not stats_path.is_file():
        raise FileNotFoundError((source, stats_path))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    model.decoder = IntentAuxiliaryDecoder(
        model.token_dim,
        model.pred_len,
        stats,
        joint_intent_gradient=EXPERIMENTS[name],
    )
    state = torch.load(source, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {
        key
        for key in model.state_dict()
        if key.startswith("decoder.intent_flow.")
        or key in {
            "decoder.intent_control_mean",
            "decoder.intent_control_std",
        }
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            f"Unexpected C00 missing keys: {sorted(incompatible.missing_keys)}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected C00 keys: {incompatible.unexpected_keys}")

    required_groups = {
        "encoder": any(
            parameter.requires_grad
            for key, parameter in model.named_parameters()
            if not key.startswith("decoder.")
        ),
        "proposal": any(
            parameter.requires_grad
            for key, parameter in model.decoder.named_parameters()
            if not key.startswith("refiner.")
            and not key.startswith("score_head.")
            and not key.startswith("intent_flow.")
        ),
        "score": any(p.requires_grad for p in model.decoder.score_head.parameters()),
        "correction": any(p.requires_grad for p in model.decoder.refiner.parameters()),
        "intent_flow": any(p.requires_grad for p in model.decoder.intent_flow.parameters()),
    }
    if not all(required_groups.values()):
        raise RuntimeError(f"IRA1 trainable group failure: {required_groups}")

    model.decoder.source_checkpoint = str(source)
    model.decoder.stats_path = str(stats_path)
    model.decoder.experiment_name = name
    model.decoder.training_policy_name = (
        "joint_c00_plus_intent_flow_"
        + ("context_gradient" if EXPERIMENTS[name] else "detached_context_control")
    )
    model.decoder.trainable_groups = required_groups
    model.decoder.cv_reference = False
    cfg.MODEL.DECODER.NUM_MODES = 20

