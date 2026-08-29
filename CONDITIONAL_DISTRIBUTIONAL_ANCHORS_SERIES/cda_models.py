"""CDA1: fixed-scale conditional mixture likelihood over the three anchors."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
W = ROOT.parent
SOURCE_PATH = W / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"
EXPERIMENTS = {"cda0_matched_continuation", "cda1_fixed_scale_anchor_mixture"}
ANCHOR_INDICES = (3, 7, 11)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


SOURCE = load("cda_grouped_source", SOURCE_PATH)


class CDADecoder(SOURCE.GroupedModeTemplateDecoder):
    def __init__(self, token_dim: int, pred_len: int, sigma: Tensor, use_mixture: bool) -> None:
        super().__init__(token_dim, pred_len, use_template=True, use_groups=False)
        self.register_buffer("anchor_sigma", sigma.float().reshape(3))
        self.use_anchor_mixture = bool(use_mixture)
        self.mixture_weight = 0.10
        self.last_mixture_diagnostics = None
        self.cv_reference = False

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.train(True)
        self.eval()

    def anchor_mixture(self, prediction: Tensor, target: Tensor, logits: Tensor):
        pred_anchor = prediction[..., list(ANCHOR_INDICES), :]
        target_anchor = target[..., list(ANCHOR_INDICES), :].unsqueeze(-3)
        sigma = self.anchor_sigma.view(1, 1, 1, 3, 1).clamp_min(1.0e-4)
        standardized = (pred_anchor - target_anchor) / sigma
        # Isotropic x/y scale at each anchor; constants are retained for a valid NLL.
        log_component = -0.5 * (
            standardized.square() + 2.0 * sigma.log() + math.log(2.0 * math.pi)
        ).sum(dim=(-1, -2))
        log_joint = torch.log_softmax(logits, dim=-1) + log_component
        nll = -torch.logsumexp(log_joint, dim=-1)
        posterior = torch.softmax(log_joint.detach(), dim=-1)
        entropy = -(posterior * posterior.clamp_min(1.0e-12).log()).sum(dim=-1)
        self.last_mixture_diagnostics = {
            "nll_per_dimension": (nll.detach() / 6.0),
            "posterior": posterior,
            "effective_modes": entropy.exp(),
        }
        return nll.mean() / 6.0

    def auxiliary_training_loss(self, prediction: Tensor, target: Tensor, logits: Tensor) -> Tensor:
        if not self.use_anchor_mixture:
            return prediction.sum() * 0.0
        return self.mixture_weight * self.anchor_mixture(prediction, target, logits)


def replace_decoder(model, cfg, experiment: str) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(experiment)
    checkpoint = Path(os.environ.get("CDA_SOURCE_CHECKPOINT", ""))
    scale_path = Path(os.environ.get("CDA_SCALE_JSON", ""))
    if not checkpoint.is_file() or not scale_path.is_file():
        raise FileNotFoundError((checkpoint, scale_path))
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    sigma = torch.tensor(scale["selected_sigma"], dtype=torch.float32)
    use_mixture = experiment == "cda1_fixed_scale_anchor_mixture"
    model.decoder = CDADecoder(model.token_dim, model.pred_len, sigma, use_mixture)
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    target = model.state_dict()
    compatible = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
    incompatible = model.load_state_dict(compatible, strict=False)
    if set(incompatible.missing_keys) != {"decoder.anchor_sigma"} or incompatible.unexpected_keys:
        raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))
    skipped = [k for k in source if k not in compatible]
    if skipped:
        raise RuntimeError(f"Skipped source keys: {skipped}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.agent_aware_trans_encoder.parameters():
        parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.scale_calibration = str(scale_path)
    cfg.MODEL.DECODER.NUM_MODES = 20
