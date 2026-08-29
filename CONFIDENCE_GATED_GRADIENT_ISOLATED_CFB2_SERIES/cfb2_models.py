"""CFB2: confidence-gated pseudo futures with strict gradient isolation."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SOURCE_PATH = WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"
BASE_LOSS_PATH = WORKSPACE / "FRONT_ENCODER_SERIES" / "locked_mode_query_loss.py"
EXPERIMENTS = {"cfb2_confidence_gradient_isolated"}
PSEUDO_MODES = (18, 19)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = load("cfb2_grouped_source", SOURCE_PATH)
BASE = load("cfb2_base_loss", BASE_LOSS_PATH)


class PseudoAdapter(nn.Module):
    """Small residual head; its inputs are explicitly detached from the B0 graph."""

    def __init__(self, input_dim: int, pred_len: int) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 32), nn.GELU(),
            nn.Linear(32, pred_len * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("time_envelope", torch.linspace(0.15, 1.0, pred_len))

    def forward(self, features: Tensor) -> Tensor:
        raw = self.net(features).reshape(*features.shape[:-1], self.pred_len, 2)
        return 0.35 * torch.tanh(raw) * self.time_envelope.view(1, 1, 1, -1, 1)


class CFB2Decoder(SOURCE.GroupedModeTemplateDecoder):
    def __init__(self, token_dim: int, pred_len: int, cache_path: Path) -> None:
        super().__init__(token_dim, pred_len, use_template=True, use_groups=False)
        cache = np.load(cache_path, allow_pickle=False)
        self.cache_metadata = json.loads(str(cache["metadata_json"].item()))
        self._pseudo = torch.from_numpy(np.asarray(cache["pseudo_futures"], np.float32))
        self._confidence = torch.from_numpy(np.asarray(cache["confidence"], np.float32))
        self._valid = torch.from_numpy(np.asarray(cache["valid_mask"], bool))
        self._batch_indices = None
        self.last_origin = None
        self.last_base_trajectories = None
        self.last_adapter_delta = None
        # context(64) + mode(64) + three anchors(6)
        self.pseudo_adapter = PseudoAdapter(134, pred_len)
        self.register_buffer("pseudo_usage_counts", torch.zeros(3, dtype=torch.long))
        self.register_buffer("pseudo_mode_counts", torch.zeros(20, dtype=torch.long))
        self.register_buffer("pseudo_confidence_sum", torch.zeros(()))
        self.register_buffer("pseudo_confidence_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("adapter_delta_norm_sum", torch.zeros(()))
        self.register_buffer("adapter_delta_norm_count", torch.zeros((), dtype=torch.long))
        self.cv_reference = False

    def set_scene_batch(self, encoded_indices: Tensor, past: Tensor) -> None:
        del past
        indices = -encoded_indices.long() - 2
        if (indices < 0).any() or (indices >= len(self._pseudo)).any():
            raise RuntimeError("CFB2 sample-index sidecar is incompatible")
        self._batch_indices = indices.detach().cpu()

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.eval()
        self.pseudo_adapter.train(True)

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        trajectories, logits = super().forward(encoded_tokens, past_loc)
        self.last_base_trajectories = trajectories.detach()
        recent = encoded_tokens[:, :, -1]
        pooled = encoded_tokens.mean(dim=2)
        context = self.context(torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1))
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        anchors = self.last_corrected_anchors - past_loc[:, :, None, None, -1, :]
        slots = torch.as_tensor(PSEUDO_MODES, device=trajectories.device)
        features = torch.cat((
            context[:, :1, None].expand(-1, -1, len(PSEUDO_MODES), -1),
            mode[:, :1, slots], anchors[:, :1, slots].flatten(-2),
        ), dim=-1).detach()
        delta = self.pseudo_adapter(features)
        adapted = trajectories.clone()
        adapted[:, :1, slots] = adapted[:, :1, slots] + delta
        corrected = self.last_corrected_anchors.clone()
        corrected[:, :1, slots] = corrected[:, :1, slots] + delta[..., [3, 7, self.pred_len - 1], :]
        self.last_corrected_anchors = corrected.detach()
        self.last_adapter_delta = delta
        self.last_origin = past_loc[:, :1, -1]
        return adapted, logits


class CFB2Loss(nn.Module):
    def __init__(self, cfg, decoder: CFB2Decoder) -> None:
        super().__init__()
        self.decoder = decoder
        self.base = BASE.ModeQueryLoss(cfg)
        self.pseudo_weight = 0.10

    def forward(self, predictions, target, logits, cfg=None, sample_weight=None):
        # B0 loss is monitored on the exactly frozen trajectories. It contributes no gradient.
        base_predictions = self.decoder.last_base_trajectories[:, :predictions.shape[1]]
        base_total, parts = self.base(base_predictions, target, logits, cfg, sample_weight=sample_weight)
        if self.decoder._batch_indices is None or self.decoder.last_origin is None:
            raise RuntimeError("CFB2 batch state missing")
        device = predictions.device
        indices = self.decoder._batch_indices
        pseudo = self.decoder._pseudo[indices].to(device)[:, None]
        pseudo = pseudo + self.decoder.last_origin[:, :, None, None, :]
        confidence = self.decoder._confidence[indices].to(device)[:, None]
        valid = self.decoder._valid[indices].to(device)[:, None]
        modes = torch.as_tensor(PSEUDO_MODES, device=device)
        winner = parts["winner"]
        valid = valid & (winner[..., None] != modes.view(1, 1, -1))
        selected = predictions[:, :, modes]
        values = self.base.regression.criterion(selected, pseudo).sum(dim=-1)
        values = (values * self.base.regression.weights).mean(dim=-1)
        weights = confidence * valid.to(confidence.dtype)
        denom = weights.sum()
        pseudo_loss = (values * weights).sum() / denom.clamp_min(1.0)
        if not bool(valid.any()):
            pseudo_loss = selected.sum() * 0.0
        # base_total is detached by construction; only pseudo_adapter can receive gradients.
        total = base_total + self.pseudo_weight * pseudo_loss
        with torch.no_grad():
            per_query = valid[:, 0].sum(dim=-1)
            usage = torch.bincount(per_query, minlength=3).to(self.decoder.pseudo_usage_counts.device)
            self.decoder.pseudo_usage_counts.add_(usage)
            for slot, mode in enumerate(PSEUDO_MODES):
                self.decoder.pseudo_mode_counts[mode].add_(valid[..., slot].sum())
            self.decoder.pseudo_confidence_sum.add_((confidence * valid).sum())
            self.decoder.pseudo_confidence_count.add_(valid.sum())
            delta_norm = torch.linalg.vector_norm(self.decoder.last_adapter_delta, dim=-1).mean(dim=-1)
            self.decoder.adapter_delta_norm_sum.add_((delta_norm * valid[:, :1]).sum())
            self.decoder.adapter_delta_norm_count.add_(valid.sum())
        parts["pseudo"] = pseudo_loss.detach()
        parts["pseudo_valid"] = valid.detach()
        parts["pseudo_confidence"] = confidence.detach()
        return total, parts


def build_loss(cfg, model):
    if not isinstance(model.decoder, CFB2Decoder):
        raise TypeError(type(model.decoder))
    return CFB2Loss(cfg, model.decoder)


def replace_decoder(model, cfg, experiment: str) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(experiment)
    checkpoint = Path(os.environ.get("CFB2_SOURCE_CHECKPOINT", ""))
    cache = Path(os.environ.get("CFB2_CACHE", ""))
    if not checkpoint.is_file() or not cache.is_file():
        raise FileNotFoundError((checkpoint, cache))
    model.decoder = CFB2Decoder(model.token_dim, model.pred_len, cache)
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    target = model.state_dict()
    compatible = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
    incompatible = model.load_state_dict(compatible, strict=False)
    expected = {k for k in target if k.startswith("decoder.pseudo_adapter.") or k.startswith("decoder.pseudo_") or k.startswith("decoder.adapter_")}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))
    skipped = [k for k in source if k not in compatible]
    if skipped:
        raise RuntimeError(f"Skipped source keys: {skipped}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.pseudo_adapter.parameters():
        parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.future_bank_cache = str(cache)
    cfg.MODEL.DECODER.NUM_MODES = 20
