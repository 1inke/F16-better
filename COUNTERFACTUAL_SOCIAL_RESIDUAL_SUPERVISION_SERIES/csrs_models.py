"""CSRS1: actual-gain need classification plus counterfactual social residual correction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
IR1_PATH = WORKSPACE / "INTERACTION_RESIDUAL_SUPERVISION_SERIES" / "irs_models.py"
SOURCE_PATH = WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"
EXPERIMENTS = {"csrs1_actual_gain_residual"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


IR1 = load_module("csrs_ir1_taps", IR1_PATH)
SOURCE = load_module("csrs_grouped_source", SOURCE_PATH)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


class NeedResidualHead(nn.Module):
    def __init__(self, social_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(social_dim)
        self.down = nn.Linear(social_dim, hidden_dim)
        self.activation = nn.GELU()
        self.need = nn.Linear(hidden_dim, 1)
        self.residual = nn.Linear(hidden_dim, 6)
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)

    def forward(self, social: Tensor):
        hidden = self.activation(self.down(self.norm(social)))
        return hidden, self.need(hidden).squeeze(-1), self.residual(hidden).reshape(*social.shape[:-1], 3, 2)


class CSRSDecoder(SOURCE.GroupedModeTemplateDecoder):
    def __init__(self, token_dim: int, pred_len: int, social_dim: int, cache_path: Path) -> None:
        super().__init__(token_dim, pred_len, use_template=True, use_groups=False)
        self.social_dim = int(social_dim)
        self.head = NeedResidualHead(self.social_dim, 16)
        self.mode_gate = nn.Linear(int(self.mode_embedding.shape[-1]) + 16, 1)
        nn.init.zeros_(self.mode_gate.weight); nn.init.constant_(self.mode_gate.bias, -2.0)
        cache = np.load(cache_path, allow_pickle=False)
        self.cache_metadata = json.loads(str(cache["metadata_json"].item()))
        self._cache_need = torch.from_numpy(np.asarray(cache["need_label"], dtype=np.float32))
        self._cache_confidence = torch.from_numpy(np.asarray(cache["confidence"], dtype=np.float32))
        self._cache_residual = torch.from_numpy(np.asarray(cache["residual_target"], dtype=np.float32))
        self._cache_gain = torch.from_numpy(np.asarray(cache["social_gain"], dtype=np.float32))
        self._batch_indices: Tensor | None = None
        self.last_need_logits: Tensor | None = None
        self.last_need_probability: Tensor | None = None
        self.last_predicted_residual: Tensor | None = None
        self.last_mode_gates: Tensor | None = None
        self.last_applied_anchor_delta: Tensor | None = None
        self.last_aux_need: Tensor | None = None
        self.last_aux_residual: Tensor | None = None
        self.last_batch_gain: Tensor | None = None

    def set_scene_batch(self, encoded_indices: Tensor, past: Tensor) -> None:
        del past
        indices = -encoded_indices.long() - 2
        if (indices < 0).any() or (indices >= len(self._cache_need)).any():
            raise RuntimeError("CSRS sample-index sidecar is missing or incompatible")
        self._batch_indices = indices.detach().cpu()

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.train(True); self.eval(); self.head.train(True); self.mode_gate.train(True)

    def auxiliary_training_loss(self, prediction: Tensor, target: Tensor, logits: Tensor) -> Tensor:
        del prediction, target, logits
        if self._batch_indices is None or self.last_need_logits is None or self.last_predicted_residual is None:
            raise RuntimeError("CSRS auxiliary state was not populated")
        device = self.last_need_logits.device
        idx = self._batch_indices
        label = self._cache_need[idx].to(device)
        confidence = self._cache_confidence[idx].to(device)
        residual_target = self._cache_residual[idx].to(device)
        gain = self._cache_gain[idx].to(device)
        need_logits = self.last_need_logits[:, 0]
        predicted = self.last_predicted_residual[:, 0]
        need_loss = torch.nn.functional.binary_cross_entropy_with_logits(need_logits, label)
        per_sample = (predicted - residual_target).abs().mean(dim=(-1, -2))
        residual_loss = (confidence * per_sample).mean()
        self.last_aux_need = need_loss.detach(); self.last_aux_residual = residual_loss.detach()
        self.last_batch_gain = gain.detach()
        return 0.05 * need_loss + 0.10 * residual_loss

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        social = IR1._SOCIAL_DELTA.get()
        if social is None or social.shape[-1] != self.social_dim:
            raise RuntimeError("CSRS did not receive the F16 social delta")
        recent, pooled = encoded_tokens[:, :, -1], encoded_tokens.mean(dim=2)
        context = self.context(torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1))
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        hidden, need_logits, residual = self.head(social)
        need_probability = torch.sigmoid(need_logits)
        gate_input = torch.cat((mode, hidden[:, :, None].expand(-1, -1, mode.shape[2], -1)), dim=-1)
        mode_gates = torch.sigmoid(self.mode_gate(gate_input)).squeeze(-1)
        anchor_delta = mode_gates[..., None, None] * need_probability[:, :, None, None, None] * residual[:, :, None]

        scale = torch.nn.functional.softplus(self.proposal_scale(context)) + 0.25
        proposal_delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(*mode.shape[:-1], 3, 2)
        source_anchors = scale[:, :, None, None] * (self.anchor_templates[None, None] + proposal_delta)
        base_anchors = source_anchors + self.refiner(mode, source_anchors)
        conditioned_anchors = base_anchors + anchor_delta
        base_relative = torch.einsum("tp,bakpc->baktc", self.interpolation, conditioned_anchors)
        focal_relative, focal_anchors = self.template_head(
            context[:, :1], mode[:, :1], conditioned_anchors[:, :1], self.interpolation, None,
        )
        final_relative = torch.cat((focal_relative, base_relative[:, 1:]), dim=1)
        final_anchors = torch.cat((focal_anchors, conditioned_anchors[:, 1:]), dim=1)
        score_anchors = source_anchors + anchor_delta
        logits = self.score_head(torch.cat((mode, score_anchors.flatten(-2)), dim=-1)).squeeze(-1)
        last = past_loc[:, :, None, -1:, :]
        trajectories = last + final_relative
        batch, agents, modes = logits.shape
        selected = torch.arange(modes, device=logits.device).view(1, 1, modes)
        self.last_selected_indices = selected.expand(batch, agents, modes).detach()
        self.last_coarse_trajectories = (last + base_relative).detach()
        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + score_anchors).detach()
        self.last_corrected_anchors = (past_loc[:, :, None, None, -1, :] + final_anchors).detach()
        self.last_need_logits = need_logits
        self.last_need_probability = need_probability.detach()
        self.last_predicted_residual = residual
        self.last_mode_gates = mode_gates.detach()
        self.last_applied_anchor_delta = anchor_delta.detach()
        self._training_target = None
        return trajectories, logits


def replace_decoder(model, cfg, experiment: str) -> None:
    if experiment not in EXPERIMENTS: raise ValueError(experiment)
    checkpoint = Path(os.environ.get("CSRS_SOURCE_CHECKPOINT", ""))
    cache_path = Path(os.environ.get("CSRS_TEACHER_CACHE", ""))
    if not checkpoint.is_file() or not cache_path.is_file(): raise FileNotFoundError((checkpoint, cache_path))
    cache = np.load(cache_path, allow_pickle=False)
    metadata = json.loads(str(cache["metadata_json"].item()))
    if metadata.get("source_checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("Teacher cache/checkpoint fingerprint mismatch")
    base = model.agent_aware_trans_encoder
    if not bool(getattr(base, "use_temporal_social_cross", False)): raise ValueError("CSRS requires F16")
    base.social_gate = IR1.SocialDeltaTapGate(*list(base.social_gate.children()), dim=int(base.dim))
    base.register_forward_pre_hook(IR1.EncoderContextTap())
    model.decoder = CSRSDecoder(model.token_dim, model.pred_len, int(base.dim), cache_path)
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    target = model.state_dict()
    compatible = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
    incompatible = model.load_state_dict(compatible, strict=False)
    prefixes = ("decoder.head.", "decoder.mode_gate.")
    expected = {k for k in target if k.startswith(prefixes)}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))
    skipped = [k for k in source if k not in compatible]
    if skipped: raise RuntimeError(f"Skipped source keys: {skipped}")
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for parameter in model.agent_aware_trans_encoder.parameters(): parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes): parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint); model.decoder.teacher_cache = str(cache_path)
    model.decoder.cv_reference = False; cfg.MODEL.DECODER.NUM_MODES = 20
