"""EXP-146 anchored mode templates and soft hierarchical motion groups."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
CONTROLLED_PATH = ROOT.parent / "CONTROLLED_ANCHOR_CORRECTION_SERIES" / "controlled_anchor_correction_models.py"
EXPERIMENTS = {
    "m1_soft_group_reranking": (False, True),
    "m2_anchored_mode_template": (True, False),
    "m3_grouped_anchored_mode_template": (True, True),
}


def _load_controlled():
    spec = importlib.util.spec_from_file_location("exp146_controlled_source", CONTROLLED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(CONTROLLED_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


CONTROLLED = _load_controlled()


def cluster_mode_shapes(shapes: Tensor, groups: int = 5, iterations: int = 30) -> Tensor:
    """Deterministic k-means over normalized trajectory shapes."""
    endpoint_scale = torch.linalg.vector_norm(shapes[:, -1], dim=-1, keepdim=True).clamp_min(0.20)
    normalized = shapes / endpoint_scale[:, None]
    features = torch.cat((normalized.flatten(1), endpoint_scale.log()), dim=-1)
    first = torch.linalg.vector_norm(features - features.mean(dim=0), dim=-1).argmax()
    centers = [features[first]]
    for _ in range(1, groups):
        distance = torch.cdist(features, torch.stack(centers)).min(dim=-1).values
        centers.append(features[distance.argmax()])
    centers = torch.stack(centers)
    labels = torch.zeros(features.shape[0], dtype=torch.long)
    for _ in range(iterations):
        labels = torch.cdist(features, centers).argmin(dim=-1)
        updated = []
        for group in range(groups):
            members = features[labels == group]
            updated.append(members.mean(dim=0) if len(members) else centers[group])
        new_centers = torch.stack(updated)
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


class AnchoredModeTemplateHead(nn.Module):
    """Align a learned mode shape to anchors, then predict all residuals at once."""

    def __init__(self, num_modes: int = 20, pred_len: int = 12, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_modes = int(num_modes); self.pred_len = int(pred_len)
        self.mode_shapes = nn.Parameter(torch.zeros(num_modes, pred_len, 2))
        self.anchor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 6), nn.Linear(hidden_dim * 2 + 6, 64),
            nn.GELU(), nn.Linear(64, 6),
        )
        nn.init.zeros_(self.anchor_head[-1].weight); nn.init.zeros_(self.anchor_head[-1].bias)
        self.time_embedding = nn.Parameter(torch.empty(pred_len, 32))
        nn.init.normal_(self.time_embedding, std=0.02)
        residual_input = hidden_dim * 2 + 6 + 2 + 32
        self.residual_head = nn.Sequential(
            nn.LayerNorm(residual_input), nn.Linear(residual_input, 96),
            nn.GELU(), nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 2),
        )
        nn.init.zeros_(self.residual_head[-1].weight); nn.init.zeros_(self.residual_head[-1].bias)
        self.register_buffer("anchor_limits", torch.tensor([0.10, 0.15, 0.25]))
        envelope = torch.ones(pred_len, dtype=torch.float32)
        envelope[[3, 7, pred_len - 1]] = 0.0
        self.register_buffer("residual_envelope", envelope)

    def initialize_shapes(self, interpolation: Tensor, anchors: Tensor) -> None:
        with torch.no_grad():
            self.mode_shapes.copy_(torch.einsum("tp,kpc->ktc", interpolation, anchors))

    def forward(
        self, context: Tensor, mode: Tensor, anchors: Tensor,
        interpolation: Tensor, group_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        conditioned_mode = mode if group_features is None else mode + group_features
        shared = torch.cat((context[:, :, None].expand_as(mode), conditioned_mode), dim=-1)
        anchor_input = torch.cat((shared, anchors.flatten(-2)), dim=-1)
        raw_anchor = self.anchor_head(anchor_input).reshape(*anchors.shape)
        anchor_delta = torch.tanh(raw_anchor) * self.anchor_limits.view(1, 1, 1, 3, 1)
        final_anchors = anchors + anchor_delta

        template = self.mode_shapes.view(1, 1, self.num_modes, self.pred_len, 2)
        template_at_anchors = self.mode_shapes[:, [3, 7, self.pred_len - 1]].view(1, 1, self.num_modes, 3, 2)
        alignment = final_anchors - template_at_anchors
        aligned = template + torch.einsum("tp,bakpc->baktc", interpolation, alignment)
        time = self.time_embedding.view(1, 1, 1, self.pred_len, 32).expand(
            aligned.shape[0], aligned.shape[1], aligned.shape[2], -1, -1
        )
        repeated = shared[:, :, :, None].expand(*aligned.shape[:-1], shared.shape[-1])
        anchor_features = final_anchors.flatten(-2)[:, :, :, None].expand(
            *aligned.shape[:-1], 6
        )
        residual_input = torch.cat((repeated, anchor_features, aligned, time), dim=-1)
        residual = 0.35 * torch.tanh(self.residual_head(residual_input))
        residual = residual * self.residual_envelope.view(1, 1, 1, self.pred_len, 1)
        return aligned + residual, final_anchors


class GroupedModeTemplateDecoder(CONTROLLED.FrozenControlledAnchorDecoder):
    def __init__(self, token_dim: int, pred_len: int, use_template: bool, use_groups: bool) -> None:
        super().__init__(token_dim, pred_len)
        self.refiner.max_offsets.copy_(torch.tensor([0.20, 0.25, 0.50]))
        self.use_mode_template = bool(use_template); self.use_motion_groups = bool(use_groups)
        self.template_head = AnchoredModeTemplateHead(20, pred_len, 64) if use_template else None
        self.num_groups = 5
        if use_groups:
            self.group_embedding = nn.Parameter(torch.empty(self.num_groups, 64))
            nn.init.normal_(self.group_embedding, std=0.02)
            self.group_head = nn.Sequential(
                nn.LayerNorm(64), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, self.num_groups),
            )
            nn.init.zeros_(self.group_head[-1].weight); nn.init.zeros_(self.group_head[-1].bias)
        else:
            self.register_parameter("group_embedding", None); self.group_head = None
        self.register_buffer("mode_groups", torch.arange(20, dtype=torch.long) // 4)
        self.last_group_logits: Tensor | None = None

    def initialize_structure(self) -> None:
        shapes = torch.einsum("tp,kpc->ktc", self.interpolation, self.anchor_templates.detach())
        self.mode_groups.copy_(cluster_mode_shapes(shapes, self.num_groups))
        if self.template_head is not None:
            self.template_head.initialize_shapes(self.interpolation, self.anchor_templates.detach())

    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.eval()
        if self.template_head is not None:
            self.template_head.train(True)
        if self.group_head is not None:
            self.group_head.train(True)

    def auxiliary_training_loss(self, prediction: Tensor, target: Tensor, logits: Tensor) -> Tensor:
        del logits
        if self.group_head is None or self.last_group_logits is None:
            return prediction.new_zeros(())
        distance = torch.linalg.vector_norm(prediction[:, 0] - target[:, 0, None], dim=-1)
        cost = distance.mean(dim=-1) + 0.25 * distance[..., -1]
        group_costs = []
        for group in range(self.num_groups):
            members = self.mode_groups == group
            group_costs.append(-0.10 * torch.logsumexp(-cost[:, members] / 0.10, dim=-1))
        group_cost = torch.stack(group_costs, dim=-1)
        soft_target = torch.softmax(-group_cost.detach() / 0.15, dim=-1)
        group_loss = -(soft_target * torch.log_softmax(self.last_group_logits[:, 0], dim=-1)).sum(dim=-1).mean()
        return 0.20 * group_loss

    def forward(self, encoded_tokens: Tensor, past_loc: Tensor):
        recent = encoded_tokens[:, :, -1]; pooled = encoded_tokens.mean(dim=2)
        context = self.context(torch.cat((recent, pooled, self.motion_features(past_loc)), dim=-1))
        mode = self.mode_mlp(context[:, :, None] + self.mode_embedding[None, None])
        scale = torch.nn.functional.softplus(self.proposal_scale(context)) + 0.25
        delta = 0.75 * torch.tanh(self.proposal_delta(mode)).reshape(*mode.shape[:-1], 3, 2)
        source_anchors = scale[:, :, None, None] * (self.anchor_templates[None, None] + delta)
        base_anchors = source_anchors + self.refiner(mode, source_anchors)
        base_relative = torch.einsum("tp,bakpc->baktc", self.interpolation, base_anchors)
        final_relative = base_relative
        final_anchors = base_anchors

        group_features = None
        if self.group_head is not None:
            self.last_group_logits = self.group_head(context)
            group_features = self.group_embedding[self.mode_groups].view(1, 1, 20, 64)
        else:
            self.last_group_logits = None
        if self.template_head is not None:
            focal_relative, focal_anchors = self.template_head(
                context[:, :1], mode[:, :1], base_anchors[:, :1],
                self.interpolation, group_features,
            )
            final_relative = torch.cat((focal_relative, base_relative[:, 1:]), dim=1)
            final_anchors = torch.cat((focal_anchors, base_anchors[:, 1:]), dim=1)

        logits = self.score_head(torch.cat((mode, source_anchors.flatten(-2)), dim=-1)).squeeze(-1)
        if self.last_group_logits is not None:
            group_log_prior = torch.log_softmax(self.last_group_logits, dim=-1)
            logits = logits + 0.35 * group_log_prior[..., self.mode_groups]
        last = past_loc[:, :, None, -1:, :]
        trajectories = last + final_relative
        batch, agents, modes = logits.shape
        selected = torch.arange(modes, device=logits.device).view(1, 1, modes)
        self.last_selected_indices = selected.expand(batch, agents, modes).detach()
        self.last_coarse_trajectories = (last + base_relative).detach()
        self.last_anchor_proposals = (past_loc[:, :, None, None, -1, :] + base_anchors).detach()
        self.last_corrected_anchors = (past_loc[:, :, None, None, -1, :] + final_anchors).detach()
        self._training_target = None
        return trajectories, logits


def replace_decoder(model, cfg, name: str) -> None:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown EXP-146 decoder: {name}")
    checkpoint_text = os.environ.get("EXP146_SOURCE_CHECKPOINT", "").strip()
    if not checkpoint_text:
        raise RuntimeError("EXP146_SOURCE_CHECKPOINT is required")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    use_template, use_groups = EXPERIMENTS[name]
    model.decoder = GroupedModeTemplateDecoder(model.token_dim, model.pred_len, use_template, use_groups)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected = {
        key for key in model.state_dict()
        if key.startswith("decoder.template_head.")
        or key.startswith("decoder.group_head.")
        or key == "decoder.group_embedding"
        or key == "decoder.mode_groups"
    }
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError((incompatible.missing_keys, incompatible.unexpected_keys))
    model.decoder.initialize_structure()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter_name, parameter in model.named_parameters():
        if use_template and parameter_name.startswith("decoder.template_head."):
            parameter.requires_grad_(True)
        if use_groups and parameter_name.startswith("decoder.group_head."):
            parameter.requires_grad_(True)
        if use_groups and use_template and parameter_name == "decoder.group_embedding":
            parameter.requires_grad_(True)
    model.decoder.source_checkpoint = str(checkpoint)
    model.decoder.mode_template_experiment = name
    cfg.MODEL.DECODER.NUM_MODES = 20
