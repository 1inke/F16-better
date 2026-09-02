"""Tiny, capacity-controlled intent probes for EXP-183A."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class IntentHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, x):
        raw = self.net(x)
        direction = F.normalize(raw[..., :2], dim=-1, eps=1.0e-6)
        radius = 3.0 * torch.sigmoid(raw[..., 2:3])
        return torch.cat((direction, radius), dim=-1)


class SceneConditionedIntentProbe(nn.Module):
    def __init__(self, motion_dim: int, scene_dim: int):
        super().__init__()
        self.scene_adapter = nn.Sequential(
            nn.LayerNorm(scene_dim), nn.Linear(scene_dim, 64), nn.GELU(), nn.Linear(64, 2 * motion_dim)
        )
        self.head = IntentHead(motion_dim)

    def forward(self, motion, scene):
        gamma, beta = self.scene_adapter(scene).chunk(2, dim=-1)
        refined = (1.0 + 0.1 * torch.tanh(gamma)) * motion + 0.1 * beta
        return self.head(refined)


def build_probe(variant: str, motion_dim: int, lite_dim: int, rich_dim: int):
    if variant == "P0": return IntentHead(motion_dim)
    if variant == "P1_lite": return IntentHead(lite_dim)
    if variant == "P1_rich": return IntentHead(rich_dim)
    if variant == "P2_lite": return IntentHead(motion_dim + lite_dim)
    if variant == "P2_rich": return IntentHead(motion_dim + rich_dim)
    if variant == "P3_lite": return SceneConditionedIntentProbe(motion_dim, lite_dim)
    if variant == "P3_rich": return SceneConditionedIntentProbe(motion_dim, rich_dim)
    raise ValueError(variant)


def probe_forward(model, variant: str, motion, lite, rich):
    if variant == "P0": return model(motion)
    if variant == "P1_lite": return model(lite)
    if variant == "P1_rich": return model(rich)
    if variant == "P2_lite": return model(torch.cat((motion, lite), dim=-1))
    if variant == "P2_rich": return model(torch.cat((motion, rich), dim=-1))
    if variant == "P3_lite": return model(motion, lite)
    if variant == "P3_rich": return model(motion, rich)
    raise ValueError(variant)


def probe_forward(model, variant: str, motion, lite, rich):
    if variant == "P0": return model(motion)
    if variant == "P1_lite": return model(lite)
    if variant == "P1_rich": return model(rich)
    if variant == "P2_lite": return model(torch.cat((motion, lite), dim=-1))
    if variant == "P2_rich": return model(torch.cat((motion, rich), dim=-1))
    if variant == "P3_lite": return model(motion, lite)
    if variant == "P3_rich": return model(motion, rich)
    raise ValueError(variant)
