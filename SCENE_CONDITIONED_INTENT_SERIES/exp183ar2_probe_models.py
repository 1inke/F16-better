"""Small equal-capacity candidate-quality probes for EXP-183A-R2-V."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class TinyCandidateQualityProbe(nn.Module):
    """Project a 96-D per-mode scene descriptor, then predict centered quality."""

    def __init__(self, model_dim: int = 71, scene_dim: int = 96, mask_dim: int = 3) -> None:
        super().__init__()
        self.scene_projector = nn.Sequential(
            nn.LayerNorm(scene_dim),
            nn.Linear(scene_dim, 32),
            nn.GELU(),
        )
        self.quality_head = nn.Sequential(
            nn.LayerNorm(model_dim + 32 + mask_dim),
            nn.Linear(model_dim + 32 + mask_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, model_feature: Tensor, scene_feature: Tensor, in_bounds: Tensor) -> Tensor:
        scene = self.scene_projector(scene_feature)
        return self.quality_head(torch.cat((model_feature, scene, in_bounds), dim=-1)).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
