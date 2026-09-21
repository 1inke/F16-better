"""
Lightweight Diagnostic Ranker Model Architecture for Rankability Probe Series.

Enforces specification §13:
LayerNorm(input_dim)
Linear(input_dim, 64)
GELU()
Dropout(0.10)
Linear(64, 32)
GELU()
Linear(32, 1)

Parameter scale: ~6.8k to ~12.2k parameters.
Acts strictly as a diagnostic probe, NOT a heavy selector.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiagnosticProbeRanker(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.10):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, K, input_dim] or [N, input_dim]
        Returns:
            scores: [B, K] or [N] (higher score = predicted higher quality / lower ADE)
        """
        out = self.net(x)
        return out.squeeze(-1)
