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


class SetAwareDiagnosticProbeRanker(nn.Module):
    """
    R7 Architecture: Candidate-Set-Aware Ranker (§17).
    Input: [B, K, input_dim] (default input_dim=134 from R3)
    Structure:
      LayerNorm(input_dim)
      Linear(input_dim, 64)
      1-layer Self-Attention (hidden=64, heads=4, ffn=128, dropout=0.10)
      Linear(64, 1)
    Parameters: ~26k (< 50k strict limit).
    """
    def __init__(self, input_dim: int = 134, hidden_dim: int = 64, nheads: int = 4, ffn_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, K, input_dim]
        Returns:
            scores: [B, K]
        """
        h = self.proj(self.norm(x))  # [B, K, 64]
        h = self.transformer(h)      # [B, K, 64]
        out = self.out_head(h)       # [B, K, 1]
        return out.squeeze(-1)       # [B, K]
