"""
Local Ranker Models and Nested Calibration for S6 Local Fine Reranking.

Implements:
1. LinearResidualLocalRanker
2. MLPResidualLocalRanker
3. TinySetAttentionRanker

Adheres to:
- Pure offline execution on frozen candidate features
- Nested Fold-Level Inner Validation for residual alpha selection
- Train-only feature standardization and delta-scaling
- Native residual formulation: score = native_logit + alpha * scale * delta
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

if sys.platform == "win32":
    torch_lib = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.linear_model import Ridge

from common.training_config import BATCH_SIZE, LEARNING_RATE, NEURAL_EPOCHS, WEIGHT_DECAY

ALPHA_GRID = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50]


class LocalMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B, K)


class TinySetAttention(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        return self.head(h).squeeze(-1)  # (B, K)


def infer_batched(net: nn.Module, x: torch.Tensor, device: torch.device, batch_size: int = 512) -> np.ndarray:
    net.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            bx = x[start : start + batch_size].to(device)
            outputs.append(net(bx).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def compute_combined_ranking_loss(
    pred: torch.Tensor,               # (B, K)
    ade: torch.Tensor,                # (B, K)
    sample_weights: torch.Tensor | None = None, # (B,)
    tau: float = 0.10,
    margin: float = 0.02,
) -> torch.Tensor:
    B, K = pred.shape
    device = pred.device

    # 1. Soft Relative-Regret Target KL Loss per sample
    min_ade = torch.min(ade, dim=-1, keepdim=True).values
    rel_ade = ade - min_ade
    soft_target = F.softmax(-rel_ade / tau, dim=-1)
    kl_per_k = F.kl_div(F.log_softmax(pred, dim=-1), soft_target, reduction="none")  # (B, K)
    kl_per_sample = torch.sum(kl_per_k, dim=-1)  # (B,)

    # 2. Local Pairwise Loss over all pairs
    pairs_i, pairs_j = torch.triu_indices(K, K, offset=1, device=device)
    diff_ade = ade[:, pairs_i] - ade[:, pairs_j]      # (B, num_pairs)
    diff_score = pred[:, pairs_i] - pred[:, pairs_j]  # (B, num_pairs)

    valid = (torch.abs(diff_ade) >= margin).float()
    labels = (diff_ade < 0).float()
    pair_weights = torch.clamp(torch.abs(diff_ade) / 0.10, min=0.5, max=3.0)
    bce = F.binary_cross_entropy_with_logits(diff_score, labels, reduction="none")
    weighted_bce = bce * pair_weights * valid
    pair_per_sample = torch.sum(weighted_bce, dim=-1) / (torch.sum(pair_weights * valid, dim=-1) + 1e-6)

    sample_loss = 0.5 * kl_per_sample + 0.5 * pair_per_sample
    if sample_weights is not None:
        loss = torch.mean(sample_weights * sample_loss)
    else:
        loss = torch.mean(sample_loss)

    # 3. Residual regularizer
    loss_reg = 1e-3 * torch.mean(pred ** 2)

    return loss + loss_reg


def train_neural_local_ranker(
    net: nn.Module,
    train_x: np.ndarray,              # (N, K, D)
    train_ade: np.ndarray,            # (N, K)
    device: torch.device,
    *,
    sample_weights: np.ndarray | None = None, # (N,)
    sample_mask: np.ndarray | None = None,    # (N,) boolean
    tau: float = 0.10,
    margin: float = 0.02,
    epochs: int = NEURAL_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    optimizer: torch.optim.Optimizer | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    if mean is None:
        mean = np.mean(train_x, axis=(0, 1), keepdims=True)
    if std is None:
        std = np.std(train_x, axis=(0, 1), keepdims=True) + 1e-6

    cur_x = train_x
    cur_ade = train_ade
    cur_w = sample_weights if sample_weights is not None else np.ones(len(train_x), dtype=np.float32)

    if sample_mask is not None:
        cur_x = cur_x[sample_mask]
        cur_ade = cur_ade[sample_mask]
        cur_w = cur_w[sample_mask]

    norm_x = torch.from_numpy((cur_x - mean) / std).float()
    ade_t = torch.from_numpy(cur_ade).float()
    w_t = torch.from_numpy(cur_w).float()

    dataset = TensorDataset(norm_x, ade_t, w_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if optimizer is None:
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    for _ in range(epochs):
        for bx, bade, bw in loader:
            bx, bade, bw = bx.to(device), bade.to(device), bw.to(device)
            optimizer.zero_grad()
            pred = net(bx)
            loss = compute_combined_ranking_loss(pred, bade, sample_weights=bw, tau=tau, margin=margin)
            loss.backward()
            optimizer.step()

    return net, mean, std


def count_parameters(model: Any) -> int:
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif isinstance(model, Ridge):
        return int(model.coef_.size + 1)
    return 0
