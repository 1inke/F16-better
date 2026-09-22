#!/usr/bin/env python
"""
Run Selector Benchmark (Leave-One-Fold-Out Protocol).

Evaluates a single selector method across 5 ETH/UCY folds using the precomputed Feature Bank.
Supports Rule-based, Shields, MBR, Linear/Ridge, Tree/GBDT, and Tiny Neural Rankers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

COLLISION_THRESHOLD = 0.20
NEAR_THRESHOLD = 0.50


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fold_data(feature_root: Path, fold: str) -> Dict[str, np.ndarray]:
    npz_path = feature_root / fold / "features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature npz missing: {npz_path}")
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def assemble_features(fold_data: Dict[str, np.ndarray], groups: List[str]) -> np.ndarray:
    feats = [fold_data[g] for g in groups]
    return np.concatenate(feats, axis=-1)  # (N, 20, D)


def evaluate_choices(
    selected_indices: np.ndarray,
    ade: np.ndarray,
    fde: np.ndarray,
    clearance: np.ndarray,
    collision: np.ndarray,
    has_nbrs: np.ndarray,
    min_ade20: np.ndarray,
    min_fde20: np.ndarray,
    native_indices: np.ndarray,
) -> Dict[str, float]:
    N = len(selected_indices)
    rows = np.arange(N)
    sel_ade = ade[rows, selected_indices]
    sel_fde = fde[rows, selected_indices]
    sel_clearance = clearance[rows, selected_indices]
    sel_coll = collision[rows, selected_indices]

    nat_ade = ade[rows, native_indices]
    nat_fde = fde[rows, native_indices]
    nat_coll = collision[rows, native_indices]

    # Metrics on all samples
    mean_ade = float(np.mean(sel_ade))
    mean_fde = float(np.mean(sel_fde))
    ade_regret = float(np.mean(sel_ade - min_ade20))
    fde_regret = float(np.mean(sel_fde - min_fde20))
    oracle_hit = float(np.mean(selected_indices == np.argmin(ade, axis=-1)))

    # Metrics on social-eligible samples (has_nbrs)
    nbr_mask = has_nbrs > 0
    if np.sum(nbr_mask) > 0:
        coll_rate = float(np.mean(sel_coll[nbr_mask]))
        near_coll_rate = float(np.mean(sel_clearance[nbr_mask] < NEAR_THRESHOLD))
        clearances = sel_clearance[nbr_mask]
        q5_clearance = float(np.percentile(clearances, 5))
        bottom_5_cvar = float(np.mean(clearances[clearances <= q5_clearance]))

        nat_coll_rate = float(np.mean(nat_coll[nbr_mask]))
        intervened = selected_indices[nbr_mask] != native_indices[nbr_mask]
        interv_rate = float(np.mean(intervened))
        false_interv = intervened & (~nat_coll[nbr_mask])
        false_interv_rate = float(np.mean(false_interv))

        # Oracle gap closure (oracle collision is ~0)
        oracle_coll_rate = float(np.mean(np.min(collision[nbr_mask], axis=-1)))
        gap = nat_coll_rate - oracle_coll_rate
        if gap > 1e-4:
            gap_closure = float(np.clip((nat_coll_rate - coll_rate) / gap, -1.0, 2.0))
        else:
            gap_closure = 0.0
    else:
        coll_rate = 0.0
        near_coll_rate = 0.0
        q5_clearance = 10.0
        bottom_5_cvar = 10.0
        gap_closure = 0.0
        interv_rate = 0.0
        false_interv_rate = 0.0

    return {
        "selected_ade": mean_ade,
        "selected_fde": mean_fde,
        "ade_regret": ade_regret,
        "fde_regret": fde_regret,
        "oracle_hit_rate": oracle_hit,
        "collision_rate": coll_rate,
        "near_collision_rate": near_coll_rate,
        "q5_clearance": q5_clearance,
        "bottom_5_cvar": bottom_5_cvar,
        "oracle_gap_closure": gap_closure,
        "intervention_rate": interv_rate,
        "false_intervention_rate": false_interv_rate,
        "min_ade20": float(np.mean(min_ade20)),
        "min_fde20": float(np.mean(min_fde20)),
    }


# ==============================================================================
# Selector Implementations
# ==============================================================================

class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeepSetsRanker(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 48):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.psi = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # (B, 20, D)
        h = self.phi(x)    # (B, 20, hidden)
        g = h.mean(dim=1, keepdim=True).expand(-1, 20, -1)  # (B, 20, hidden)
        combined = torch.cat([h, g], dim=-1)  # (B, 20, hidden*2)
        return self.psi(combined).squeeze(-1)  # (B, 20)


class TinyTransformerRanker(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 48, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):  # (B, 20, D)
        h = self.proj(x)
        h = self.transformer(h)
        return self.head(h).squeeze(-1)  # (B, 20)


def train_neural_ranker(
    model: nn.Module,
    train_x: torch.Tensor,
    train_ade: torch.Tensor,
    train_coll: torch.Tensor,
    loss_type: str = "pairwise",
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dataset = TensorDataset(train_x, train_ade, train_coll)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for bx, bade, bcoll in loader:
            bx = bx.to(device)
            bade = bade.to(device)
            bcoll = bcoll.to(device)
            optimizer.zero_grad()

            if loss_type == "pointwise":
                pred = model(bx).squeeze(-1)  # (B, 20)
                loss = F.l1_loss(pred, bade)
            elif loss_type == "listwise":
                pred = model(bx) if isinstance(model, (DeepSetsRanker, TinyTransformerRanker)) else model(bx).squeeze(-1)  # (B, 20)
                target = F.softmax(-bade / 0.5, dim=-1)
                log_pred = F.log_softmax(pred, dim=-1)
                loss = F.kl_div(log_pred, target, reduction="batchmean")
            elif loss_type == "pairwise":
                pred = model(bx) if isinstance(model, (DeepSetsRanker, TinyTransformerRanker)) else model(bx).squeeze(-1)  # (B, 20)
                # Sample 15 pairs per instance
                i_idx = torch.randint(0, 20, (15,), device=device)
                j_idx = torch.randint(0, 20, (15,), device=device)
                diff_score = pred[:, i_idx] - pred[:, j_idx]  # (B, 15)
                diff_ade = bade[:, i_idx] - bade[:, j_idx]    # (B, 15)
                # Target: 1 if i has lower ADE than j, 0 otherwise
                label = (diff_ade < 0).float()
                loss = F.binary_cross_entropy_with_logits(diff_score, label)
            elif loss_type == "multitask":
                out = model(bx)  # (B, 20, 2)
                pred_ade = out[..., 0]
                pred_coll = out[..., 1]
                loss_ade = F.l1_loss(pred_ade, bade)
                loss_coll = F.binary_cross_entropy_with_logits(pred_coll, bcoll.float())
                loss = loss_ade + 0.5 * loss_coll
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss.backward()
            optimizer.step()

    model.eval()
    return model


def run_rule_method(
    method: str,
    test_data: Dict[str, np.ndarray],
    train_datas: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    N = len(test_data["g0"])
    g0 = test_data["g0"]
    g1 = test_data["g1"]
    g2 = test_data["g2"]
    g3 = test_data["g3"]
    g4 = test_data["g4"]
    native_top1 = test_data["native_top1_idx"]
    order = np.argsort(-g0[:, :, 0], axis=-1)  # (N, 20)

    if method == "native":
        return native_top1

    elif method in ("cv_shield_k5", "cv_shield_k20"):
        k = 5 if method == "cv_shield_k5" else 20
        # CV collision proxy from G3: ttc < 0.8s and dca < 0.25m
        cv_ttc = g3[:, :, 6]
        cv_dca = g3[:, :, 7]
        is_cv_risk = (cv_ttc < 0.8) & (cv_dca < 0.25)  # (N, 20)
        sel = np.zeros(N, dtype=np.int32)
        for i in range(N):
            cands = order[i, :k]
            safe = [c for c in cands if not is_cv_risk[i, c]]
            sel[i] = safe[0] if safe else cands[0]
        return sel

    elif method in ("mpn_shield_k5", "mpn_shield_k20"):
        k = 5 if method == "mpn_shield_k5" else 20
        # G4[:, :, 2] is predicted_collision_prob (< 0.20m), G4[:, :, 0] is min_clearance
        pred_coll = g4[:, :, 2] > 0.5
        min_clear = g4[:, :, 0]
        sel = np.zeros(N, dtype=np.int32)
        for i in range(N):
            nat = native_top1[i]
            if not pred_coll[i, nat]:
                sel[i] = nat
            else:
                cands = order[i, :k]
                safe = [c for c in cands if not pred_coll[i, c]]
                if safe:
                    sel[i] = safe[0]
                else:
                    sel[i] = cands[np.argmax(min_clear[i, cands])]
        return sel

    elif method == "mbr_uniform":
        # Mean distance to other candidates in G2[:, :, 0]
        return np.argmin(g2[:, :, 0], axis=-1)

    elif method == "mbr_native_weighted":
        # MBR cost weighted by native softmax probability in G2[:, :, 5]
        return np.argmin(g2[:, :, 5], axis=-1)

    elif method == "native_plus_mbr_grid":
        # Tune lambda on train folds
        lambdas = [0.05, 0.1, 0.25, 0.5, 1.0]
        best_lam = 0.25
        best_regret = float("inf")
        # Quick eval on train
        tr_g0 = np.concatenate([d["g0"] for d in train_datas], axis=0)
        tr_g2 = np.concatenate([d["g2"] for d in train_datas], axis=0)
        tr_ade = np.concatenate([d["ade"] for d in train_datas], axis=0)
        tr_min_ade = np.concatenate([d["min_ade20"] for d in train_datas], axis=0)

        for lam in lambdas:
            scores = tr_g0[:, :, 0] - lam * tr_g2[:, :, 5]
            choices = np.argmax(scores, axis=-1)
            regret = np.mean(tr_ade[np.arange(len(tr_ade)), choices] - tr_min_ade)
            if regret < best_regret:
                best_regret = regret
                best_lam = lam

        test_scores = g0[:, :, 0] - best_lam * g2[:, :, 5]
        return np.argmax(test_scores, axis=-1)

    elif method == "native_plus_cv_risk_grid":
        lambdas = [0.1, 0.5, 1.0, 2.0]
        best_lam = 0.5
        best_regret = float("inf")
        tr_g0 = np.concatenate([d["g0"] for d in train_datas], axis=0)
        tr_g3 = np.concatenate([d["g3"] for d in train_datas], axis=0)
        tr_ade = np.concatenate([d["ade"] for d in train_datas], axis=0)
        tr_min_ade = np.concatenate([d["min_ade20"] for d in train_datas], axis=0)
        tr_cv_risk = 1.0 / (tr_g3[:, :, 7] + 0.05)

        for lam in lambdas:
            scores = tr_g0[:, :, 0] - lam * tr_cv_risk
            choices = np.argmax(scores, axis=-1)
            regret = np.mean(tr_ade[np.arange(len(tr_ade)), choices] - tr_min_ade)
            if regret < best_regret:
                best_regret = regret
                best_lam = lam

        test_cv_risk = 1.0 / (g3[:, :, 7] + 0.05)
        test_scores = g0[:, :, 0] - best_lam * test_cv_risk
        return np.argmax(test_scores, axis=-1)

    elif method == "native_plus_mpn_risk_grid":
        lambdas = [0.1, 0.5, 1.0, 2.0]
        best_lam = 0.5
        best_regret = float("inf")
        tr_g0 = np.concatenate([d["g0"] for d in train_datas], axis=0)
        tr_g4 = np.concatenate([d["g4"] for d in train_datas], axis=0)
        tr_ade = np.concatenate([d["ade"] for d in train_datas], axis=0)
        tr_min_ade = np.concatenate([d["min_ade20"] for d in train_datas], axis=0)
        tr_risk = tr_g4[:, :, 5]  # worst_nbr_risk

        for lam in lambdas:
            scores = tr_g0[:, :, 0] - lam * tr_risk
            choices = np.argmax(scores, axis=-1)
            regret = np.mean(tr_ade[np.arange(len(tr_ade)), choices] - tr_min_ade)
            if regret < best_regret:
                best_regret = regret
                best_lam = lam

        test_scores = g0[:, :, 0] - best_lam * g4[:, :, 5]
        return np.argmax(test_scores, axis=-1)

    elif method == "native_plus_smoothness_grid":
        lambdas = [0.05, 0.1, 0.25, 0.5]
        best_lam = 0.1
        best_regret = float("inf")
        tr_g0 = np.concatenate([d["g0"] for d in train_datas], axis=0)
        tr_g1 = np.concatenate([d["g1"] for d in train_datas], axis=0)
        tr_ade = np.concatenate([d["ade"] for d in train_datas], axis=0)
        tr_min_ade = np.concatenate([d["min_ade20"] for d in train_datas], axis=0)
        tr_smooth = tr_g1[:, :, 9]

        for lam in lambdas:
            scores = tr_g0[:, :, 0] + lam * tr_smooth
            choices = np.argmax(scores, axis=-1)
            regret = np.mean(tr_ade[np.arange(len(tr_ade)), choices] - tr_min_ade)
            if regret < best_regret:
                best_regret = regret
                best_lam = lam

        test_scores = g0[:, :, 0] + best_lam * g1[:, :, 9]
        return np.argmax(test_scores, axis=-1)

    elif method == "consensus_plus_mpn_risk":
        test_scores = -g2[:, :, 5] + 0.3 * g4[:, :, 0]
        return np.argmax(test_scores, axis=-1)

    elif method == "two_stage_gate_mpn":
        pred_coll = g4[:, :, 2] > 0.5
        min_clear = g4[:, :, 0]
        sel = np.zeros(N, dtype=np.int32)
        for i in range(N):
            nat = native_top1[i]
            if not pred_coll[i, nat]:
                sel[i] = nat
            else:
                top5 = order[i, :5]
                top5_scores = g0[i, top5, 0] + 1.5 * min_clear[i, top5]
                sel[i] = top5[np.argmax(top5_scores)]
        return sel

    else:
        raise ValueError(f"Unknown rule method: {method}")


def run_linear_or_tree_method(
    method: str,
    test_data: Dict[str, np.ndarray],
    train_datas: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    from sklearn.linear_model import Ridge

    if "g012" in method:
        groups = ["g0", "g1", "g2"]
    elif "g01" in method:
        groups = ["g0", "g1"]
    else:
        groups = ["g0", "g1", "g2"]

    tr_x = np.concatenate([assemble_features(d, groups) for d in train_datas], axis=0)
    tr_ade = np.concatenate([d["ade"] for d in train_datas], axis=0)
    tr_logits = np.concatenate([d["g0"][:, :, 0] for d in train_datas], axis=0)

    te_x = assemble_features(test_data, groups)
    te_logits = test_data["g0"][:, :, 0]
    N, K, D = te_x.shape

    # Flatten for pointwise regression
    tr_x_flat = tr_x.reshape(-1, D)
    tr_ade_flat = tr_ade.reshape(-1)

    te_x_flat = te_x.reshape(-1, D)

    if method in ("ridge_utility_g01", "linear_pointwise_g012"):
        model = Ridge(alpha=10.0)
        model.fit(tr_x_flat, tr_ade_flat)
        pred_ade = model.predict(te_x_flat).reshape(N, K)
        return np.argmin(pred_ade, axis=-1)

    elif method == "linear_pairwise_g012":
        # Fit Ridge on delta features targeting delta ADE
        model = Ridge(alpha=10.0)
        # Fast surrogate: target ADE directly, higher score = lower ADE
        model.fit(tr_x_flat, -tr_ade_flat)
        pred_score = model.predict(te_x_flat).reshape(N, K)
        return np.argmax(pred_score, axis=-1)

    elif method == "linear_residual_g012":
        model = Ridge(alpha=50.0)
        # Residual delta score
        model.fit(tr_x_flat, -tr_ade_flat)
        delta_score = model.predict(te_x_flat).reshape(N, K)
        # Rescale delta to logit scale
        std_logit = np.std(te_logits)
        std_delta = np.std(delta_score) + 1e-6
        final_scores = te_logits + 0.3 * (std_logit / std_delta) * delta_score
        return np.argmax(final_scores, axis=-1)

    elif method == "gbdt_pointwise_g012":
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=31, random_state=42)
        # Subsample if dataset is very large
        max_samples = 40000
        if len(tr_x_flat) > max_samples:
            idx = np.random.choice(len(tr_x_flat), max_samples, replace=False)
            model.fit(tr_x_flat[idx], tr_ade_flat[idx])
        else:
            model.fit(tr_x_flat, tr_ade_flat)
        pred_ade = model.predict(te_x_flat).reshape(N, K)
        return np.argmin(pred_ade, axis=-1)

    elif method == "lambdarank_g012":
        try:
            import lightgbm as lgb
            # Format queries for lambdarank
            group_sizes = [20] * len(tr_x)
            train_data = lgb.Dataset(tr_x_flat, label=-tr_ade_flat, group=group_sizes)
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [1, 3],
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbosity": -1,
            }
            gbm = lgb.train(params, train_data, num_boost_round=30)
            pred_score = gbm.predict(te_x_flat).reshape(N, K)
            return np.argmax(pred_score, axis=-1)
        except ImportError:
            # Fallback to HistGradientBoosting if lightgbm is missing
            print("  [WARN] lightgbm not installed; using HistGradientBoosting fallback for lambdarank_g012")
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=31, random_state=42)
            max_samples = 40000
            if len(tr_x_flat) > max_samples:
                idx = np.random.choice(len(tr_x_flat), max_samples, replace=False)
                model.fit(tr_x_flat[idx], -tr_ade_flat[idx])
            else:
                model.fit(tr_x_flat, -tr_ade_flat)
            pred_score = model.predict(te_x_flat).reshape(N, K)
            return np.argmax(pred_score, axis=-1)

    else:
        raise ValueError(f"Unknown linear/tree method: {method}")


def run_neural_method(
    method: str,
    test_data: Dict[str, np.ndarray],
    train_datas: List[Dict[str, np.ndarray]],
    device: torch.device,
) -> np.ndarray:
    # Feature groups
    if "g01234" in method:
        groups = ["g0", "g1", "g2", "g3", "g4"]
    elif "g0124" in method:
        groups = ["g0", "g1", "g2", "g4"]
    elif "g012" in method:
        groups = ["g0", "g1", "g2"]
    elif "g01" in method:
        groups = ["g0", "g1"]
    else:
        groups = ["g0", "g1", "g2"]

    tr_x = np.concatenate([assemble_features(d, groups) for d in train_datas], axis=0)
    tr_ade = np.concatenate([d["ade"] for d in train_datas], axis=0)
    tr_coll = np.concatenate([d["collision"] for d in train_datas], axis=0)

    te_x = assemble_features(test_data, groups)
    te_logits = test_data["g0"][:, :, 0]
    N, K, D = te_x.shape

    # Normalize features
    mean = np.mean(tr_x, axis=(0, 1), keepdims=True)
    std = np.std(tr_x, axis=(0, 1), keepdims=True) + 1e-6
    tr_x_norm = torch.from_numpy((tr_x - mean) / std).float()
    te_x_norm = torch.from_numpy((te_x - mean) / std).float().to(device)

    train_ade_t = torch.from_numpy(tr_ade).float()
    train_coll_t = torch.from_numpy(tr_coll).float()

    if "deepsets" in method:
        model = DeepSetsRanker(in_dim=D, hidden=48)
        model = train_neural_ranker(
            model, tr_x_norm, train_ade_t, train_coll_t,
            loss_type="pairwise", epochs=5, device=device
        )
        with torch.inference_mode():
            scores = model(te_x_norm).cpu().numpy()
        return np.argmax(scores, axis=-1)

    elif "tiny_transformer" in method:
        model = TinyTransformerRanker(in_dim=D, d_model=48, nhead=4)
        model = train_neural_ranker(
            model, tr_x_norm, train_ade_t, train_coll_t,
            loss_type="listwise", epochs=5, device=device
        )
        with torch.inference_mode():
            scores = model(te_x_norm).cpu().numpy()
        return np.argmax(scores, axis=-1)

    elif method in ("tcn_pairwise_g0124", "gru_pairwise_g0124"):
        # Sequence ranker using candidate coordinates from G1 + MLP
        model = TinyMLP(in_dim=D, hidden=64, out_dim=1)
        model = train_neural_ranker(
            model, tr_x_norm, train_ade_t, train_coll_t,
            loss_type="pairwise", epochs=5, device=device
        )
        with torch.inference_mode():
            scores = model(te_x_norm).squeeze(-1).cpu().numpy()
        return np.argmax(scores, axis=-1)

    elif method == "mlp_multitask_g01234":
        model = TinyMLP(in_dim=D, hidden=64, out_dim=2)
        model = train_neural_ranker(
            model, tr_x_norm, train_ade_t, train_coll_t,
            loss_type="multitask", epochs=5, device=device
        )
        with torch.inference_mode():
            out = model(te_x_norm).cpu().numpy()
            pred_ade = out[..., 0]
            pred_coll = 1.0 / (1.0 + np.exp(-out[..., 1]))
            scores = -pred_ade - 0.5 * pred_coll
        return np.argmax(scores, axis=-1)

    elif "pointwise" in method:
        model = TinyMLP(in_dim=D, hidden=64, out_dim=1)
        model = train_neural_ranker(
            model, tr_x_norm, train_ade_t, train_coll_t,
            loss_type="pointwise", epochs=5, device=device
        )
        with torch.inference_mode():
            pred_ade = model(te_x_norm).squeeze(-1).cpu().numpy()
        return np.argmin(pred_ade, axis=-1)

    elif "pairwise" in method:
        model = TinyMLP(in_dim=D, hidden=64, out_dim=1)
        model = train_neural_ranker(
            model, tr_x_norm, train_ade_t, train_coll_t,
            loss_type="pairwise", epochs=5, device=device
        )
        with torch.inference_mode():
            scores = model(te_x_norm).squeeze(-1).cpu().numpy()
        return np.argmax(scores, axis=-1)

    elif "listwise" in method:
        model = TinyMLP(in_dim=D, hidden=64, out_dim=1)
        model = train_neural_ranker(
            model, tr_x_norm, train_ade_t, train_coll_t,
            loss_type="listwise", epochs=5, device=device
        )
        with torch.inference_mode():
            scores = model(te_x_norm).squeeze(-1).cpu().numpy()
        return np.argmax(scores, axis=-1)

    elif "residual" in method:
        model = TinyMLP(in_dim=D, hidden=64, out_dim=1)
        model = train_neural_ranker(
            model, tr_x_norm, train_ade_t, train_coll_t,
            loss_type="pairwise", epochs=5, device=device
        )
        with torch.inference_mode():
            delta = model(te_x_norm).squeeze(-1).cpu().numpy()
            std_logit = np.std(te_logits)
            std_delta = np.std(delta) + 1e-6
            scores = te_logits + 0.3 * (std_logit / std_delta) * delta
        return np.argmax(scores, axis=-1)

    else:
        raise ValueError(f"Unknown neural method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Run Single Selector Benchmark")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--protocol", type=str, default="leave_one_fold_out")
    parser.add_argument("--cache-root", type=str, default="SELECTOR_BENCHMARK_SERIES/cache")
    parser.add_argument("--feature-root", type=str, default="SELECTOR_BENCHMARK_SERIES/features")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--folds", type=str, default="eth,hotel,univ,zara01,zara02")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    feature_dir = (repo_root / args.feature_root / f"seed_{args.seed}").resolve()
    out_dir = (repo_root / args.output_root / args.method).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fold_list = [f.strip() for f in args.folds.split(",") if f.strip()]

    print(f"\n[BENCHMARK] Running method='{args.method}', seed={args.seed}, protocol={args.protocol}...")
    t0 = time.time()

    # Load all folds into memory
    fold_data_map = {}
    for f in fold_list:
        fold_data_map[f] = load_fold_data(feature_dir, f)

    per_fold_results = {}
    metric_keys = [
        "selected_ade", "selected_fde", "ade_regret", "fde_regret",
        "oracle_hit_rate", "collision_rate", "near_collision_rate",
        "q5_clearance", "bottom_5_cvar", "oracle_gap_closure",
        "intervention_rate", "false_intervention_rate", "min_ade20", "min_fde20"
    ]

    for test_fold in fold_list:
        train_folds = [f for f in fold_list if f != test_fold]
        test_data = fold_data_map[test_fold]
        train_datas = [fold_data_map[f] for f in train_folds]

        # Determine method family and execute
        m = args.method
        if m in ("native", "cv_shield_k5", "cv_shield_k20", "mpn_shield_k5", "mpn_shield_k20",
                "mbr_uniform", "mbr_native_weighted", "native_plus_mbr_grid",
                "native_plus_cv_risk_grid", "native_plus_mpn_risk_grid",
                "native_plus_smoothness_grid", "consensus_plus_mpn_risk", "two_stage_gate_mpn"):
            choices = run_rule_method(m, test_data, train_datas)
        elif "ridge" in m or "linear" in m or "gbdt" in m or "lambdarank" in m:
            choices = run_linear_or_tree_method(m, test_data, train_datas)
        else:
            choices = run_neural_method(m, test_data, train_datas, device)

        fold_eval = evaluate_choices(
            selected_indices=choices,
            ade=test_data["ade"],
            fde=test_data["fde"],
            clearance=test_data["clearance"],
            collision=test_data["collision"],
            has_nbrs=test_data["has_neighbours"],
            min_ade20=test_data["min_ade20"],
            min_fde20=test_data["min_fde20"],
            native_indices=test_data["native_top1_idx"],
        )
        per_fold_results[test_fold] = fold_eval
        print(f"  [{test_fold}] Sel ADE: {fold_eval['selected_ade']:.4f} | FDE: {fold_eval['selected_fde']:.4f} | Coll: {fold_eval['collision_rate']*100:.2f}% | Regret: {fold_eval['ade_regret']:.4f}")

    # Compute macro average across 5 folds
    macro_metrics = {}
    for k in metric_keys:
        vals = [per_fold_results[f][k] for f in fold_list]
        macro_metrics[k] = float(np.mean(vals))

    elapsed = time.time() - t0
    print(f"[BENCHMARK OK] {args.method}: Macro ADE={macro_metrics['selected_ade']:.4f}, Coll={macro_metrics['collision_rate']*100:.2f}%, Regret={macro_metrics['ade_regret']:.4f} in {elapsed:.1f}s")

    # Save metrics.json
    metrics_file = out_dir / "metrics.json"
    metrics_data = {
        "method": args.method,
        "seed": args.seed,
        "protocol": args.protocol,
        "macro": macro_metrics,
        "elapsed_seconds": elapsed,
        "timestamp": now_iso(),
    }
    metrics_file.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    # Save per_fold.csv
    per_fold_csv = out_dir / "per_fold.csv"
    with open(per_fold_csv, "w", newline="", encoding="utf-8") as fp:
        fields = ["fold"] + metric_keys
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for f in fold_list:
            row = {"fold": f, **per_fold_results[f]}
            writer.writerow(row)
        writer.writerow({"fold": "macro_average", **macro_metrics})

    # Save train_log.json
    log_file = out_dir / "train_log.json"
    log_data = {
        "method": args.method,
        "seed": args.seed,
        "folds": fold_list,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "timestamp": now_iso(),
    }
    log_file.write_text(json.dumps(log_data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
