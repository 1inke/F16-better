#!/usr/bin/env python
"""
EXP-T1: Tiny L5 Selector 3-Seed Robustness & Multi-Seed Validation.

Compares:
A. Native (Frozen F16 Top-1)
B. Global Linear Residual (G0 + G1 + G2, Ridge alpha=50.0)
C. Global Transformer Residual (G0 + G1 + G2 + G4, TinyTransformer d_model=48)
D. Tiny L5 Linear Residual (G0 + G1 + raw L5, Ridge alpha=50.0)

Evaluated across Seeds: 1, 2, 3.
Features loaded per seed from SELECTOR_BENCHMARK_SERIES/features/seed_{seed}.
Latents loaded per seed from SELECTOR_RANKABILITY_SERIES/S2V2_CONTEXT_LATENT/latent_cache_v2/seed_{seed}.

Outputs:
- results/tiny_selector_multiseed.csv
- results/tiny_selector_per_fold.csv
- results/predictions_cache.npz (Per-sample predictions across all seeds for bootstrap)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO / "SELECTOR_LOCAL_RERANK_SERIES")]

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
from sklearn.linear_model import Ridge

from common.cache_io import assemble_feature_groups, load_fold_features
from common.reproducibility import set_deterministic_seed, stable_seed
from common.training_config import BATCH_SIZE, LEARNING_RATE, NEURAL_EPOCHS, WEIGHT_DECAY
from common.local_models import ALPHA_GRID, count_parameters
from common.local_metrics import evaluate_local_predictions

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]
SEEDS = [1, 2, 3]


class TinyTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 48, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x)
        h = self.transformer(h)
        return self.head(h).squeeze(-1)  # (B, 20)


def infer_scores_batched(net: nn.Module, x: torch.Tensor, device: torch.device, batch_size: int = 512) -> np.ndarray:
    net.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            bx = x[start : start + batch_size].to(device)
            outputs.append(net(bx).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_transformer_ranker(
    net: nn.Module,
    tr_x: np.ndarray,
    tr_ade: np.ndarray,
    device: torch.device,
    epochs: int = NEURAL_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    mean = np.mean(tr_x, axis=(0, 1), keepdims=True).astype(np.float32)
    std = (np.std(tr_x, axis=(0, 1), keepdims=True) + 1e-6).astype(np.float32)
    tr_norm = torch.from_numpy((tr_x - mean) / std).float()
    tr_ade_t = torch.from_numpy(tr_ade).float()

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(tr_norm, tr_ade_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net.train()
    for _ in range(epochs):
        for bx, bade in loader:
            bx, bade = bx.to(device), bade.to(device)
            optimizer.zero_grad()
            pred = net(bx)

            i_idx = torch.randint(0, 20, (20,), device=device)
            j_idx = torch.randint(0, 20, (20,), device=device)
            diff_score = pred[:, i_idx] - pred[:, j_idx]
            diff_ade = bade[:, i_idx] - bade[:, j_idx]
            valid = torch.abs(diff_ade) > 0.05
            if valid.any():
                label = (diff_ade < 0).float()
                loss = F.binary_cross_entropy_with_logits(diff_score[valid], label[valid])
            else:
                loss = F.mse_loss(pred, -bade)

            loss.backward()
            optimizer.step()

    return net, mean, std


def evaluate_seed_pipeline(
    seed: int,
    feature_root: Path,
    latent_root: Path,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, np.ndarray]]:
    print(f"\n=======================================================")
    print(f"   EVALUATING PIPELINE FOR SEED {seed}")
    print(f"=======================================================")

    feat_dir = feature_root / f"seed_{seed}"
    lat_dir = latent_root / f"seed_{seed}"

    raw_data = {f: load_fold_features(feature_root, f, seed=seed) for f in FOLDS}
    latent_data = {}
    for f in FOLDS:
        lat_npz = lat_dir / f / "latents.npz"
        if not lat_npz.exists():
            raise FileNotFoundError(f"Missing latents: {lat_npz}")
        latent_data[f] = np.load(lat_npz)["l5"]

    methods = ["Native", "Global_Linear", "Global_Transformer", "Tiny_L5_Linear"]
    summary_rows = []
    fold_rows = []

    # Prepare prediction cache storage for this seed
    pred_cache: Dict[str, List] = {
        "fold": [],
        "sample_id": [],
        "native_idx": [],
        "transformer_selected_idx": [],
        "tiny_selected_idx": [],
        "native_ade": [],
        "transformer_ade": [],
        "tiny_ade": [],
        "oracle_ade": [],
    }

    # Pre-build feature tensors per fold
    # 1. Global Linear: G0 + G1 + G2
    x_lin = {f: assemble_feature_groups(raw_data[f], ["g0", "g1", "g2"]) for f in FOLDS}
    # 2. Global Transformer: G0 + G1 + G2 + G4
    x_trans = {f: assemble_feature_groups(raw_data[f], ["g0", "g1", "g2", "g4"]) for f in FOLDS}
    # 3. Tiny L5 Linear: G0 + G1 + raw L5
    x_tiny = {
        f: np.concatenate([assemble_feature_groups(raw_data[f], ["g0", "g1"]), latent_data[f]], axis=-1).astype(np.float32)
        for f in FOLDS
    }

    # Store per-fold selected indices for methods
    fold_selections: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in methods}
    fold_scores_dict: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in methods}
    fold_latencies: Dict[str, float] = {m: 0.0 for m in methods}
    method_params: Dict[str, int] = {"Native": 0}

    # --- 1. Evaluate Native ---
    t0 = time.time()
    for f in FOLDS:
        nat_sel = raw_data[f]["native_top1_idx"]
        fold_selections["Native"][f] = nat_sel
        fold_scores_dict["Native"][f] = raw_data[f]["g0"][:, :, 0]
    fold_latencies["Native"] = (time.time() - t0) * 1000.0 / sum(len(raw_data[f]["ade"]) for f in FOLDS)

    # --- 2. Evaluate Global Linear Residual ---
    t0 = time.time()
    for test_fold in FOLDS:
        train_folds = [f for f in FOLDS if f != test_fold]
        inner_val_fold = train_folds[-1]
        inner_tr_folds = train_folds[:-1]

        in_tr_x = np.concatenate([x_lin[f] for f in inner_tr_folds], axis=0)
        in_tr_ade = np.concatenate([raw_data[f]["ade"] for f in inner_tr_folds], axis=0)
        in_tr_log = np.concatenate([raw_data[f]["g0"][:, :, 0] for f in inner_tr_folds], axis=0)

        in_val_x = x_lin[inner_val_fold]
        in_val_ade = raw_data[inner_val_fold]["ade"]
        in_val_log = raw_data[inner_val_fold]["g0"][:, :, 0]

        N_in_tr, K, D = in_tr_x.shape
        N_in_val = len(in_val_x)

        m = Ridge(alpha=50.0)
        m.fit(in_tr_x.reshape(-1, D), -in_tr_ade.reshape(-1))
        val_delta = m.predict(in_val_x.reshape(-1, D)).reshape(N_in_val, K)
        tr_delta = m.predict(in_tr_x.reshape(-1, D)).reshape(N_in_tr, K)
        scale_in = np.std(in_tr_log) / (np.std(tr_delta) + 1e-6)

        best_a = 0.40
        best_v = float("inf")
        for a in ALPHA_GRID:
            sc = in_val_log + a * scale_in * val_delta
            sel = np.argmax(sc, axis=-1)
            v = float(np.mean(in_val_ade[np.arange(N_in_val), sel]))
            if v < best_v:
                best_v = v
                best_a = a

        full_tr_x = np.concatenate([x_lin[f] for f in train_folds], axis=0)
        full_tr_ade = np.concatenate([raw_data[f]["ade"] for f in train_folds], axis=0)
        full_tr_log = np.concatenate([raw_data[f]["g0"][:, :, 0] for f in train_folds], axis=0)

        te_x = x_lin[test_fold]
        N_tr = len(full_tr_x)
        N_te = len(te_x)

        m_out = Ridge(alpha=50.0)
        m_out.fit(full_tr_x.reshape(-1, D), -full_tr_ade.reshape(-1))
        tr_d_out = m_out.predict(full_tr_x.reshape(-1, D)).reshape(N_tr, K)
        te_d_out = m_out.predict(te_x.reshape(-1, D)).reshape(N_te, K)
        scale_out = np.std(full_tr_log) / (np.std(tr_d_out) + 1e-6)

        te_scores = raw_data[test_fold]["g0"][:, :, 0] + best_a * scale_out * te_d_out
        fold_selections["Global_Linear"][test_fold] = np.argmax(te_scores, axis=-1)
        fold_scores_dict["Global_Linear"][test_fold] = te_scores
        method_params["Global_Linear"] = D + 1
    fold_latencies["Global_Linear"] = (time.time() - t0) * 1000.0 / sum(len(raw_data[f]["ade"]) for f in FOLDS)

    # --- 3. Evaluate Global Transformer Residual ---
    t0 = time.time()
    for test_fold in FOLDS:
        train_folds = [f for f in FOLDS if f != test_fold]
        inner_val_fold = train_folds[-1]
        inner_tr_folds = train_folds[:-1]

        inner_seed = stable_seed(seed, "Global_Transformer", test_fold, inner_val_fold, "inner")
        set_deterministic_seed(inner_seed)

        in_tr_x = np.concatenate([x_trans[f] for f in inner_tr_folds], axis=0)
        in_tr_ade = np.concatenate([raw_data[f]["ade"] for f in inner_tr_folds], axis=0)
        in_tr_log = np.concatenate([raw_data[f]["g0"][:, :, 0] for f in inner_tr_folds], axis=0)

        in_val_x = x_trans[inner_val_fold]
        in_val_ade = raw_data[inner_val_fold]["ade"]
        in_val_log = raw_data[inner_val_fold]["g0"][:, :, 0]

        N_in_tr, K, D = in_tr_x.shape
        N_in_val = len(in_val_x)

        net_in = TinyTransformer(in_dim=D, d_model=48, nhead=4).to(device)
        net_in, mean_in, std_in = train_transformer_ranker(net_in, in_tr_x, in_tr_ade, device, epochs=5)

        tr_norm = torch.from_numpy((in_tr_x - mean_in) / std_in).float()
        val_norm = torch.from_numpy((in_val_x - mean_in) / std_in).float()
        tr_delta = infer_scores_batched(net_in, tr_norm, device)
        val_delta = infer_scores_batched(net_in, val_norm, device)
        scale_in = np.std(in_tr_log) / (np.std(tr_delta) + 1e-6)

        best_a = 0.50
        best_v = float("inf")
        for a in ALPHA_GRID:
            sc = in_val_log + a * scale_in * val_delta
            sel = np.argmax(sc, axis=-1)
            v = float(np.mean(in_val_ade[np.arange(N_in_val), sel]))
            if v < best_v:
                best_v = v
                best_a = a

        # Outer Training
        outer_seed = stable_seed(seed, "Global_Transformer", test_fold, "outer")
        set_deterministic_seed(outer_seed)

        full_tr_x = np.concatenate([x_trans[f] for f in train_folds], axis=0)
        full_tr_ade = np.concatenate([raw_data[f]["ade"] for f in train_folds], axis=0)
        full_tr_log = np.concatenate([raw_data[f]["g0"][:, :, 0] for f in train_folds], axis=0)

        te_x = x_trans[test_fold]
        N_tr = len(full_tr_x)
        N_te = len(te_x)

        net_out = TinyTransformer(in_dim=D, d_model=48, nhead=4).to(device)
        net_out, mean_out, std_out = train_transformer_ranker(net_out, full_tr_x, full_tr_ade, device)

        tr_norm_out = torch.from_numpy((full_tr_x - mean_out) / std_out).float()
        te_norm_out = torch.from_numpy((te_x - mean_out) / std_out).float()
        tr_d_out = infer_scores_batched(net_out, tr_norm_out, device)
        te_d_out = infer_scores_batched(net_out, te_norm_out, device)
        scale_out = np.std(full_tr_log) / (np.std(tr_d_out) + 1e-6)

        te_scores = raw_data[test_fold]["g0"][:, :, 0] + best_a * scale_out * te_d_out
        fold_selections["Global_Transformer"][test_fold] = np.argmax(te_scores, axis=-1)
        fold_scores_dict["Global_Transformer"][test_fold] = te_scores
        method_params["Global_Transformer"] = count_parameters(net_out)
    fold_latencies["Global_Transformer"] = (time.time() - t0) * 1000.0 / sum(len(raw_data[f]["ade"]) for f in FOLDS)

    # --- 4. Evaluate Tiny L5 Linear Residual ---
    t0 = time.time()
    for test_fold in FOLDS:
        train_folds = [f for f in FOLDS if f != test_fold]
        inner_val_fold = train_folds[-1]
        inner_tr_folds = train_folds[:-1]

        in_tr_x = np.concatenate([x_tiny[f] for f in inner_tr_folds], axis=0)
        in_tr_ade = np.concatenate([raw_data[f]["ade"] for f in inner_tr_folds], axis=0)
        in_tr_log = np.concatenate([raw_data[f]["g0"][:, :, 0] for f in inner_tr_folds], axis=0)

        in_val_x = x_tiny[inner_val_fold]
        in_val_ade = raw_data[inner_val_fold]["ade"]
        in_val_log = raw_data[inner_val_fold]["g0"][:, :, 0]

        N_in_tr, K, D = in_tr_x.shape
        N_in_val = len(in_val_x)

        m = Ridge(alpha=50.0)
        m.fit(in_tr_x.reshape(-1, D), -in_tr_ade.reshape(-1))
        val_delta = m.predict(in_val_x.reshape(-1, D)).reshape(N_in_val, K)
        tr_delta = m.predict(in_tr_x.reshape(-1, D)).reshape(N_in_tr, K)
        scale_in = np.std(in_tr_log) / (np.std(tr_delta) + 1e-6)

        best_a = 0.30
        best_v = float("inf")
        for a in ALPHA_GRID:
            sc = in_val_log + a * scale_in * val_delta
            sel = np.argmax(sc, axis=-1)
            v = float(np.mean(in_val_ade[np.arange(N_in_val), sel]))
            if v < best_v:
                best_v = v
                best_a = a

        full_tr_x = np.concatenate([x_tiny[f] for f in train_folds], axis=0)
        full_tr_ade = np.concatenate([raw_data[f]["ade"] for f in train_folds], axis=0)
        full_tr_log = np.concatenate([raw_data[f]["g0"][:, :, 0] for f in train_folds], axis=0)

        te_x = x_tiny[test_fold]
        N_tr = len(full_tr_x)
        N_te = len(te_x)

        m_out = Ridge(alpha=50.0)
        m_out.fit(full_tr_x.reshape(-1, D), -full_tr_ade.reshape(-1))
        tr_d_out = m_out.predict(full_tr_x.reshape(-1, D)).reshape(N_tr, K)
        te_d_out = m_out.predict(te_x.reshape(-1, D)).reshape(N_te, K)
        scale_out = np.std(full_tr_log) / (np.std(tr_d_out) + 1e-6)

        te_scores = raw_data[test_fold]["g0"][:, :, 0] + best_a * scale_out * te_d_out
        fold_selections["Tiny_L5_Linear"][test_fold] = np.argmax(te_scores, axis=-1)
        fold_scores_dict["Tiny_L5_Linear"][test_fold] = te_scores
        method_params["Tiny_L5_Linear"] = D + 1
    fold_latencies["Tiny_L5_Linear"] = (time.time() - t0) * 1000.0 / sum(len(raw_data[f]["ade"]) for f in FOLDS)

    # --- Populate Prediction Cache across samples ---
    for f in FOLDS:
        d = raw_data[f]
        N_f = len(d["ade"])
        sample_ids = [f"{f}_{i}" for i in range(N_f)]
        nat_s = fold_selections["Native"][f]
        trans_s = fold_selections["Global_Transformer"][f]
        tiny_s = fold_selections["Tiny_L5_Linear"][f]

        sample_range = np.arange(N_f)
        pred_cache["fold"].extend([f] * N_f)
        pred_cache["sample_id"].extend(sample_ids)
        pred_cache["native_idx"].extend(nat_s.tolist())
        pred_cache["transformer_selected_idx"].extend(trans_s.tolist())
        pred_cache["tiny_selected_idx"].extend(tiny_s.tolist())
        pred_cache["native_ade"].extend(d["ade"][sample_range, nat_s].tolist())
        pred_cache["transformer_ade"].extend(d["ade"][sample_range, trans_s].tolist())
        pred_cache["tiny_ade"].extend(d["ade"][sample_range, tiny_s].tolist())
        pred_cache["oracle_ade"].extend(d["min_ade20"].tolist())

    # --- Evaluate Metrics per Method (Macro & Per-Fold) ---
    for m in methods:
        ev_list = []
        sample_counts = []
        ade_sums = []

        for f in FOLDS:
            d = raw_data[f]
            N_f = len(d["ade"])
            sample_counts.append(N_f)
            sel = fold_selections[m][f]
            scores = fold_scores_dict[m][f]

            ev = evaluate_local_predictions(
                selected_indices=sel,
                topk_indices=np.tile(np.arange(20), (N_f, 1)),
                topk_scores=scores,
                full_ade=d["ade"],
                full_fde=d["fde"],
                full_collision=d["collision"],
                native_indices=d["native_top1_idx"],
                has_neighbors=d.get("has_neighbours"),
            )
            ev_list.append(ev)
            ade_sums.append(float(np.sum(d["ade"][np.arange(N_f), sel])))

            # Fold row
            f_row = {
                "Method": m,
                "Seed": seed,
                "Fold": f,
                "Selected_ADE": f"{ev['selected_ade']:.4f}",
                "Selected_FDE": f"{ev['selected_fde']:.4f}",
                "ADE_Regret": f"{ev['ade_regret']:.4f}",
                "Oracle_Hit": f"{ev['oracle_hit']*100:.2f}%",
                "NearOracle05": f"{ev['near_oracle_05']*100:.2f}%",
                "Pairwise10": f"{ev['local_pairwise_10']*100:.2f}%",
                "Collision_Social": f"{ev['collision_social']*100:.2f}%",
                "Intervention_Rate": f"{ev['intervention_rate']*100:.2f}%",
                "False_Intervention": f"{ev['false_intervention_rate']*100:.2f}%",
            }
            fold_rows.append(f_row)

        macro_ev = {k: float(np.mean([e[k] for e in ev_list])) for k in ev_list[0]}
        micro_ade = float(sum(ade_sums) / sum(sample_counts))

        print(
            f"  [{m}] Macro ADE: {macro_ev['selected_ade']:.4f}m | Micro ADE: {micro_ade:.4f}m | "
            f"Regret: {macro_ev['ade_regret']:.4f}m | Pairwise@10: {macro_ev['local_pairwise_10']*100:.2f}% | "
            f"Social Coll: {macro_ev['collision_social']*100:.2f}%"
        )

        sum_row = {
            "Method": m,
            "Seed": seed,
            "ADE_Macro_m": f"{macro_ev['selected_ade']:.4f}",
            "ADE_Micro_m": f"{micro_ade:.4f}",
            "FDE_Macro_m": f"{macro_ev['selected_fde']:.4f}",
            "ADE_Regret_m": f"{macro_ev['ade_regret']:.4f}",
            "Oracle_Hit": f"{macro_ev['oracle_hit']*100:.2f}%",
            "NearOracle05": f"{macro_ev['near_oracle_05']*100:.2f}%",
            "Pairwise10": f"{macro_ev['local_pairwise_10']*100:.2f}%",
            "Collision_Social": f"{macro_ev['collision_social']*100:.2f}%",
            "Intervention_Rate": f"{macro_ev['intervention_rate']*100:.2f}%",
            "False_Intervention": f"{macro_ev['false_intervention_rate']*100:.2f}%",
            "Latency_ms": f"{fold_latencies[m]:.3f}",
            "Params": str(method_params[m]),
        }
        summary_rows.append(sum_row)

    return summary_rows, fold_rows, pred_cache


def main():
    parser = argparse.ArgumentParser(description="EXP-T1: Tiny L5 Selector 3-Seed Validation")
    parser.add_argument("--feature-root", type=str, default="SELECTOR_BENCHMARK_SERIES/features")
    parser.add_argument("--latent-root", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V2_CONTEXT_LATENT/latent_cache_v2")
    parser.add_argument("--output-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V3_CENTERED_LATENT/results")
    args = parser.parse_args()

    feat_root = (REPO / args.feature_root).resolve()
    lat_root = (REPO / args.latent_root).resolve()
    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-T1] Starting Tiny L5 vs Transformer 3-Seed Benchmark on {device}...")

    all_summaries = []
    all_folds = []
    all_predictions: Dict[str, Any] = {}

    for s in SEEDS:
        seed_dump_file = out_dir / f"seed_{s}_checkpoint.json"
        seed_npz_file = out_dir / f"seed_{s}_pred.npz"

        if seed_dump_file.exists() and seed_npz_file.exists():
            print(f"\n[CACHE HIT] Loading cached evaluation for Seed {s}...")
            with open(seed_dump_file, "r", encoding="utf-8") as fp:
                cached_data = json.load(fp)
            sum_rows = cached_data["summaries"]
            f_rows = cached_data["folds"]
            cached_npz = np.load(seed_npz_file)
            pred_cache = {k: cached_npz[k] for k in cached_npz.files}
        else:
            sum_rows, f_rows, pred_cache = evaluate_seed_pipeline(
                seed=s,
                feature_root=feat_root,
                latent_root=lat_root,
                device=device,
            )
            # Save checkpoint
            with open(seed_dump_file, "w", encoding="utf-8") as fp:
                json.dump({"summaries": sum_rows, "folds": f_rows}, fp, indent=2)
            np.savez_compressed(seed_npz_file, **pred_cache)

        all_summaries.extend(sum_rows)
        all_folds.extend(f_rows)
        for k, v in pred_cache.items():
            cache_key = f"seed_{s}_{k}"
            all_predictions[cache_key] = np.array(v)

    # Compute 3-Seed Mean and Std for each method
    methods = ["Native", "Global_Linear", "Global_Transformer", "Tiny_L5_Linear"]
    metric_cols = [
        "ADE_Macro_m", "ADE_Micro_m", "FDE_Macro_m", "ADE_Regret_m",
        "Oracle_Hit", "NearOracle05", "Pairwise10", "Collision_Social",
        "Intervention_Rate", "False_Intervention", "Latency_ms"
    ]

    mean_std_rows = []
    for m in methods:
        m_rows = [r for r in all_summaries if r["Method"] == m]
        mean_std_row = {"Method": f"{m}_3seed_mean_std", "Seed": "all", "Params": m_rows[0]["Params"]}
        for col in metric_cols:
            vals = [float(r[col].replace("%", "")) for r in m_rows]
            avg = np.mean(vals)
            std = np.std(vals)
            if "%" in m_rows[0][col]:
                mean_std_row[col] = f"{avg:.2f}% +/- {std:.2f}%"
            else:
                mean_std_row[col] = f"{avg:.4f} +/- {std:.4f}"
        mean_std_rows.append(mean_std_row)

    # Save summary CSV
    sum_csv_path = out_dir / "tiny_selector_multiseed.csv"
    with open(sum_csv_path, "w", newline="", encoding="utf-8") as fp:
        fieldnames = list(all_summaries[0].keys())
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_summaries)
        w.writerows(mean_std_rows)

    # Save per-fold CSV
    fold_csv_path = out_dir / "tiny_selector_per_fold.csv"
    with open(fold_csv_path, "w", newline="", encoding="utf-8") as fp:
        fieldnames = list(all_folds[0].keys())
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_folds)

    # Save predictions cache NPZ
    pred_cache_path = out_dir / "predictions_cache.npz"
    np.savez_compressed(pred_cache_path, **all_predictions)

    print(f"\n[EXP-T1 OK] Saved Multi-seed Summary -> {sum_csv_path}")
    print(f"            Saved Per-Fold Summary   -> {fold_csv_path}")
    print(f"            Saved Predictions Cache  -> {pred_cache_path}")


if __name__ == "__main__":
    main()
