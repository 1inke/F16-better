#!/usr/bin/env python
"""
EXP-S2V2D: Local Latent Probe for Top-K Fine Discrimination.

Investigates whether context-conditioned decoder latents (L4, L5) enhance fine discrimination
within local Top-K pools (K in {3, 5, 10}):
1. Feature Sets:
   - G0 + GL + L4
   - G0 + GL + L5
   - G0 + G1 + GL + L5
2. Models:
   - Linear Residual (Ridge alpha=50.0)
   - MLP Residual (LocalMLP hidden=64)
   - Tiny Set Attention (TinySetAttention d_model=64, nhead=4)
3. Target Subpools:
   - K = 3, 5, 10
4. Key Discrimination Metrics:
   - Local Pairwise@5cm (Native baseline ~64%)
   - Local Pairwise@10cm
   - Local Spearman Correlation
   - Selected ADE (Macro / Micro)
   - Oracle Hit & Near-Oracle@5cm
   - Useful / False Intervention Rates

Output:
- results/local_latent_probe.csv
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
from sklearn.linear_model import Ridge

from common.cache_io import load_fold_features
from common.local_pool import build_native_topk_pool
from common.local_features import assemble_local_features
from common.reproducibility import set_deterministic_seed, stable_seed
from common.training_config import BATCH_SIZE, LEARNING_RATE, NEURAL_EPOCHS, WEIGHT_DECAY
from common.local_models import (
    ALPHA_GRID,
    LocalMLP,
    TinySetAttention,
    count_parameters,
    infer_batched,
    train_neural_local_ranker,
)
from common.local_metrics import evaluate_local_predictions

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]

K_VALUES = [3, 5, 10]

LOCAL_FEATURE_CONFIGS = [
    (["g0", "gl"], "l4", "G0_GL_L4"),
    (["g0", "gl"], "l5", "G0_GL_L5"),
    (["g0", "g1", "gl"], "l5", "G0_G1_GL_L5"),
]

MODEL_TYPES = ["linear_residual", "mlp_residual", "tiny_set_attention"]


def load_fold_latents_v2(latent_dir: Path, fold: str) -> Dict[str, np.ndarray]:
    npz_path = latent_dir / fold / "latents.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing latents: {npz_path}")
    data = np.load(npz_path)
    return {
        "l3": data["l3"],
        "l4": data["l4"],
        "l5": data["l5"],
        "native_logits": data["native_logits"],
    }


def build_local_probe_features(
    raw_fold_data: Dict[str, np.ndarray],
    latent_fold_data: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    base_groups: List[str],
    latent_key: str,
) -> np.ndarray:
    base_x = assemble_local_features(raw_fold_data, pool, base_groups) # (N, K, D_base)
    lat_full = latent_fold_data[latent_key] # (N, 20, D_lat)
    topk_indices = pool["indices"] # (N, K)
    topk_lat = np.take_along_axis(lat_full, topk_indices[:, :, None], axis=1) # (N, K, D_lat)
    return np.concatenate([base_x, topk_lat], axis=-1).astype(np.float32)


def evaluate_local_probe_config(
    k: int,
    base_groups: List[str],
    latent_key: str,
    feat_name: str,
    model_type: str,
    raw_data: Dict[str, Any],
    latent_data: Dict[str, Any],
    base_seed: int,
    device: torch.device,
) -> Tuple[Dict[str, float], float, int, float]:
    cfg_id = f"K{k}_{model_type}_{feat_name}"
    pools = {f: build_native_topk_pool(raw_data[f], k) for f in FOLDS}
    features = {
        f: build_local_probe_features(raw_data[f], latent_data[f], pools[f], base_groups, latent_key)
        for f in FOLDS
    }

    t0 = time.time()
    fold_evals = []
    fold_alphas = []
    param_count = 0

    for test_fold in FOLDS:
        train_folds = [f for f in FOLDS if f != test_fold]
        inner_val_fold = train_folds[-1]
        inner_tr_folds = train_folds[:-1]

        inner_seed = stable_seed(base_seed, cfg_id, test_fold, inner_val_fold, "inner")
        set_deterministic_seed(inner_seed)

        in_tr_x = np.concatenate([features[f] for f in inner_tr_folds], axis=0)
        in_tr_logits = np.concatenate([pools[f]["logits"] for f in inner_tr_folds], axis=0)
        in_tr_ade = np.concatenate([pools[f]["ade"] for f in inner_tr_folds], axis=0)

        in_val_x = features[inner_val_fold]
        in_val_logits = pools[inner_val_fold]["logits"]
        in_val_ade = pools[inner_val_fold]["ade"]

        N_in_tr, _, D = in_tr_x.shape
        N_in_val = len(in_val_x)

        if model_type == "linear_residual":
            m_in = Ridge(alpha=50.0)
            m_in.fit(in_tr_x.reshape(-1, D), -in_tr_ade.reshape(-1))
            val_delta = m_in.predict(in_val_x.reshape(-1, D)).reshape(N_in_val, k)
            tr_delta = m_in.predict(in_tr_x.reshape(-1, D)).reshape(N_in_tr, k)
            scale_in = np.std(in_tr_logits) / (np.std(tr_delta) + 1e-6)
        elif model_type == "mlp_residual":
            net_in = LocalMLP(in_dim=D, hidden=64).to(device)
            net_in, mean_in, std_in = train_neural_local_ranker(
                net_in, in_tr_x, in_tr_ade, device,
                epochs=NEURAL_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            )
            tr_norm = torch.from_numpy((in_tr_x - mean_in) / std_in).float()
            val_norm = torch.from_numpy((in_val_x - mean_in) / std_in).float()
            tr_delta = infer_batched(net_in, tr_norm, device)
            val_delta = infer_batched(net_in, val_norm, device)
            scale_in = np.std(in_tr_logits) / (np.std(tr_delta) + 1e-6)
        else:
            net_in = TinySetAttention(in_dim=D, d_model=64, nhead=4).to(device)
            net_in, mean_in, std_in = train_neural_local_ranker(
                net_in, in_tr_x, in_tr_ade, device,
                epochs=NEURAL_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            )
            tr_norm = torch.from_numpy((in_tr_x - mean_in) / std_in).float()
            val_norm = torch.from_numpy((in_val_x - mean_in) / std_in).float()
            tr_delta = infer_batched(net_in, tr_norm, device)
            val_delta = infer_batched(net_in, val_norm, device)
            scale_in = np.std(in_tr_logits) / (np.std(tr_delta) + 1e-6)

        best_alpha = 0.50
        best_val_ade = float("inf")
        sample_idx = np.arange(N_in_val)
        for a in ALPHA_GRID:
            scores = in_val_logits + a * scale_in * val_delta
            sel = np.argmax(scores, axis=-1)
            v_ade = float(np.mean(in_val_ade[sample_idx, sel]))
            if v_ade < best_val_ade:
                best_val_ade = v_ade
                best_alpha = a
        fold_alphas.append(best_alpha)

        # Outer Training
        outer_seed = stable_seed(base_seed, cfg_id, test_fold, "outer")
        set_deterministic_seed(outer_seed)

        full_tr_x = np.concatenate([features[f] for f in train_folds], axis=0)
        full_tr_logits = np.concatenate([pools[f]["logits"] for f in train_folds], axis=0)
        full_tr_ade = np.concatenate([pools[f]["ade"] for f in train_folds], axis=0)

        te_x = features[test_fold]
        te_logits = pools[test_fold]["logits"]
        N_tr = len(full_tr_x)
        N_te = len(te_x)

        if model_type == "linear_residual":
            m_out = Ridge(alpha=50.0)
            m_out.fit(full_tr_x.reshape(-1, D), -full_tr_ade.reshape(-1))
            tr_delta_out = m_out.predict(full_tr_x.reshape(-1, D)).reshape(N_tr, k)
            te_delta_out = m_out.predict(te_x.reshape(-1, D)).reshape(N_te, k)
            scale_out = np.std(full_tr_logits) / (np.std(tr_delta_out) + 1e-6)
            param_count = D + 1
        elif model_type == "mlp_residual":
            net_out = LocalMLP(in_dim=D, hidden=64).to(device)
            net_out, mean_out, std_out = train_neural_local_ranker(
                net_out, full_tr_x, full_tr_ade, device,
                epochs=NEURAL_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            )
            tr_norm_out = torch.from_numpy((full_tr_x - mean_out) / std_out).float()
            te_norm_out = torch.from_numpy((te_x - mean_out) / std_out).float()
            tr_delta_out = infer_batched(net_out, tr_norm_out, device)
            te_delta_out = infer_batched(net_out, te_norm_out, device)
            scale_out = np.std(full_tr_logits) / (np.std(tr_delta_out) + 1e-6)
            param_count = count_parameters(net_out)
        else:
            net_out = TinySetAttention(in_dim=D, d_model=64, nhead=4).to(device)
            net_out, mean_out, std_out = train_neural_local_ranker(
                net_out, full_tr_x, full_tr_ade, device,
                epochs=NEURAL_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            )
            tr_norm_out = torch.from_numpy((full_tr_x - mean_out) / std_out).float()
            te_norm_out = torch.from_numpy((te_x - mean_out) / std_out).float()
            tr_delta_out = infer_batched(net_out, tr_norm_out, device)
            te_delta_out = infer_batched(net_out, te_norm_out, device)
            scale_out = np.std(full_tr_logits) / (np.std(tr_delta_out) + 1e-6)
            param_count = count_parameters(net_out)

        te_scores = te_logits + best_alpha * scale_out * te_delta_out
        te_topk_indices = pools[test_fold]["indices"]
        te_local_sel = np.argmax(te_scores, axis=-1)
        te_sel_indices = te_topk_indices[np.arange(N_te), te_local_sel]

        fold_eval = evaluate_local_predictions(
            selected_indices=te_sel_indices,
            topk_indices=te_topk_indices,
            topk_scores=te_scores,
            full_ade=raw_data[test_fold]["ade"],
            full_fde=raw_data[test_fold]["fde"],
            full_collision=raw_data[test_fold]["collision"],
            native_indices=raw_data[test_fold]["native_top1_idx"],
            has_neighbors=raw_data[test_fold].get("has_neighbours"),
        )
        fold_evals.append(fold_eval)

    macro = {m: float(np.mean([e[m] for e in fold_evals])) for m in fold_evals[0]}
    mean_alpha = float(np.mean(fold_alphas)) if fold_alphas else 0.0
    total_samples = sum(len(raw_data[f]["ade"]) for f in FOLDS)
    latency_ms = (time.time() - t0) * 1000.0 / total_samples
    return macro, mean_alpha, param_count, latency_ms


def main():
    parser = argparse.ArgumentParser(description="EXP-S2V2D: Local Latent Probe")
    parser.add_argument("--feature-root", type=str, default="SELECTOR_BENCHMARK_SERIES/features")
    parser.add_argument("--latent-root", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V2_CONTEXT_LATENT/latent_cache_v2/seed_1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V2_CONTEXT_LATENT/results")
    args = parser.parse_args()

    feature_root = (REPO / args.feature_root).resolve()
    latent_dir = (REPO / args.latent_root).resolve()
    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-S2V2D] Starting Local Latent Probes on {device}...")

    raw_data = {}
    for fold in FOLDS:
        d = load_fold_features(feature_root, fold, seed=args.seed)
        cand_path = feature_root / f"seed_{args.seed}" / fold / "candidates.npy"
        if cand_path.exists():
            d["candidates"] = np.load(cand_path)
        raw_data[fold] = d

    latent_data = {f: load_fold_latents_v2(latent_dir, f) for f in FOLDS}

    rows = []
    csv_path = out_dir / "local_latent_probe.csv"
    done_cfgs = set()

    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as fp:
                rows = list(csv.DictReader(fp))
                done_cfgs = {r["Config"] for r in rows}
        except Exception:
            rows = []
            done_cfgs = set()

    for k in K_VALUES:
        for base_groups, lat_key, feat_name in LOCAL_FEATURE_CONFIGS:
            for m_type in MODEL_TYPES:
                cfg_id = f"K{k}_{m_type}_{feat_name}"
                if cfg_id in done_cfgs:
                    print(f"\n--- Skipping completed {cfg_id} ---")
                    continue

                print(f"\n--- Evaluating {cfg_id} ---")
                macro, mean_alpha, p_count, lat_ms = evaluate_local_probe_config(
                    k=k,
                    base_groups=base_groups,
                    latent_key=lat_key,
                    feat_name=feat_name,
                    model_type=m_type,
                    raw_data=raw_data,
                    latent_data=latent_data,
                    base_seed=args.seed,
                    device=device,
                )

                print(
                    f"  ADE: {macro['selected_ade']:.4f}m | Regret: {macro['ade_regret']:.4f}m | "
                    f"Pairwise@5cm: {macro['local_pairwise_05']*100:.2f}% | Spearman: {macro['local_spearman']:.4f}"
                )

                row = {
                    "Config": cfg_id,
                    "K": str(k),
                    "Model": m_type,
                    "Feature": feat_name,
                    "Latent": lat_key,
                    "ADE_Macro_m": f"{macro['selected_ade']:.4f}",
                    "FDE_Macro_m": f"{macro['selected_fde']:.4f}",
                    "ADE_Regret_m": f"{macro['ade_regret']:.4f}",
                    "Oracle_Hit": f"{macro['oracle_hit']*100:.2f}%",
                    "NearOracle05": f"{macro['near_oracle_05']*100:.2f}%",
                    "LocalPairwise05": f"{macro['local_pairwise_05']*100:.2f}%",
                    "LocalPairwise10": f"{macro['local_pairwise_10']*100:.2f}%",
                    "LocalSpearman": f"{macro['local_spearman']:.4f}",
                    "Intervention_Rate": f"{macro['intervention_rate']*100:.2f}%",
                    "Useful_Intervention": f"{macro['useful_intervention_rate']*100:.2f}%",
                    "False_Intervention": f"{macro['false_intervention_rate']*100:.2f}%",
                    "Collision_Social": f"{macro['collision_social']*100:.2f}%",
                    "Mean_Alpha": f"{mean_alpha:.2f}",
                    "Latency_ms": f"{lat_ms:.3f}",
                    "Params": str(p_count),
                }
                rows.append(row)

                # Incremental flush
                with open(csv_path, "w", newline="", encoding="utf-8") as fp:
                    w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    print(f"\n[EXP-S2V2D OK] Complete -> {csv_path}")


if __name__ == "__main__":
    main()
