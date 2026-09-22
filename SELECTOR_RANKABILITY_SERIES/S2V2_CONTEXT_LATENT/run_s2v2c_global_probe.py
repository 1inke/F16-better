#!/usr/bin/env python
"""
EXP-S2V2C: Global Latent Probe & Latent Feature Ablation.

Evaluates whether context-conditioned internal representations (L3, L4, L5) provide
meaningful ranking signal across the global 20-candidate pool:
1. Feature Sets:
   - G0 (Native baseline)
   - G0 + L3 (Mode Latent)
   - G0 + L4 (Refiner Latent)
   - G0 + L5 (Pre-Logit Latent)
   - G0 + G1 + L4
   - G0 + G1 + L5
   - G0 + G1 + G4 + L5
   - [Ablation] L5_only
   - [Ablation] G0 + G4 + L5
2. Models:
   - Linear Residual (Ridge alpha=50.0)
   - MLP Residual (LocalMLP hidden=64)
3. Evaluation Protocol:
   - Strict Nested CV: Outer 4 train / 1 test, Inner 3 train / 1 val
   - Stable deterministic seed per fold/cfg/phase
   - Alpha grid: [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50]

Outputs:
- results/global_latent_probe.csv
- results/latent_ablation.csv
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

from common.cache_io import assemble_feature_groups, load_fold_features
from common.reproducibility import set_deterministic_seed, stable_seed
from common.training_config import BATCH_SIZE, LEARNING_RATE, NEURAL_EPOCHS, WEIGHT_DECAY
from common.local_models import (
    ALPHA_GRID,
    LocalMLP,
    count_parameters,
    infer_batched,
    train_neural_local_ranker,
)
from common.local_metrics import evaluate_local_predictions

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]

PROBE_CONFIGS = [
    # Feature group list, latent key, config name
    (["g0"], None, "G0_native"),
    (["g0"], "l3", "G0_L3"),
    (["g0"], "l4", "G0_L4"),
    (["g0"], "l5", "G0_L5"),
    (["g0", "g1"], "l4", "G0_G1_L4"),
    (["g0", "g1"], "l5", "G0_G1_L5"),
    (["g0", "g1", "g4"], "l5", "G0_G1_G4_L5"),
    ([], "l5", "L5_only"),
    (["g0", "g4"], "l5", "G0_G4_L5"),
]


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


def build_probe_features(
    raw_fold_data: Dict[str, np.ndarray],
    latent_fold_data: Dict[str, np.ndarray],
    feat_groups: List[str],
    latent_key: str | None,
) -> np.ndarray:
    parts = []
    if feat_groups:
        parts.append(assemble_feature_groups(raw_fold_data, feat_groups))
    if latent_key is not None:
        parts.append(latent_fold_data[latent_key])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def evaluate_probe_model(
    cfg_name: str,
    feat_groups: List[str],
    latent_key: str | None,
    model_type: str, # "linear_residual" or "mlp_residual"
    raw_data: Dict[str, Any],
    latent_data: Dict[str, Any],
    base_seed: int,
    device: torch.device,
) -> Tuple[Dict[str, float], float, int]:
    # Build per-fold features
    fold_x = {
        f: build_probe_features(raw_data[f], latent_data[f], feat_groups, latent_key)
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

        # Inner CV for alpha tuning
        inner_seed = stable_seed(base_seed, cfg_name, model_type, test_fold, inner_val_fold, "inner")
        set_deterministic_seed(inner_seed)

        in_tr_x = np.concatenate([fold_x[f] for f in inner_tr_folds], axis=0)
        in_tr_ade = np.concatenate([raw_data[f]["ade"] for f in inner_tr_folds], axis=0)
        in_tr_logits = np.concatenate([raw_data[f]["g0"][:, :, 0] for f in inner_tr_folds], axis=0)

        in_val_x = fold_x[inner_val_fold]
        in_val_ade = raw_data[inner_val_fold]["ade"]
        in_val_logits = raw_data[inner_val_fold]["g0"][:, :, 0]

        N_in_tr, K, D = in_tr_x.shape
        N_in_val = len(in_val_x)

        if cfg_name == "G0_native":
            best_alpha = 0.0
            scale_out = 1.0
            te_delta_out = np.zeros_like(raw_data[test_fold]["ade"])
        else:
            if model_type == "linear_residual":
                m_in = Ridge(alpha=50.0)
                m_in.fit(in_tr_x.reshape(-1, D), -in_tr_ade.reshape(-1))
                val_delta = m_in.predict(in_val_x.reshape(-1, D)).reshape(N_in_val, K)
                tr_delta = m_in.predict(in_tr_x.reshape(-1, D)).reshape(N_in_tr, K)
                scale_in = np.std(in_tr_logits) / (np.std(tr_delta) + 1e-6)
            else:
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

            # Outer Model Training
            outer_seed = stable_seed(base_seed, cfg_name, model_type, test_fold, "outer")
            set_deterministic_seed(outer_seed)

            full_tr_x = np.concatenate([fold_x[f] for f in train_folds], axis=0)
            full_tr_ade = np.concatenate([raw_data[f]["ade"] for f in train_folds], axis=0)
            full_tr_logits = np.concatenate([raw_data[f]["g0"][:, :, 0] for f in train_folds], axis=0)

            te_x = fold_x[test_fold]
            N_tr = len(full_tr_x)
            N_te = len(te_x)

            if model_type == "linear_residual":
                m_out = Ridge(alpha=50.0)
                m_out.fit(full_tr_x.reshape(-1, D), -full_tr_ade.reshape(-1))
                tr_delta_out = m_out.predict(full_tr_x.reshape(-1, D)).reshape(N_tr, K)
                te_delta_out = m_out.predict(te_x.reshape(-1, D)).reshape(N_te, K)
                scale_out = np.std(full_tr_logits) / (np.std(tr_delta_out) + 1e-6)
                param_count = D + 1
            else:
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

        te_logits = raw_data[test_fold]["g0"][:, :, 0]
        N_te = len(te_logits)
        te_scores = te_logits + best_alpha * scale_out * te_delta_out
        te_sel = np.argmax(te_scores, axis=-1)

        ev = evaluate_local_predictions(
            selected_indices=te_sel,
            topk_indices=np.tile(np.arange(20), (N_te, 1)),
            topk_scores=te_scores,
            full_ade=raw_data[test_fold]["ade"],
            full_fde=raw_data[test_fold]["fde"],
            full_collision=raw_data[test_fold]["collision"],
            native_indices=raw_data[test_fold]["native_top1_idx"],
            has_neighbors=raw_data[test_fold].get("has_neighbours"),
        )
        fold_evals.append(ev)

    macro = {m: float(np.mean([e[m] for e in fold_evals])) for m in fold_evals[0]}
    mean_alpha = float(np.mean(fold_alphas)) if fold_alphas else 0.0
    total_samples = sum(len(raw_data[f]["ade"]) for f in FOLDS)
    latency_ms = (time.time() - t0) * 1000.0 / total_samples
    return macro, mean_alpha, param_count, latency_ms


def main():
    parser = argparse.ArgumentParser(description="EXP-S2V2C: Global Latent Probe")
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
    print(f"[EXP-S2V2C] Starting Global Latent Probes on {device}...")

    raw_data = {f: load_fold_features(feature_root, f, seed=args.seed) for f in FOLDS}
    latent_data = {f: load_fold_latents_v2(latent_dir, f) for f in FOLDS}

    all_results = []
    models_to_run = ["linear_residual", "mlp_residual"]

    for feat_groups, latent_key, cfg_name in PROBE_CONFIGS:
        for m_type in models_to_run:
            if cfg_name == "G0_native" and m_type == "mlp_residual":
                continue # G0 native is fixed baseline

            print(f"\n--- Evaluating {cfg_name} [{m_type}] ---")
            macro, mean_alpha, p_count, lat_ms = evaluate_probe_model(
                cfg_name=cfg_name,
                feat_groups=feat_groups,
                latent_key=latent_key,
                model_type=m_type,
                raw_data=raw_data,
                latent_data=latent_data,
                base_seed=args.seed,
                device=device,
            )

            print(f"  Macro ADE: {macro['selected_ade']:.4f}m | Regret: {macro['ade_regret']:.4f}m | Pairwise@10: {macro['local_pairwise_10']*100:.2f}% | Alpha: {mean_alpha:.2f}")

            row = {
                "Configuration": cfg_name,
                "Model_Type": m_type,
                "Feature_Groups": "+".join(feat_groups) if feat_groups else "None",
                "Latent": latent_key or "None",
                "ADE_Macro_m": f"{macro['selected_ade']:.4f}",
                "FDE_Macro_m": f"{macro['selected_fde']:.4f}",
                "ADE_Regret_m": f"{macro['ade_regret']:.4f}",
                "Oracle_Hit": f"{macro['oracle_hit']*100:.2f}%",
                "NearOracle05": f"{macro['near_oracle_05']*100:.2f}%",
                "PairwiseAcc_05": f"{macro['local_pairwise_05']*100:.2f}%",
                "PairwiseAcc_10": f"{macro['local_pairwise_10']*100:.2f}%",
                "Collision_Social": f"{macro['collision_social']*100:.2f}%",
                "Mean_Alpha": f"{mean_alpha:.2f}",
                "Latency_ms": f"{lat_ms:.3f}",
                "Params": str(p_count),
            }
            all_results.append(row)

    # 1. Global Latent Probe CSV
    global_csv_path = out_dir / "global_latent_probe.csv"
    with open(global_csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)

    # 2. Latent Ablation CSV (Configs investigating L5 and combinations)
    ablation_cfgs = {"G0_native", "G0_L5", "G0_G1_L5", "G0_G4_L5", "G0_G1_G4_L5", "L5_only"}
    ablation_rows = [r for r in all_results if r["Configuration"] in ablation_cfgs]
    ablation_csv_path = out_dir / "latent_ablation.csv"
    with open(ablation_csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(ablation_rows)

    print(f"\n[EXP-S2V2C OK] Saved Global Probes -> {global_csv_path}, Latent Ablation -> {ablation_csv_path}")


if __name__ == "__main__":
    main()
