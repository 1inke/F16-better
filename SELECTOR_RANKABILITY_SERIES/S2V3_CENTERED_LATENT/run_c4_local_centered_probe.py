#!/usr/bin/env python
"""
EXP-C4: Local Top-K Centered Latent Probe.

Investigates whether centered latent representations (L5c, L3c) improve fine discrimination
within local Top-K candidate pools (K in {3, 5, 10}, focusing on K=5):
- G0 + GL + L5c
- G0 + G1 + GL + L5c
- G0 + G1 + GL + L3c
- G0 + G1 + GL + raw_L5 (Direct control)

Model:
- Linear Residual (Ridge alpha=50.0)

Outputs:
- results/centered_local_probe.csv
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
from sklearn.linear_model import Ridge

from common.cache_io import load_fold_features
from common.local_pool import build_native_topk_pool
from common.local_features import assemble_local_features
from common.reproducibility import set_deterministic_seed, stable_seed
from common.local_models import ALPHA_GRID
from common.local_metrics import evaluate_local_predictions

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]
K_VALUES = [3, 5, 10]

LOCAL_CONFIGS = [
    (["g0", "gl"], "centered", "l5", "G0_GL_L5c"),
    (["g0", "g1", "gl"], "centered", "l5", "G0_G1_GL_L5c"),
    (["g0", "g1", "gl"], "centered", "l3", "G0_G1_GL_L3c"),
    (["g0", "g1", "gl"], "raw", "l5", "G0_G1_GL_raw_L5"),
]


def load_fold_latents(
    raw_dir: Path,
    centered_dir: Path,
    fold: str,
) -> Dict[str, np.ndarray]:
    raw_npz = raw_dir / fold / "latents.npz"
    raw_data = np.load(raw_npz)

    c_dir = centered_dir / fold
    l3_c = np.load(c_dir / "l3_centered.npy")
    l4_c = np.load(c_dir / "l4_centered.npy")
    l5_c = np.load(c_dir / "l5_centered.npy")

    return {
        "raw_l3": raw_data["l3"],
        "raw_l4": raw_data["l4"],
        "raw_l5": raw_data["l5"],
        "centered_l3": l3_c,
        "centered_l4": l4_c,
        "centered_l5": l5_c,
        "native_logits": raw_data["native_logits"],
    }


def build_local_features(
    raw_fold_data: Dict[str, np.ndarray],
    latent_fold_data: Dict[str, np.ndarray],
    pool: Dict[str, np.ndarray],
    base_groups: List[str],
    latent_type: str,
    latent_key: str,
) -> np.ndarray:
    base_x = assemble_local_features(raw_fold_data, pool, base_groups) # (N, K, D_base)
    lat_full = latent_fold_data[f"{latent_type}_{latent_key}"] # (N, 20, D_lat)
    topk_indices = pool["indices"] # (N, K)
    topk_lat = np.take_along_axis(lat_full, topk_indices[:, :, None], axis=1) # (N, K, D_lat)
    return np.concatenate([base_x, topk_lat], axis=-1).astype(np.float32)


def evaluate_c4_config(
    k: int,
    base_groups: List[str],
    latent_type: str,
    latent_key: str,
    cfg_name: str,
    raw_data: Dict[str, Any],
    latent_data: Dict[str, Any],
    base_seed: int,
) -> Tuple[Dict[str, float], float, int, float]:
    cfg_id = f"K{k}_{cfg_name}"
    pools = {f: build_native_topk_pool(raw_data[f], k) for f in FOLDS}
    features = {
        f: build_local_features(raw_data[f], latent_data[f], pools[f], base_groups, latent_type, latent_key)
        for f in FOLDS
    }

    t0 = time.time()
    fold_evals = []
    fold_alphas = []
    fold_sample_counts = []
    fold_ade_sums = []
    param_count = 0

    for test_fold in FOLDS:
        train_folds = [f for f in FOLDS if f != test_fold]
        inner_val_fold = train_folds[-1]
        inner_tr_folds = train_folds[:-1]

        N_te = len(raw_data[test_fold]["ade"])
        fold_sample_counts.append(N_te)

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

        m_in = Ridge(alpha=50.0)
        m_in.fit(in_tr_x.reshape(-1, D), -in_tr_ade.reshape(-1))
        val_delta = m_in.predict(in_val_x.reshape(-1, D)).reshape(N_in_val, k)
        tr_delta = m_in.predict(in_tr_x.reshape(-1, D)).reshape(N_in_tr, k)
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

        m_out = Ridge(alpha=50.0)
        m_out.fit(full_tr_x.reshape(-1, D), -full_tr_ade.reshape(-1))
        tr_delta_out = m_out.predict(full_tr_x.reshape(-1, D)).reshape(N_tr, k)
        te_delta_out = m_out.predict(te_x.reshape(-1, D)).reshape(N_te, k)
        scale_out = np.std(full_tr_logits) / (np.std(tr_delta_out) + 1e-6)
        param_count = D + 1

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
        fold_ade_sums.append(float(np.sum(raw_data[test_fold]["ade"][np.arange(N_te), te_sel_indices])))

    macro = {m: float(np.mean([e[m] for e in fold_evals])) for m in fold_evals[0]}
    micro_ade = float(sum(fold_ade_sums) / sum(fold_sample_counts))
    macro["selected_ade_micro"] = micro_ade

    mean_alpha = float(np.mean(fold_alphas)) if fold_alphas else 0.0
    total_samples = sum(fold_sample_counts)
    latency_ms = (time.time() - t0) * 1000.0 / total_samples
    return macro, mean_alpha, param_count, latency_ms


def main():
    parser = argparse.ArgumentParser(description="EXP-C4: Local Centered Latent Probe")
    parser.add_argument("--feature-root", type=str, default="SELECTOR_BENCHMARK_SERIES/features")
    parser.add_argument("--raw-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V2_CONTEXT_LATENT/latent_cache_v2")
    parser.add_argument("--centered-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V3_CENTERED_LATENT/centered_cache")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V3_CENTERED_LATENT/results")
    args = parser.parse_args()

    feature_root = (REPO / args.feature_root).resolve()
    raw_dir = (REPO / args.raw_dir / f"seed_{args.seed}").resolve()
    centered_dir = (REPO / args.centered_dir / f"seed_{args.seed}").resolve()
    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXP-C4] Starting Local Top-K Centered Latent Probes (Seed {args.seed})...")

    raw_data = {f: load_fold_features(feature_root, f, seed=args.seed) for f in FOLDS}
    latent_data = {f: load_fold_latents(raw_dir, centered_dir, f) for f in FOLDS}

    rows = []
    csv_path = out_dir / "centered_local_probe.csv"

    for k in K_VALUES:
        for base_groups, l_type, l_key, feat_name in LOCAL_CONFIGS:
            cfg_id = f"K{k}_{feat_name}"
            print(f"\n--- Evaluating {cfg_id} ---")
            macro, mean_alpha, p_count, lat_ms = evaluate_c4_config(
                k=k,
                base_groups=base_groups,
                latent_type=l_type,
                latent_key=l_key,
                cfg_name=feat_name,
                raw_data=raw_data,
                latent_data=latent_data,
                base_seed=args.seed,
            )

            print(
                f"  ADE Macro: {macro['selected_ade']:.4f}m | ADE Micro: {macro['selected_ade_micro']:.4f}m | "
                f"Pairwise@5cm: {macro['local_pairwise_05']*100:.2f}% | Pairwise@10cm: {macro['local_pairwise_10']*100:.2f}% | "
                f"Spearman: {macro['local_spearman']:.4f}"
            )

            row = {
                "Config": cfg_id,
                "K": str(k),
                "Feature": feat_name,
                "Latent_Type": l_type,
                "Latent_Key": l_key,
                "ADE_Macro_m": f"{macro['selected_ade']:.4f}",
                "ADE_Micro_m": f"{macro['selected_ade_micro']:.4f}",
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

            # Incremental save
            with open(csv_path, "w", newline="", encoding="utf-8") as fp:
                w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    print(f"\n[EXP-C4 OK] Centered local probe results saved to {csv_path}")


if __name__ == "__main__":
    main()
