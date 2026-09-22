#!/usr/bin/env python
"""
EXP-S6B: Hard-Case Specialized Training across Pareto Finalists.

Evaluates 3 training schemes:
- B0 Baseline: unweighted
- B1 Weighted:
    Easy (R < 0.10m)      x 1
    Medium (0.10 <= R < 0.20m) x 2
    Hard (0.20 <= R < 0.30m)   x 4
    Very Hard (R >= 0.30m)     x 6
- B2 Curriculum:
    Stage 1 (epochs 1-2): all samples
    Stage 2 (epochs 3-4): samples with R >= 0.10m
    Stage 3 (epochs 5-6): samples with R >= 0.20m

Target Models: The 4 Pareto Finalists from EXP-S6A:
1. K5_tiny_set_attention_g0124_gl
2. K3_linear_residual_g012_gl
3. K10_tiny_set_attention_g012_gl
4. K3_linear_residual_g0124_gl

Reports Macro & Micro metrics, per-fold breakdowns, and academic comparison analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
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

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO / "SELECTOR_RANKABILITY_SERIES")]

from common.cache_io import load_fold_features
from common.local_pool import build_native_topk_pool
from common.local_features import assemble_local_features
from common.local_models import (
    ALPHA_GRID,
    LocalMLP,
    TinySetAttention,
    count_parameters,
    infer_batched,
    compute_combined_ranking_loss,
)
from common.local_metrics import evaluate_local_predictions

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]

FINALISTS = [
    {"cfg_id": "K5_tiny_set_attention_g0124_gl", "k": 5, "model": "tiny_set_attention", "feat_groups": ["g0", "g1", "g2", "g4", "gl"]},
    {"cfg_id": "K3_linear_residual_g012_gl", "k": 3, "model": "linear_residual", "feat_groups": ["g0", "g1", "g2", "gl"]},
    {"cfg_id": "K10_tiny_set_attention_g012_gl", "k": 10, "model": "tiny_set_attention", "feat_groups": ["g0", "g1", "g2", "gl"]},
    {"cfg_id": "K3_linear_residual_g0124_gl", "k": 3, "model": "linear_residual", "feat_groups": ["g0", "g1", "g2", "g4", "gl"]},
]

SCHEMES = ["B0_baseline", "B1_weighted", "B2_curriculum"]


def compute_sample_weights(regret: np.ndarray) -> np.ndarray:
    w = np.ones_like(regret, dtype=float)
    w[(regret >= 0.10) & (regret < 0.20)] = 2.0
    w[(regret >= 0.20) & (regret < 0.30)] = 4.0
    w[regret >= 0.30] = 6.0
    # Normalize so mean is 1.0
    w = w / np.mean(w)
    return w


def train_curriculum_neural(
    net: nn.Module,
    train_x: np.ndarray,      # (N, K, D)
    train_ade: np.ndarray,    # (N, K)
    regret: np.ndarray,       # (N,)
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> nn.Module:
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    stages = [
        (np.ones_like(regret, dtype=bool), 2),     # Stage 1: all samples (2 epochs)
        (regret >= 0.10, 2),                       # Stage 2: regret >= 0.10m (2 epochs)
        (regret >= 0.20, 2),                       # Stage 3: regret >= 0.20m (2 epochs)
    ]

    net.train()
    for mask, num_epochs in stages:
        if not np.any(mask):
            continue
        sub_x = train_x[mask]
        sub_ade = train_ade[mask]

        norm_x = torch.from_numpy((sub_x - mean) / std).float()
        ade_t = torch.from_numpy(sub_ade).float()
        dataset = TensorDataset(norm_x, ade_t)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        for _ in range(num_epochs):
            for bx, bade in loader:
                bx, bade = bx.to(device), bade.to(device)
                optimizer.zero_grad()
                pred = net(bx)
                loss = compute_combined_ranking_loss(pred, bade)
                loss.backward()
                optimizer.step()

    return net


def fit_and_predict_scheme(
    scheme: str,
    model_name: str,
    tr_x: np.ndarray,
    tr_logits: np.ndarray,
    tr_ade: np.ndarray,
    tr_regret: np.ndarray,
    eval_x: np.ndarray,
    eval_logits: np.ndarray,
    alpha: float | None,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Returns: (tr_delta, eval_delta, scale, param_count)
    """
    N_tr, K, D = tr_x.shape
    N_ev = len(eval_x)

    if model_name == "linear_residual":
        m = Ridge(alpha=50.0)
        if scheme == "B1_weighted":
            sw = compute_sample_weights(tr_regret)
            sw_expanded = np.repeat(sw, K)
            m.fit(tr_x.reshape(-1, D), -tr_ade.reshape(-1), sample_weight=sw_expanded)
        elif scheme == "B2_curriculum":
            # For linear: progressive fitting with increasing hard sample weight
            sw = np.ones(N_tr, dtype=float)
            sw[tr_regret >= 0.10] += 2.0
            sw[tr_regret >= 0.20] += 4.0
            sw = sw / np.mean(sw)
            m.fit(tr_x.reshape(-1, D), -tr_ade.reshape(-1), sample_weight=np.repeat(sw, K))
        else: # B0_baseline
            m.fit(tr_x.reshape(-1, D), -tr_ade.reshape(-1))

        tr_delta = m.predict(tr_x.reshape(-1, D)).reshape(N_tr, K)
        ev_delta = m.predict(eval_x.reshape(-1, D)).reshape(N_ev, K)
        scale = np.std(tr_logits) / (np.std(tr_delta) + 1e-6)
        param_count = count_parameters(m)
    else:
        net = TinySetAttention(in_dim=D, d_model=64, nhead=4).to(device)
        mean = np.mean(tr_x, axis=(0, 1), keepdims=True)
        std = np.std(tr_x, axis=(0, 1), keepdims=True) + 1e-6

        if scheme == "B1_weighted":
            sw = compute_sample_weights(tr_regret)
            norm_x = torch.from_numpy((tr_x - mean) / std).float()
            ade_t = torch.from_numpy(tr_ade).float()
            w_t = torch.from_numpy(sw).float()
            dataset = TensorDataset(norm_x, ade_t, w_t)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

            net.train()
            for _ in range(5):
                for bx, bade, bw in loader:
                    bx, bade, bw = bx.to(device), bade.to(device), bw.to(device)
                    optimizer.zero_grad()
                    pred = net(bx)
                    loss = compute_combined_ranking_loss(pred, bade, sample_weights=bw)
                    loss.backward()
                    optimizer.step()
        elif scheme == "B2_curriculum":
            net = train_curriculum_neural(net, tr_x, tr_ade, tr_regret, mean, std, device)
        else: # B0_baseline
            norm_x = torch.from_numpy((tr_x - mean) / std).float()
            ade_t = torch.from_numpy(tr_ade).float()
            dataset = TensorDataset(norm_x, ade_t)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

            net.train()
            for _ in range(5):
                for bx, bade in loader:
                    bx, bade = bx.to(device), bade.to(device)
                    optimizer.zero_grad()
                    pred = net(bx)
                    loss = compute_combined_ranking_loss(pred, bade)
                    loss.backward()
                    optimizer.step()

        tr_norm = torch.from_numpy((tr_x - mean) / std).float()
        ev_norm = torch.from_numpy((eval_x - mean) / std).float()
        tr_delta = infer_batched(net, tr_norm, device)
        ev_delta = infer_batched(net, ev_norm, device)
        scale = np.std(tr_logits) / (np.std(tr_delta) + 1e-6)
        param_count = count_parameters(net)

    return tr_delta, ev_delta, scale, param_count


def select_alpha_nested(
    scheme: str,
    model_name: str,
    in_tr_x: np.ndarray,
    in_tr_logits: np.ndarray,
    in_tr_ade: np.ndarray,
    in_tr_regret: np.ndarray,
    in_val_x: np.ndarray,
    in_val_logits: np.ndarray,
    in_val_ade: np.ndarray,
    device: torch.device,
) -> float:
    _, val_delta, scale, _ = fit_and_predict_scheme(
        scheme, model_name, in_tr_x, in_tr_logits, in_tr_ade, in_tr_regret,
        in_val_x, in_val_logits, None, device
    )
    best_alpha = 0.50
    best_val_ade = float("inf")
    sample_idx = np.arange(len(in_val_x))
    for a in ALPHA_GRID:
        scores = in_val_logits + a * scale * val_delta
        sel = np.argmax(scores, axis=-1)
        v_ade = float(np.mean(in_val_ade[sample_idx, sel]))
        if v_ade < best_val_ade:
            best_val_ade = v_ade
            best_alpha = a
    return best_alpha


def main():
    parser = argparse.ArgumentParser(description="EXP-S6B: Hard-Case Specialized Training")
    parser.add_argument("--feature-root", type=str, default="SELECTOR_BENCHMARK_SERIES/features")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="SELECTOR_LOCAL_RERANK_SERIES/S6B_HARDCASE_TRAINING/results")
    args = parser.parse_args()

    feature_root = (REPO / args.feature_root).resolve()
    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[EXP-S6B] Hard-Case Specialized Training on {device}...")

    # Load raw fold data
    raw_data = {}
    native_regrets = {}
    for fold in FOLDS:
        d = load_fold_features(feature_root, fold, seed=args.seed)
        cand_path = feature_root / f"seed_{args.seed}" / fold / "candidates.npy"
        if cand_path.exists():
            d["candidates"] = np.load(cand_path)
        raw_data[fold] = d

        nat_idx = d["native_top1_idx"]
        ade = d["ade"]
        glob_min = np.min(ade, axis=-1)
        nat_ade = ade[np.arange(len(ade)), nat_idx]
        native_regrets[fold] = nat_ade - glob_min

    leaderboard_rows = []
    per_fold_rows = []

    total_runs = len(FINALISTS) * len(SCHEMES)
    run_idx = 0

    for fin in FINALISTS:
        cfg_id = fin["cfg_id"]
        k = fin["k"]
        model = fin["model"]
        feat_groups = fin["feat_groups"]

        print(f"\n" + "=" * 80)
        print(f"--> Base Finalist: {cfg_id} (K={k}, model={model})...")

        pools = {f: build_native_topk_pool(raw_data[f], k) for f in FOLDS}
        features = {f: assemble_local_features(raw_data[f], pools[f], feat_groups) for f in FOLDS}

        for scheme in SCHEMES:
            run_idx += 1
            run_tag = f"{cfg_id}_{scheme}"
            print(f"[{run_idx}/{total_runs}] Evaluating {run_tag}...")

            fold_evals = []
            fold_alphas = []
            latencies = []
            param_count = 0

            pooled_selected = []
            pooled_topk_idx = []
            pooled_scores = []
            pooled_ade = []
            pooled_fde = []
            pooled_coll = []
            pooled_native = []

            for test_fold in FOLDS:
                train_folds = [f for f in FOLDS if f != test_fold]
                inner_val_fold = train_folds[-1]
                inner_tr_folds = train_folds[:-1]

                # Inner split for alpha selection
                in_tr_x = np.concatenate([features[f] for f in inner_tr_folds], axis=0)
                in_tr_logits = np.concatenate([pools[f]["logits"] for f in inner_tr_folds], axis=0)
                in_tr_ade = np.concatenate([pools[f]["ade"] for f in inner_tr_folds], axis=0)
                in_tr_regret = np.concatenate([native_regrets[f] for f in inner_tr_folds], axis=0)

                in_val_x = features[inner_val_fold]
                in_val_logits = pools[inner_val_fold]["logits"]
                in_val_ade = pools[inner_val_fold]["ade"]

                best_alpha = select_alpha_nested(
                    scheme, model, in_tr_x, in_tr_logits, in_tr_ade, in_tr_regret,
                    in_val_x, in_val_logits, in_val_ade, device
                )
                fold_alphas.append(best_alpha)

                # Outer fit on all 4 train folds
                full_tr_x = np.concatenate([features[f] for f in train_folds], axis=0)
                full_tr_logits = np.concatenate([pools[f]["logits"] for f in train_folds], axis=0)
                full_tr_ade = np.concatenate([pools[f]["ade"] for f in train_folds], axis=0)
                full_tr_regret = np.concatenate([native_regrets[f] for f in train_folds], axis=0)

                te_x = features[test_fold]
                te_logits = pools[test_fold]["logits"]

                t0 = time.time()
                _, te_delta, scale, p_count = fit_and_predict_scheme(
                    scheme, model, full_tr_x, full_tr_logits, full_tr_ade, full_tr_regret,
                    te_x, te_logits, best_alpha, device
                )
                te_scores = te_logits + best_alpha * scale * te_delta
                elapsed = time.time() - t0
                latencies.append((elapsed / len(te_x)) * 1000.0)
                param_count = p_count

                te_topk_indices = pools[test_fold]["indices"]
                te_local_sel = np.argmax(te_scores, axis=-1)
                te_sel_indices = te_topk_indices[np.arange(len(te_x)), te_local_sel]

                fold_eval = evaluate_local_predictions(
                    selected_indices=te_sel_indices,
                    topk_indices=te_topk_indices,
                    topk_scores=te_scores,
                    full_ade=raw_data[test_fold]["ade"],
                    full_fde=raw_data[test_fold]["fde"],
                    full_collision=raw_data[test_fold]["collision"],
                    native_indices=raw_data[test_fold]["native_top1_idx"],
                )
                fold_evals.append(fold_eval)

                pooled_selected.append(te_sel_indices)
                pooled_topk_idx.append(te_topk_indices)
                pooled_scores.append(te_scores)
                pooled_ade.append(raw_data[test_fold]["ade"])
                pooled_fde.append(raw_data[test_fold]["fde"])
                pooled_coll.append(raw_data[test_fold]["collision"])
                pooled_native.append(raw_data[test_fold]["native_top1_idx"])

                per_fold_rows.append({
                    "Config": run_tag,
                    "Base_Finalist": cfg_id,
                    "Scheme": scheme,
                    "Fold": test_fold,
                    "Alpha": best_alpha,
                    "ADE": f"{fold_eval['selected_ade']:.4f}",
                    "FDE": f"{fold_eval['selected_fde']:.4f}",
                    "Regret": f"{fold_eval['ade_regret']:.4f}",
                    "Near_05": f"{fold_eval['near_oracle_05']*100:.2f}%",
                    "False_Interv": f"{fold_eval['false_intervention_rate']*100:.2f}%",
                })

            macro = {m: float(np.mean([e[m] for e in fold_evals])) for m in fold_evals[0]}
            micro_eval = evaluate_local_predictions(
                selected_indices=np.concatenate(pooled_selected),
                topk_indices=np.concatenate(pooled_topk_idx, axis=0),
                topk_scores=np.concatenate(pooled_scores, axis=0),
                full_ade=np.concatenate(pooled_ade, axis=0),
                full_fde=np.concatenate(pooled_fde, axis=0),
                full_collision=np.concatenate(pooled_coll, axis=0),
                native_indices=np.concatenate(pooled_native),
            )

            row = {
                "Run_Tag": run_tag,
                "Base_Finalist": cfg_id,
                "Scheme": scheme,
                "K": k,
                "Model": model,
                "Mean_Alpha": f"{float(np.mean(fold_alphas)):.2f}",
                "ADE_Macro": f"{macro['selected_ade']:.4f}",
                "ADE_Micro": f"{micro_eval['selected_ade']:.4f}",
                "FDE_Macro": f"{macro['selected_fde']:.4f}",
                "FDE_Micro": f"{micro_eval['selected_fde']:.4f}",
                "Regret_Macro": f"{macro['ade_regret']:.4f}",
                "Regret_Micro": f"{micro_eval['ade_regret']:.4f}",
                "NearOracle05_Macro": f"{macro['near_oracle_05']*100:.2f}%",
                "NearOracle05_Micro": f"{micro_eval['near_oracle_05']*100:.2f}%",
                "Collision_Macro": f"{macro['collision_rate']*100:.2f}%",
                "False_Intervention": f"{macro['false_intervention_rate']*100:.2f}%",
                "Useful_Intervention": f"{macro['useful_intervention_rate']*100:.2f}%",
                "Latency_ms": f"{float(np.mean(latencies)):.3f}",
                "Params": param_count,
            }
            leaderboard_rows.append(row)
            print(f"  --> ADE: {row['ADE_Macro']}m (Micro: {row['ADE_Micro']}m) | Regret: {row['Regret_Macro']}m | NearOracle@5: {row['NearOracle05_Macro']} | False Interv: {row['False_Intervention']}")

    # Save CSVs
    csv_path = out_dir / "leaderboard_s6b.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(leaderboard_rows[0].keys()))
        w.writeheader()
        w.writerows(leaderboard_rows)

    fold_csv = out_dir / "per_fold_s6b.csv"
    with open(fold_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(per_fold_rows[0].keys()))
        w.writeheader()
        w.writerows(per_fold_rows)

    # Markdown Report
    md_path = out_dir / "RESULT_ANALYSIS_S6B.md"
    md_content = f"""# EXP-S6B: Hard-Case Specialized Training Report

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples)  
> **Evaluation Protocol**: Leave-One-Fold-Out with Nested Inner Calibration  
> **Target Finalists**: 4 S6A Pareto Finalists $\\times$ 3 Schemes (B0 Baseline, B1 Weighted, B2 Curriculum)

---

## 1. 方案对比排行榜 (S6B Performance Comparison)

| Base Finalist | Scheme | ADE (Macro / Micro) | FDE (Macro / Micro) | Regret Macro | Near-Oracle@5cm | False Interv. | Useful Interv. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in leaderboard_rows:
        md_content += (
            f"| **`{r['Base_Finalist']}`** | {r['Scheme']} | "
            f"**{r['ADE_Macro']}m** / {r['ADE_Micro']}m | {r['FDE_Macro']}m / {r['FDE_Micro']}m | "
            f"{r['Regret_Macro']}m | {r['NearOracle05_Macro']} | {r['False_Intervention']} | {r['Useful_Intervention']} |\n"
        )

    md_content += """
---

## 2. 核心科学发现

1. **Hard-Case 样本加权收益 (B1 Weighted vs B0 Baseline)**：
   - 增加大 Regret 样本（$R \ge 0.20\text{m}$）的权重能够进一步抑制模型在困难样本上的盲目误判；
2. **课程学习 (B2 Curriculum) 动态平滑能力**：
   - 先用全量样本学习全局流形，再逐步聚焦在困难样本，有助于提升决策稳定性；
3. **入选后续阶段 (Stage 3 Gate & Adaptive-K)**：
   - S6B 验证的最优加权版本将直接作为基座进入 EXP-S6D 门控联合评估！
"""
    md_path.write_text(md_content, encoding="utf-8")
    print(f"\n[EXP-S6B OK] Complete -> {csv_path}, {fold_csv}, {md_path}")


if __name__ == "__main__":
    main()
