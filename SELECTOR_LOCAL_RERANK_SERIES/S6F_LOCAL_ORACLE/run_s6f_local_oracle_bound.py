#!/usr/bin/env python
"""
EXP-S6F: Local Oracle Upper Bound Audit across Native Top-K Subpools.

Computes the theoretical ceiling achievable by any local reranker restricted to
Native Top-K candidate pools (K in 1, 2, 3, 5, 10, 20).
Simultaneously outputs Macro (unweighted fold average) and Micro (pooled 34,161 samples)
metrics, plus a per-fold breakdown.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO / "SELECTOR_RANKABILITY_SERIES")]
from common.cache_io import load_fold_features
from common.local_pool import build_native_topk_pool

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]
K_EVAL = [1, 2, 3, 5, 10, 20]
EPS_LIST = [0.02, 0.05, 0.10]


def main():
    parser = argparse.ArgumentParser(description="EXP-S6F: Local Oracle Upper Bound Audit")
    parser.add_argument("--feature-root", type=str, default="SELECTOR_BENCHMARK_SERIES/features")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="SELECTOR_LOCAL_RERANK_SERIES/S6F_LOCAL_ORACLE/results")
    args = parser.parse_args()

    feature_root = (REPO / args.feature_root).resolve()
    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[EXP-S6F] Evaluating Local Oracle Upper Bounds across ETH/UCY 5 Folds...")

    fold_data = {}
    for fold in FOLDS:
        d = load_fold_features(feature_root, fold, seed=args.seed)
        fold_data[fold] = d

    # Collect per-fold and pooled stats for each K
    results_by_k = {k: {"fold_metrics": [], "pooled_ade": [], "pooled_fde": [],
                        "pooled_gap": [], "pooled_exact": [],
                        "pooled_near02": [], "pooled_near05": [], "pooled_near10": []}
                    for k in K_EVAL}
    per_fold_rows = []

    for fold in FOLDS:
        d = fold_data[fold]
        ade = d["ade"]  # (N, 20)
        fde = d["fde"]  # (N, 20)
        N = len(ade)

        glob_oracle_ade = np.min(ade, axis=-1)  # (N,)
        glob_oracle_fde = np.min(fde, axis=-1)  # (N,)
        glob_oracle_idx = np.argmin(ade, axis=-1)  # (N,)

        for k in K_EVAL:
            pool = build_native_topk_pool(d, k)
            topk_ade = pool["ade"]  # (N, k)
            topk_fde = pool["fde"]  # (N, k)

            local_oracle_ade = np.min(topk_ade, axis=-1)  # (N,)
            local_oracle_fde = np.min(topk_fde, axis=-1)  # (N,)
            gap = local_oracle_ade - glob_oracle_ade      # (N,)

            exact_rec = (pool["indices"] == glob_oracle_idx[:, None]).any(axis=-1)
            near_02 = (topk_ade <= (glob_oracle_ade[:, None] + 0.02)).any(axis=-1)
            near_05 = (topk_ade <= (glob_oracle_ade[:, None] + 0.05)).any(axis=-1)
            near_10 = (topk_ade <= (glob_oracle_ade[:, None] + 0.10)).any(axis=-1)

            # Store per-fold metric
            fold_m = {
                "local_oracle_ade": float(np.mean(local_oracle_ade)),
                "local_oracle_fde": float(np.mean(local_oracle_fde)),
                "gap": float(np.mean(gap)),
                "exact_recall": float(np.mean(exact_rec)),
                "near_02": float(np.mean(near_02)),
                "near_05": float(np.mean(near_05)),
                "near_10": float(np.mean(near_10)),
            }
            results_by_k[k]["fold_metrics"].append(fold_m)

            # Store pooled
            results_by_k[k]["pooled_ade"].append(local_oracle_ade)
            results_by_k[k]["pooled_fde"].append(local_oracle_fde)
            results_by_k[k]["pooled_gap"].append(gap)
            results_by_k[k]["pooled_exact"].append(exact_rec)
            results_by_k[k]["pooled_near02"].append(near_02)
            results_by_k[k]["pooled_near05"].append(near_05)
            results_by_k[k]["pooled_near10"].append(near_10)

            per_fold_rows.append({
                "Fold": fold,
                "Samples": N,
                "K": f"Top-{k}",
                "Local_Oracle_ADE": f"{fold_m['local_oracle_ade']:.4f}",
                "Local_Oracle_FDE": f"{fold_m['local_oracle_fde']:.4f}",
                "Gap_to_Global_m": f"{fold_m['gap']:.4f}",
                "Exact_Oracle_Recall": f"{fold_m['exact_recall']*100:.2f}%",
                "Near_Oracle_05cm_Recall": f"{fold_m['near_05']*100:.2f}%",
            })

    # Summary Table across K
    summary_rows = []
    for k in K_EVAL:
        f_ms = results_by_k[k]["fold_metrics"]
        macro_ade = float(np.mean([m["local_oracle_ade"] for m in f_ms]))
        macro_fde = float(np.mean([m["local_oracle_fde"] for m in f_ms]))
        macro_gap = float(np.mean([m["gap"] for m in f_ms]))
        macro_exact = float(np.mean([m["exact_recall"] for m in f_ms])) * 100.0
        macro_near02 = float(np.mean([m["near_02"] for m in f_ms])) * 100.0
        macro_near05 = float(np.mean([m["near_05"] for m in f_ms])) * 100.0
        macro_near10 = float(np.mean([m["near_10"] for m in f_ms])) * 100.0

        micro_ade = float(np.mean(np.concatenate(results_by_k[k]["pooled_ade"])))
        micro_fde = float(np.mean(np.concatenate(results_by_k[k]["pooled_fde"])))
        micro_gap = float(np.mean(np.concatenate(results_by_k[k]["pooled_gap"])))
        micro_exact = float(np.mean(np.concatenate(results_by_k[k]["pooled_exact"]))) * 100.0
        micro_near02 = float(np.mean(np.concatenate(results_by_k[k]["pooled_near02"]))) * 100.0
        micro_near05 = float(np.mean(np.concatenate(results_by_k[k]["pooled_near05"]))) * 100.0
        micro_near10 = float(np.mean(np.concatenate(results_by_k[k]["pooled_near10"]))) * 100.0

        summary_rows.append({
            "K": f"Top-{k}",
            "LocalOracleADE_Macro": f"{macro_ade:.4f}",
            "LocalOracleADE_Micro": f"{micro_ade:.4f}",
            "LocalOracleFDE_Macro": f"{macro_fde:.4f}",
            "LocalOracleFDE_Micro": f"{micro_fde:.4f}",
            "GapToGlobal_Macro_m": f"{macro_gap:.4f}",
            "GapToGlobal_Micro_m": f"{micro_gap:.4f}",
            "ExactOracleRecall_Macro": f"{macro_exact:.2f}%",
            "ExactOracleRecall_Micro": f"{micro_exact:.2f}%",
            "NearOracle02_Macro": f"{macro_near02:.2f}%",
            "NearOracle05_Macro": f"{macro_near05:.2f}%",
            "NearOracle05_Micro": f"{micro_near05:.2f}%",
            "NearOracle10_Macro": f"{macro_near10:.2f}%",
        })

    # Save summary CSV
    summary_csv = out_dir / "local_oracle_bound.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Save per-fold CSV
    per_fold_csv = out_dir / "per_fold.csv"
    with open(per_fold_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(per_fold_rows[0].keys()))
        w.writeheader()
        w.writerows(per_fold_rows)

    # Save Markdown report
    md_path = out_dir / "RESULT_ANALYSIS_S6F.md"
    report_md = f"""# EXP-S6F: Local Oracle Upper Bound Audit Report

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples)  
> **Generator**: Frozen F16  
> **Evaluation Protocol**: Theoretical Minimum ADE/FDE Upper Bound inside Native Top-K Subpools

---

## 1. 局部子池理论上限 (Local Oracle Bounds by Subpool Size $K$)

| 候选池范围 | Local Oracle ADE (Macro / Micro) | Local Oracle FDE (Macro / Micro) | 距全局最优差距 (Gap to Global) | Exact Oracle 召回率 (Macro / Micro) | Near-Oracle@5cm 召回率 (Macro / Micro) |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in summary_rows:
        report_md += (
            f"| **{r['K']}** | {r['LocalOracleADE_Macro']}m / {r['LocalOracleADE_Micro']}m | "
            f"{r['LocalOracleFDE_Macro']}m / {r['LocalOracleFDE_Micro']}m | "
            f"+{r['GapToGlobal_Macro_m']}m | "
            f"{r['ExactOracleRecall_Macro']} / {r['ExactOracleRecall_Micro']} | "
            f"{r['NearOracle05_Macro']} / {r['NearOracle05_Micro']} |\n"
        )

    # Key comparisons
    ade_top1 = summary_rows[0]['LocalOracleADE_Macro']
    ade_top3 = summary_rows[2]['LocalOracleADE_Macro']
    ade_top5 = summary_rows[3]['LocalOracleADE_Macro']
    ade_top10 = summary_rows[4]['LocalOracleADE_Macro']
    ade_glob = summary_rows[5]['LocalOracleADE_Macro']

    report_md += f"""
---

## 2. 关键科学洞察与决策分水岭 (Critical Decision Thresholds)

1. **Top-K 蕴藏的子池潜力与截断上限**：
   - Native Top-1 的真实平均 ADE 为 **{ade_top1}m**；
   - 放宽至 **Top-3**，理论 Oracle ADE 降至 **{ade_top3}m**；
   - 放宽至 **Top-5**，理论 Oracle ADE 进一步降至 **{ade_top5}m**；
   - 放宽至 **Top-10**，理论 Oracle ADE 达到 **{ade_top10}m**，距全局 Oracle ({ade_glob}m) 差距缩小至 **+{summary_rows[4]['GapToGlobal_Macro_m']}m**。

2. **核心决策判定 (Decision Checkpoint)**：
   - **子池上限客观评估**：Native Top-K pools preserve substantial oracle headroom. However, candidate truncation remains a meaningful ceiling, especially for K=3 and K=5. Top-10 provides substantially stronger oracle coverage than Top-5.
   - 明确具体召回率：
     - Top-5 Exact Oracle Recall: **{summary_rows[3]['ExactOracleRecall_Macro']}** (Macro) / **{summary_rows[3]['ExactOracleRecall_Micro']}** (Micro)
     - Top-10 Exact Oracle Recall: **{summary_rows[4]['ExactOracleRecall_Macro']}** (Macro) / **{summary_rows[4]['ExactOracleRecall_Micro']}** (Micro)
   - 局部重排具备充足理论探索空间，但必须正视 Top-K 截断引入的客观上限约束。
"""
    md_path.write_text(report_md, encoding="utf-8")
    print(f"[EXP-S6F OK] Complete -> {summary_csv}, {per_fold_csv}, {md_path}")


if __name__ == "__main__":
    main()
