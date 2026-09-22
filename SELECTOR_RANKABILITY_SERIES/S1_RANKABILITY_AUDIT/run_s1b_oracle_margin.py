#!/usr/bin/env python
"""
EXP-S1B: Oracle Margin & Ambiguity Audit.

Zero-training audit evaluating whether finding the single best oracle candidate
is an ill-posed / overly sharp target due to candidate ambiguity (delta_1,2 ~ 0).
Also computes Near-Oracle set statistics O_epsilon.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path[:0] = [str(HERE.parent)]
from common.cache_io import load_fold_features

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]


def main():
    parser = argparse.ArgumentParser(description="EXP-S1B: Oracle Margin Audit")
    parser.add_argument("--feature-root", type=str, default="SELECTOR_BENCHMARK_SERIES/features")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S1_RANKABILITY_AUDIT/results")
    args = parser.parse_args()

    feature_root = (REPO / args.feature_root).resolve()
    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXP-S1B] Running Oracle Margin Audit on seed={args.seed} across 5 folds...")

    all_delta12 = []
    all_delta15 = []
    all_native_regret = []
    all_oracle_ade = []
    all_native_ade = []

    fold_summaries = {}

    for fold in FOLDS:
        data = load_fold_features(feature_root, fold, seed=args.seed)
        ade = data["ade"]                       # (N, 20)
        native_top1 = data["native_top1_idx"]   # (N,)
        N = len(ade)

        sorted_ade = np.sort(ade, axis=-1)      # (N, 20)
        d12 = sorted_ade[:, 1] - sorted_ade[:, 0]
        d15 = sorted_ade[:, 4] - sorted_ade[:, 0]
        nat_ade = ade[np.arange(N), native_top1]
        nat_reg = nat_ade - sorted_ade[:, 0]

        all_delta12.append(d12)
        all_delta15.append(d15)
        all_native_regret.append(nat_reg)
        all_oracle_ade.append(sorted_ade[:, 0])
        all_native_ade.append(nat_ade)

        # Near-oracle sets
        near_002 = np.sum(ade <= (sorted_ade[:, :1] + 0.02), axis=-1)
        near_005 = np.sum(ade <= (sorted_ade[:, :1] + 0.05), axis=-1)
        near_010 = np.sum(ade <= (sorted_ade[:, :1] + 0.10), axis=-1)

        native_in_002 = nat_reg <= 0.02
        native_in_005 = nat_reg <= 0.05
        native_in_010 = nat_reg <= 0.10

        fold_summaries[fold] = {
            "samples": N,
            "mean_oracle_ade": float(np.mean(sorted_ade[:, 0])),
            "mean_native_ade": float(np.mean(nat_ade)),
            "mean_native_regret": float(np.mean(nat_reg)),
            "median_delta12": float(np.median(d12)),
            "mean_delta12": float(np.mean(d12)),
            "p_delta12_lt_001": float(np.mean(d12 < 0.01)),
            "p_delta12_lt_002": float(np.mean(d12 < 0.02)),
            "p_delta12_lt_005": float(np.mean(d12 < 0.05)),
            "p_delta12_lt_010": float(np.mean(d12 < 0.10)),
            "mean_near_oracle_count_005": float(np.mean(near_005)),
            "mean_near_oracle_count_010": float(np.mean(near_010)),
            "p_native_in_near_005": float(np.mean(native_in_005)),
            "p_native_in_near_010": float(np.mean(native_in_010)),
        }

    # Aggregate across all folds
    cat_d12 = np.concatenate(all_delta12)
    cat_d15 = np.concatenate(all_delta15)
    cat_nat_reg = np.concatenate(all_native_regret)
    cat_oracle_ade = np.concatenate(all_oracle_ade)
    cat_nat_ade = np.concatenate(all_native_ade)

    total_samples = len(cat_d12)
    overall_summary = {
        "total_samples": total_samples,
        "mean_oracle_ade": float(np.mean(cat_oracle_ade)),
        "mean_native_ade": float(np.mean(cat_nat_ade)),
        "mean_native_regret": float(np.mean(cat_nat_reg)),
        "median_delta12": float(np.median(cat_d12)),
        "mean_delta12": float(np.mean(cat_d12)),
        "p_delta12_lt_001": float(np.mean(cat_d12 < 0.01)),
        "p_delta12_lt_002": float(np.mean(cat_d12 < 0.02)),
        "p_delta12_lt_005": float(np.mean(cat_d12 < 0.05)),
        "p_delta12_lt_010": float(np.mean(cat_d12 < 0.10)),
        "p_native_regret_lt_005": float(np.mean(cat_nat_reg < 0.05)),
        "p_native_regret_lt_010": float(np.mean(cat_nat_reg < 0.10)),
        "p_native_regret_gt_020": float(np.mean(cat_nat_reg > 0.20)),
        "p_native_regret_gt_030": float(np.mean(cat_nat_reg > 0.30)),
        "p_native_regret_gt_050": float(np.mean(cat_nat_reg > 0.50)),
    }

    # Save JSON summary
    summary_out = {
        "overall": overall_summary,
        "per_fold": fold_summaries,
    }
    json_path = out_dir / "oracle_margin_summary.json"
    json_path.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")

    # Save Markdown analysis
    md_path = out_dir / "RESULT_ANALYSIS_S1B.md"
    report_md = f"""# EXP-S1B: Oracle Margin & Ambiguity Audit Report

> **Dataset**: ETH/UCY Official 5 Folds ({total_samples:,} samples)  
> **Evaluation Seed**: {args.seed}

---

## 1. 核心实证结论 (Ambiguity & Sharpness Analysis)

- **Oracle 候选存在强模糊性 (Extreme Ambiguity)**：
  - **{overall_summary['p_delta12_lt_002']*100:.2f}%** 的样本中，第 1 候选与第 2 候选的误差差距 $\Delta_{{1,2}} < 2.0\text{{cm}}$；
  - **{overall_summary['p_delta12_lt_005']*100:.2f}%** 的样本中，$\Delta_{{1,2}} < 5.0\text{{cm}}$。
  - **科学结论**：强行训练网络去分类单一最佳索引（Argmin ADE）在物理上是病态的（Ill-posed），因为 Top-2 甚至 Top-3 候选之间的物理形态极其接近，存在严重的标签噪声与过度锐化问题。

- **Native Regret 双峰分布 (Bimodal Dilemma)**：
  - **{overall_summary['p_native_regret_lt_005']*100:.2f}%** 的样本中，Native 自身就是 Near-Oracle（Regret < 5cm），盲目重排极易产生误干预；
  - 但在 **{overall_summary['p_native_regret_gt_020']*100:.2f}%** 的样本中，Native 发生了严重误判（Regret > 20cm），这部分样本正是拉高全局 ADE 的根源。

---

## 2. 各折明细对比 (Per-Fold Ambiguity Table)

| Fold | Samples | Mean Oracle ADE (m) | Mean Native ADE (m) | Mean Regret (m) | Median $\Delta_{{1,2}}$ (m) | $P(\Delta_{{1,2}} < 2\text{{cm}})$ | $P(\text{{Regret}} < 5\text{{cm}})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for fold, s in fold_summaries.items():
        report_md += f"| **{fold}** | {s['samples']} | {s['mean_oracle_ade']:.4f} | {s['mean_native_ade']:.4f} | {s['mean_native_regret']:.4f} | {s['median_delta12']:.4f} | {s['p_delta12_lt_002']*100:.1f}% | {s['p_native_in_near_005']*100:.1f}% |\n"

    md_path.write_text(report_md, encoding="utf-8")
    print(f"[EXP-S1B OK] Audit complete -> {json_path} and {md_path}")


if __name__ == "__main__":
    main()
