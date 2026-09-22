#!/usr/bin/env python
"""
EXP-S4A: Native Regret Distribution Analysis.

Analyzes the distribution of Native F16 regret r = ADE_native - min_ade20 across folds.
Partitions the problem space into 'safe/competent native' vs 'severely misleading native'.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path[:0] = [str(HERE.parent)]
from common.cache_io import load_fold_features

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]
THRESHOLDS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def main():
    parser = argparse.ArgumentParser(description="EXP-S4A: Native Regret Distribution")
    parser.add_argument("--feature-root", type=str, default="SELECTOR_BENCHMARK_SERIES/features")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S4_REGRET_GATE/results")
    args = parser.parse_args()

    feature_root = (REPO / args.feature_root).resolve()
    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXP-S4A] Analyzing Native Regret Distribution across 5 folds...")

    fold_rows = []
    all_regrets = []

    for fold in FOLDS:
        data = load_fold_features(feature_root, fold, seed=args.seed)
        ade = data["ade"]                       # (N, 20)
        native_top1 = data["native_top1_idx"]   # (N,)
        min_ade = np.min(ade, axis=-1)
        nat_ade = ade[np.arange(len(ade)), native_top1]
        regret = nat_ade - min_ade
        all_regrets.append(regret)

        row = {
            "fold": fold,
            "total_samples": len(regret),
            "mean_regret": float(np.mean(regret)),
            "median_regret": float(np.median(regret)),
            "std_regret": float(np.std(regret)),
            "q75_regret": float(np.percentile(regret, 75)),
            "q90_regret": float(np.percentile(regret, 90)),
        }
        for th in THRESHOLDS:
            row[f"pct_regret_ge_{int(th*100):02d}cm"] = float(np.mean(regret >= th)) * 100.0
        fold_rows.append(row)

    cat_regret = np.concatenate(all_regrets)
    overall = {
        "fold": "ALL_MACRO",
        "total_samples": len(cat_regret),
        "mean_regret": float(np.mean(cat_regret)),
        "median_regret": float(np.median(cat_regret)),
        "std_regret": float(np.std(cat_regret)),
        "q75_regret": float(np.percentile(cat_regret, 75)),
        "q90_regret": float(np.percentile(cat_regret, 90)),
    }
    for th in THRESHOLDS:
        overall[f"pct_regret_ge_{int(th*100):02d}cm"] = float(np.mean(cat_regret >= th)) * 100.0
    fold_rows.append(overall)

    csv_path = out_dir / "regret_by_fold.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        fields = list(fold_rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fold_rows)

    json_path = out_dir / "regret_summary.json"
    json_path.write_text(json.dumps({"overall": overall, "per_fold": fold_rows[:-1]}, indent=2), encoding="utf-8")

    print(f"[EXP-S4A OK] Regret distribution saved -> {csv_path}")


if __name__ == "__main__":
    main()
