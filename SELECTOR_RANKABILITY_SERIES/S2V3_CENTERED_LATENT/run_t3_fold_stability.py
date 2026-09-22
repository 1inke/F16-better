#!/usr/bin/env python
"""
EXP-T3: Fold Stability & Paired Bootstrap Analysis.

1. Fold Stability:
   Across 3 seeds, analyzes:
     Delta ADE_f = ADE^{Tiny}_f - ADE^{Transformer}_f
   Evaluates whether Tiny L5 stably maintains accuracy across all 5 folds
   (ETH, Hotel, Univ, Zara1, Zara2), verifying whether |Delta ADE_f| <= 0.005m.

2. Paired Bootstrap Analysis:
   Performs 10,000 paired bootstrap resamples on:
     Delta ADE = ADE_tiny - ADE_transformer
   Computes:
     - Mean Delta ADE
     - 95% Bootstrap Confidence Interval [2.5%, 97.5%]
     - Whether CI spans 0
   Performs analysis for each seed and across pooled seeds, both overall and per-fold.

Outputs:
- results/fold_stability_analysis.csv
- results/paired_bootstrap.json
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

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]
SEEDS = [1, 2, 3]


def run_paired_bootstrap(
    diff: np.ndarray,
    n_resamples: int = 10000,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.RandomState(rng_seed)
    n = len(diff)
    mean_val = float(np.mean(diff))

    # Fast batch bootstrap using matrix indexing
    batch_size = 1000
    resample_means = []

    for _ in range(n_resamples // batch_size):
        idx = rng.randint(0, n, size=(batch_size, n))
        b_means = np.mean(diff[idx], axis=1)
        resample_means.extend(b_means.tolist())

    resample_means = np.array(resample_means)
    ci_lower = float(np.percentile(resample_means, 2.5))
    ci_upper = float(np.percentile(resample_means, 97.5))
    spans_zero = bool(ci_lower <= 0.0 <= ci_upper)

    return {
        "num_samples": n,
        "mean_diff": mean_val,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "ci_width": float(ci_upper - ci_lower),
        "spans_zero": spans_zero,
        "std_err": float(np.std(resample_means)),
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-T3: Fold Stability & Paired Bootstrap")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V3_CENTERED_LATENT/results")
    args = parser.parse_args()

    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXP-T3] Starting Fold Stability & Paired Bootstrap Analysis ({args.resamples} resamples)...")

    # 1. Fold Stability Analysis from tiny_selector_per_fold.csv
    fold_csv_path = out_dir / "tiny_selector_per_fold.csv"
    if not fold_csv_path.exists():
        raise FileNotFoundError(f"Missing per-fold CSV: {fold_csv_path}. Run run_t1_tiny_selector_multiseed.py first.")

    fold_records = []
    with open(fold_csv_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            fold_records.append(row)

    stability_rows = []
    for f in FOLDS:
        delta_ades = []
        tiny_ades = []
        trans_ades = []

        for s in SEEDS:
            t_row = next(r for r in fold_records if r["Method"] == "Tiny_L5_Linear" and int(r["Seed"]) == s and r["Fold"] == f)
            tf_row = next(r for r in fold_records if r["Method"] == "Global_Transformer" and int(r["Seed"]) == s and r["Fold"] == f)

            t_ade = float(t_row["Selected_ADE"])
            tf_ade = float(tf_row["Selected_ADE"])
            d_ade = t_ade - tf_ade

            tiny_ades.append(t_ade)
            trans_ades.append(tf_ade)
            delta_ades.append(d_ade)

        avg_t_ade = float(np.mean(tiny_ades))
        avg_tf_ade = float(np.mean(trans_ades))
        avg_d_ade = float(np.mean(delta_ades))
        max_abs_d = float(np.max(np.abs(delta_ades)))
        stable_pass = bool(abs(avg_d_ade) <= 0.005)

        stability_rows.append({
            "Fold": f,
            "Avg_Tiny_ADE": f"{avg_t_ade:.4f}",
            "Avg_Transformer_ADE": f"{avg_tf_ade:.4f}",
            "Mean_Delta_ADE_m": f"{avg_d_ade:.4f}",
            "Max_Abs_Delta_ADE_m": f"{max_abs_d:.4f}",
            "Seed_1_Delta": f"{delta_ades[0]:.4f}",
            "Seed_2_Delta": f"{delta_ades[1]:.4f}",
            "Seed_3_Delta": f"{delta_ades[2]:.4f}",
            "Within_5mm_Bound": "PASS" if stable_pass else "WARN",
        })

    # Save stability CSV
    stab_csv_path = out_dir / "fold_stability_analysis.csv"
    with open(stab_csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(stability_rows[0].keys()))
        w.writeheader()
        w.writerows(stability_rows)

    print(f"  [Fold Stability] Saved -> {stab_csv_path}")

    # 2. Paired Bootstrap on predictions_cache.npz
    pred_cache_path = out_dir / "predictions_cache.npz"
    if not pred_cache_path.exists():
        raise FileNotFoundError(f"Missing predictions cache: {pred_cache_path}")

    pred_data = np.load(pred_cache_path)
    bootstrap_results: Dict[str, Any] = {
        "n_resamples": args.resamples,
        "seeds": {},
        "pooled": {},
        "per_fold": {},
    }

    pooled_diffs = []
    pooled_folds = []

    for s in SEEDS:
        tiny_ade = pred_data[f"seed_{s}_tiny_ade"]
        trans_ade = pred_data[f"seed_{s}_transformer_ade"]
        folds = pred_data[f"seed_{s}_fold"]
        diff = tiny_ade - trans_ade

        pooled_diffs.extend(diff.tolist())
        pooled_folds.extend(folds.tolist())

        seed_boot = run_paired_bootstrap(diff, n_resamples=args.resamples, rng_seed=42 + s)
        bootstrap_results["seeds"][f"seed_{s}"] = seed_boot
        print(f"  [Bootstrap Seed {s}] Mean Delta ADE: {seed_boot['mean_diff']*1000.0:.2f}mm | 95% CI: [{seed_boot['ci_95_lower']*1000.0:.2f}mm, {seed_boot['ci_95_upper']*1000.0:.2f}mm] | Spans 0: {seed_boot['spans_zero']}")

    # Overall Pooled Bootstrap
    pooled_arr = np.array(pooled_diffs)
    pooled_boot = run_paired_bootstrap(pooled_arr, n_resamples=args.resamples, rng_seed=123)
    bootstrap_results["pooled"] = pooled_boot
    print(f"  [Pooled Bootstrap] Mean Delta ADE: {pooled_boot['mean_diff']*1000.0:.2f}mm | 95% CI: [{pooled_boot['ci_95_lower']*1000.0:.2f}mm, {pooled_boot['ci_95_upper']*1000.0:.2f}mm] | Spans 0: {pooled_boot['spans_zero']}")

    # Per-fold Pooled Bootstrap
    pooled_folds_arr = np.array(pooled_folds)
    for f in FOLDS:
        f_mask = pooled_folds_arr == f
        f_diff = pooled_arr[f_mask]
        f_boot = run_paired_bootstrap(f_diff, n_resamples=args.resamples, rng_seed=456)
        bootstrap_results["per_fold"][f] = f_boot
        print(f"    Fold {f:>6}: Mean Delta ADE = {f_boot['mean_diff']*1000.0:.2f}mm | 95% CI: [{f_boot['ci_95_lower']*1000.0:.2f}mm, {f_boot['ci_95_upper']*1000.0:.2f}mm]")

    # Save JSON
    boot_json_path = out_dir / "paired_bootstrap.json"
    with open(boot_json_path, "w", encoding="utf-8") as fp:
        json.dump(bootstrap_results, fp, indent=2)

    print(f"[EXP-T3 OK] Paired bootstrap saved to {boot_json_path}")


if __name__ == "__main__":
    main()
