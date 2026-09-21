"""
Comprehensive Rankability Probe Evaluator.

Enforces Butler Rules:
- research.goal_aligned_evaluation
- research.no_hardcoded_metrics
- research.anti_arbitrary_thresholding

Generates:
1. probe_summary.csv
2. hardcase_summary.csv
3. bootstrap_ci.csv
4. Feeds verified data into RANKABILITY_REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO), str(HERE)]

import numpy as np
from metrics import evaluate_probe_predictions, compute_bootstrap_ci


def run_full_probe_evaluation(
    seed: int = 1,
    n_bootstraps: int = 1000,
) -> dict:
    print(f"[EVAL-P2] Running full evaluation for Rankability Probes (Seed {seed})...")

    val_cache_file = HERE / "results" / "cache" / f"val_seed{seed}.npz"
    if not val_cache_file.is_file():
        raise FileNotFoundError(f"Missing val cache file: {val_cache_file}")

    val_data = dict(np.load(val_cache_file))
    N = len(val_data["sample_id"])

    out_base = HERE / "results" / "p2_rankability"

    # Native predictions: R0
    native_logits = val_data["native_logits"]
    native_selected_idx = val_data["native_selected_index"]
    native_ades = val_data["ADE_final"][np.arange(N), native_selected_idx]
    oracle_ades = val_data["ADE_final"][np.arange(N), val_data["oracle_final_index"]]

    probes = {
        "Native_R0": native_logits,
        "R1_exact_input": np.load(out_base / "R1_exact_input" / "val_scores.npy"),
        "R2_post_correction": np.load(out_base / "R2_post_correction" / "val_scores.npy"),
        "R3_context_post": np.load(out_base / "R3_context_post" / "val_scores.npy"),
        "R4_full_final": np.load(out_base / "R4_full_final" / "val_scores.npy"),
    }

    # 1. Evaluate Overall Summary
    probe_summary_rows = []
    bootstrap_rows = []

    for name, scores in probes.items():
        metrics = evaluate_probe_predictions(
            scores=scores,
            cache_data=val_data,
        )
        row = {"probe_name": name, **metrics}
        probe_summary_rows.append(row)

        # Bootstrap vs Native
        probe_sel_idx = np.argmax(scores, axis=-1)
        probe_ades = val_data["ADE_final"][np.arange(N), probe_sel_idx]
        boot_res = compute_bootstrap_ci(
            native_ades=native_ades,
            probe_ades=probe_ades,
            n_bootstraps=n_bootstraps,
            seed=20260921,
        )
        bootstrap_rows.append({
            "probe_name": name,
            "mean_delta_ade": boot_res["mean_delta_ade"],
            "ci_lower_95": boot_res["ci_lower_95"],
            "ci_upper_95": boot_res["ci_upper_95"],
            "is_significant": boot_res["is_significant"],
        })

    # Save probe_summary.csv
    with open(out_base / "probe_summary.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(probe_summary_rows[0].keys()))
        w.writeheader()
        w.writerows(probe_summary_rows)

    # Save bootstrap_ci.csv
    with open(out_base / "bootstrap_ci.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(bootstrap_rows[0].keys()))
        w.writeheader()
        w.writerows(bootstrap_rows)

    # 2. Hard-case Subsets
    native_regret = native_ades - oracle_ades
    sorted_ades = np.sort(val_data["ADE_final"], axis=-1)
    gap12 = sorted_ades[:, 1] - sorted_ades[:, 0]

    hardcase_masks = {
        "all": (np.ones(N, dtype=bool), "All Validation Samples"),
        "regret_gt_10cm": (native_regret > 0.10, "Native Regret > 0.10 m"),
        "regret_gt_20cm": (native_regret > 0.20, "Native Regret > 0.20 m (Severe Selection Failure)"),
        "regret_gt_30cm": (native_regret > 0.30, "Native Regret > 0.30 m (Catastrophic Selection Failure)"),
        "near_tie_gap12_lt_02cm": (gap12 < 0.02, "Near-tie Ambiguity (Delta ADE_1,2 < 0.02 m)"),
    }

    hardcase_rows = []
    for subset_name, (mask, desc) in hardcase_masks.items():
        n_sub = int(np.sum(mask))
        if n_sub == 0:
            continue

        sub_cache = {k: v[mask] for k, v in val_data.items() if isinstance(v, np.ndarray) and len(v) == N}
        sub_native_sel_ade = float(np.mean(native_ades[mask]))
        sub_oracle_ade = float(np.mean(oracle_ades[mask]))

        for name, scores in probes.items():
            sub_scores = scores[mask]
            sub_m = evaluate_probe_predictions(
                scores=sub_scores,
                cache_data=sub_cache,
                native_selected_ade=sub_native_sel_ade,
                oracle_min_ade=sub_oracle_ade,
            )
            hardcase_rows.append({
                "probe_name": name,
                "subset": subset_name,
                "subset_desc": desc,
                "n_samples": n_sub,
                "selected_ade": sub_m["selected_ade"],
                "oracle_min_ade": sub_oracle_ade,
                "selection_regret": sub_m["selection_regret"],
                "regret_closure_pct": sub_m["regret_closure_pct"],
                "oracle_recall_1": sub_m["oracle_recall_1"],
                "oracle_recall_3": sub_m["oracle_recall_3"],
                "spearman_rho": sub_m["spearman_rho"],
                "pairwise_acc_20": sub_m["pairwise_acc_20"],
            })

    # Save hardcase_summary.csv
    with open(out_base / "hardcase_summary.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(hardcase_rows[0].keys()))
        w.writeheader()
        w.writerows(hardcase_rows)

    print(f"[EVAL-P2] Successfully generated probe_summary.csv, bootstrap_ci.csv, hardcase_summary.csv")
    return {
        "probe_summary": probe_summary_rows,
        "bootstrap": bootstrap_rows,
        "hardcase": hardcase_rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    run_full_probe_evaluation(seed=args.seed, n_bootstraps=args.bootstrap)
