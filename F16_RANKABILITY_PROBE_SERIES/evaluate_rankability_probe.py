"""
Comprehensive Evaluator for Rankability Probes (R0 - R7).

Enforces:
- F16_RANKABILITY_CORRECTION_AND_TOP1_PROBE_GUIDE.md
- Corrected pairwise accuracy (delta = ADE_j - ADE_i > min_delta)
- Oracle margin accuracy @10cm, @20cm
- Pairwise probe comparison bootstrap CIs (R2-R1, R3-R2, R4-R2, R4-R3, R5-R3, R7-R5)
- Descriptive validation bootstrap labeling (avoiding model-selection bias claim)
- Hardcase breakdowns based on selection regret (>10cm, >20cm, >30cm, near-tie <2cm)
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
from metrics import (
    evaluate_probe_predictions,
    compute_bootstrap_ci,
    compute_pairwise_accuracy,
    compute_oracle_margin_accuracy,
)


def compute_pairwise_probe_ci(
    probe_a_ades: np.ndarray,
    probe_b_ades: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 20260921,
) -> dict:
    """
    Computes paired bootstrap difference: probe_a - probe_b.
    Positive value means probe_b is better than probe_a (ADE_a > ADE_b).
    """
    rng = np.random.RandomState(seed)
    N = len(probe_a_ades)
    diff = probe_a_ades - probe_b_ades
    mean_diff = float(np.mean(diff))

    boot_means = np.empty(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        idx = rng.randint(0, N, size=N)
        boot_means[b] = np.mean(diff[idx])

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    # Excludes zero if both bounds have the same sign
    is_sig = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)

    return {
        "mean_diff": mean_diff,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "is_significant_descriptive": is_sig,
    }


def run_comprehensive_evaluation(seed: int = 1, n_bootstraps: int = 1000) -> None:
    print(f"[EVAL-FULL] Running comprehensive probe evaluation (Seed {seed}, Bootstraps {n_bootstraps})...")
    val_cache_file = HERE / "results" / "cache" / f"val_seed{seed}.npz"
    if not val_cache_file.is_file():
        raise FileNotFoundError(f"Missing val cache file: {val_cache_file}")

    val_data = dict(np.load(val_cache_file))
    N = len(val_data["sample_id"])
    out_base = HERE / "results" / "p2_rankability"

    native_logits = val_data["native_logits"]
    native_selected_idx = val_data["native_selected_index"]
    native_ades = val_data["ADE_final"][np.arange(N), native_selected_idx]
    oracle_ades = val_data["ADE_final"][np.arange(N), val_data["oracle_final_index"]]

    # Load all probe predictions
    # Single-run probes R1-R4
    probe_dict = {
        "Native_R0": native_logits,
        "R1_exact_input": np.load(out_base / "R1_exact_input" / "val_scores.npy"),
        "R2_post_correction": np.load(out_base / "R2_post_correction" / "val_scores.npy"),
        "R3_context_post": np.load(out_base / "R3_context_post" / "val_scores.npy"),
        "R4_full_final": np.load(out_base / "R4_full_final" / "val_scores.npy"),
    }

    # Top-1 Aligned Probes R5-R7 across seeds 1, 2, 3
    for p in ["R5_oracle_vs_rest", "R6_listwise_top1", "R7_set_aware_ranker"]:
        for s in [1, 2, 3]:
            score_p = out_base / f"{p}_s{s}" / "val_scores.npy"
            if score_p.exists():
                probe_dict[f"{p}_s{s}"] = np.load(score_p)

    # 1. Summary CSV
    summary_rows = []
    ades_by_probe = {}

    for name, scores in probe_dict.items():
        m = evaluate_probe_predictions(scores=scores, cache_data=val_data)
        summary_rows.append({"probe_name": name, **m})

        sel_idx = np.argmax(scores, axis=-1)
        ades_by_probe[name] = val_data["ADE_final"][np.arange(N), sel_idx]

    # Compute Seed Average rows for R5, R6, R7
    for p in ["R5_oracle_vs_rest", "R6_listwise_top1", "R7_set_aware_ranker"]:
        seed_names = [f"{p}_s{s}" for s in [1, 2, 3] if f"{p}_s{s}" in probe_dict]
        if len(seed_names) == 3:
            avg_row = {"probe_name": f"{p}_mean_3seeds"}
            keys = [k for k in summary_rows[0].keys() if k != "probe_name"]
            for k in keys:
                vals = [r[k] for r in summary_rows if r["probe_name"] in seed_names]
                avg_row[k] = float(np.mean(vals))
            summary_rows.append(avg_row)

    with open(out_base / "probe_summary.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # 2. Bootstrap CI vs Native
    bootstrap_rows = []
    for name, p_ades in ades_by_probe.items():
        boot_res = compute_bootstrap_ci(
            native_ades=native_ades,
            probe_ades=p_ades,
            n_bootstraps=n_bootstraps,
            seed=20260921,
        )
        bootstrap_rows.append({
            "probe_name": name,
            "mean_delta_ade": boot_res["mean_delta_ade"],
            "ci_lower_95": boot_res["ci_lower_95"],
            "ci_upper_95": boot_res["ci_upper_95"],
            "is_significant_descriptive": boot_res["is_significant"],
            "note": "Descriptive paired bootstrap on selected validation checkpoint",
        })

    with open(out_base / "bootstrap_ci.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(bootstrap_rows[0].keys()))
        w.writeheader()
        w.writerows(bootstrap_rows)

    # 3. Pairwise Probe Comparisons CI (§11)
    comparisons = [
        ("R1_exact_input", "R2_post_correction", "R2_vs_R1 (Post-Correction Geometry Gain)"),
        ("R2_post_correction", "R3_context_post", "R3_vs_R2 (Explicit Context Gain)"),
        ("R2_post_correction", "R4_full_final", "R4_vs_R2 (Full Trajectory vs Anchor Geometry)"),
        ("R3_context_post", "R4_full_final", "R4_vs_R3 (Full Trajectory vs Context+Anchor)"),
        ("R3_context_post", "R5_oracle_vs_rest_s1", "R5_vs_R3 (Oracle-vs-Rest Objective Gain)"),
        ("R5_oracle_vs_rest_s1", "R7_set_aware_ranker_s1", "R7_vs_R5 (Set-Aware Ranker Architecture Gain)"),
    ]

    pairwise_comp_rows = []
    for p_a, p_b, desc in comparisons:
        if p_a in ades_by_probe and p_b in ades_by_probe:
            # delta = ADE_a - ADE_b (>0 means probe_b has lower ADE, i.e. probe_b is better)
            res = compute_pairwise_probe_ci(
                probe_a_ades=ades_by_probe[p_a],
                probe_b_ades=ades_by_probe[p_b],
                n_bootstraps=n_bootstraps,
                seed=20260921,
            )
            pairwise_comp_rows.append({
                "comparison": f"{p_b} - {p_a}",
                "description": desc,
                "mean_improvement_m": res["mean_diff"],
                "ci_lower_95": res["ci_lower_95"],
                "ci_upper_95": res["ci_upper_95"],
                "is_significant_descriptive": res["is_significant_descriptive"],
            })

    with open(out_base / "pairwise_probe_comparison_ci.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(pairwise_comp_rows[0].keys()))
        w.writeheader()
        w.writerows(pairwise_comp_rows)

    # 4. Hard-case Subsets
    native_regret = native_ades - oracle_ades
    sorted_ades = np.sort(val_data["ADE_final"], axis=-1)
    gap12 = sorted_ades[:, 1] - sorted_ades[:, 0]

    hardcase_masks = {
        "all": (np.ones(N, dtype=bool), "All Validation Samples"),
        "regret_gt_10cm": (native_regret > 0.10, "Native Regret > 0.10 m (Mild Selection Failure)"),
        "regret_gt_20cm": (native_regret > 0.20, "Native Regret > 0.20 m (Severe Selection Failure)"),
        "regret_gt_30cm": (native_regret > 0.30, "Native Regret > 0.30 m (Catastrophic Selection Failure)"),
        "near_tie_gap12_lt_02cm": (gap12 < 0.02, "Near-tie Ambiguity (Delta ADE_1,2 < 0.02 m)"),
    }

    # Representatives for hardcase breakdown
    rep_probes = [
        "Native_R0",
        "R1_exact_input",
        "R2_post_correction",
        "R3_context_post",
        "R4_full_final",
        "R5_oracle_vs_rest_s1",
        "R6_listwise_top1_s3",
        "R7_set_aware_ranker_s2",
    ]

    hardcase_rows = []
    for subset_name, (mask, desc) in hardcase_masks.items():
        n_sub = int(np.sum(mask))
        if n_sub == 0:
            continue

        sub_cache = {k: v[mask] for k, v in val_data.items() if isinstance(v, np.ndarray) and len(v) == N}
        sub_native_sel_ade = float(np.mean(native_ades[mask]))
        sub_oracle_ade = float(np.mean(oracle_ades[mask]))

        for p_name in rep_probes:
            if p_name not in probe_dict:
                continue
            sub_scores = probe_dict[p_name][mask]
            sub_m = evaluate_probe_predictions(
                scores=sub_scores,
                cache_data=sub_cache,
                native_selected_ade=sub_native_sel_ade,
                oracle_min_ade=sub_oracle_ade,
            )
            hardcase_rows.append({
                "probe_name": p_name,
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
                "oracle_margin_10": sub_m["oracle_margin_10"],
                "oracle_margin_20": sub_m["oracle_margin_20"],
            })

    with open(out_base / "hardcase_summary.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(hardcase_rows[0].keys()))
        w.writeheader()
        w.writerows(hardcase_rows)

    print("[EVAL-FULL] Generated probe_summary.csv, bootstrap_ci.csv, pairwise_probe_comparison_ci.csv, hardcase_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    run_comprehensive_evaluation(seed=args.seed, n_bootstraps=args.bootstrap)
