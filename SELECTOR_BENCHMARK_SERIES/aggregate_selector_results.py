#!/usr/bin/env python
"""
Aggregate Selector Results across seeds and folds.

Generates LEADERBOARD.csv and SUMMARY.json summarizing screening or confirmatory runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Aggregate Selector Benchmark Results")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=str, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    input_base = (repo_root / args.input_root).resolve()
    out_dir = (repo_root / args.output_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[AGGREGATE] Aggregating from {input_base} for seeds={args.seeds} -> {out_dir}")

    # Discover all methods across all seeds
    seed_method_metrics: Dict[int, Dict[str, Dict[str, float]]] = {s: {} for s in args.seeds}
    all_methods = set()

    for s in args.seeds:
        seed_dir = input_base / f"seed_{s}"
        if not seed_dir.exists():
            print(f"  [WARN] Seed directory missing: {seed_dir}")
            continue

        for method_dir in seed_dir.iterdir():
            if method_dir.is_dir():
                m_file = method_dir / "metrics.json"
                if m_file.exists():
                    try:
                        data = json.loads(m_file.read_text(encoding="utf-8"))
                        macro = data.get("macro", {})
                        if macro:
                            method_name = method_dir.name
                            seed_method_metrics[s][method_name] = macro
                            all_methods.add(method_name)
                    except Exception as e:
                        print(f"  [WARN] Failed to read {m_file}: {e}")

    method_list = sorted(list(all_methods))
    if not method_list:
        print("[AGGREGATE WARN] No valid method metrics found!")
        return

    # Sort methods putting 'native' first, then by ADE regret
    metric_keys = [
        "selected_ade", "selected_fde", "ade_regret", "fde_regret",
        "collision_rate", "near_collision_rate", "oracle_gap_closure",
        "intervention_rate", "false_intervention_rate", "q5_clearance",
        "bottom_5_cvar", "oracle_hit_rate"
    ]

    method_summary = {}
    for m in method_list:
        m_stat = {}
        for k in metric_keys:
            vals = [seed_method_metrics[s][m][k] for s in args.seeds if m in seed_method_metrics[s]]
            if vals:
                m_stat[f"{k}_mean"] = float(np.mean(vals))
                m_stat[f"{k}_std"] = float(np.std(vals))
            else:
                m_stat[f"{k}_mean"] = float("nan")
                m_stat[f"{k}_std"] = 0.0
        method_summary[m] = m_stat

    def sort_key(item):
        m = item[0]
        if m == "native":
            return -999.0
        return item[1].get("ade_regret_mean", 999.0)

    sorted_methods = sorted(method_summary.items(), key=sort_key)

    # Write LEADERBOARD.csv
    lb_path = out_dir / "LEADERBOARD.csv"
    multi_seed = len(args.seeds) > 1

    with open(lb_path, "w", newline="", encoding="utf-8") as fp:
        fields = [
            "Method", "Selected_ADE", "Selected_FDE", "ADE_Regret", "FDE_Regret",
            "Collision_Rate", "Near_Collision_Rate", "Oracle_Gap_Closure",
            "Intervention_Rate", "False_Intervention_Rate", "Q5_Clearance", "CVaR", "Oracle_Hit_Rate"
        ]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()

        for m, s in sorted_methods:
            if multi_seed:
                row = {
                    "Method": m,
                    "Selected_ADE": f"{s['selected_ade_mean']:.4f} ± {s['selected_ade_std']:.4f}",
                    "Selected_FDE": f"{s['selected_fde_mean']:.4f} ± {s['selected_fde_std']:.4f}",
                    "ADE_Regret": f"{s['ade_regret_mean']:.4f} ± {s['ade_regret_std']:.4f}",
                    "FDE_Regret": f"{s['fde_regret_mean']:.4f} ± {s['fde_regret_std']:.4f}",
                    "Collision_Rate": f"{s['collision_rate_mean']*100:.2f}% ± {s['collision_rate_std']*100:.2f}%",
                    "Near_Collision_Rate": f"{s['near_collision_rate_mean']*100:.2f}% ± {s['near_collision_rate_std']*100:.2f}%",
                    "Oracle_Gap_Closure": f"{s['oracle_gap_closure_mean']*100:.2f}% ± {s['oracle_gap_closure_std']*100:.2f}%",
                    "Intervention_Rate": f"{s['intervention_rate_mean']*100:.2f}%",
                    "False_Intervention_Rate": f"{s['false_intervention_rate_mean']*100:.2f}%",
                    "Q5_Clearance": f"{s['q5_clearance_mean']:.3f}m",
                    "CVaR": f"{s['bottom_5_cvar_mean']:.3f}m",
                    "Oracle_Hit_Rate": f"{s['oracle_hit_rate_mean']*100:.2f}%",
                }
            else:
                row = {
                    "Method": m,
                    "Selected_ADE": f"{s['selected_ade_mean']:.4f}",
                    "Selected_FDE": f"{s['selected_fde_mean']:.4f}",
                    "ADE_Regret": f"{s['ade_regret_mean']:.4f}",
                    "FDE_Regret": f"{s['fde_regret_mean']:.4f}",
                    "Collision_Rate": f"{s['collision_rate_mean']*100:.2f}%",
                    "Near_Collision_Rate": f"{s['near_collision_rate_mean']*100:.2f}%",
                    "Oracle_Gap_Closure": f"{s['oracle_gap_closure_mean']*100:.2f}%",
                    "Intervention_Rate": f"{s['intervention_rate_mean']*100:.2f}%",
                    "False_Intervention_Rate": f"{s['false_intervention_rate_mean']*100:.2f}%",
                    "Q5_Clearance": f"{s['q5_clearance_mean']:.3f}m",
                    "CVaR": f"{s['bottom_5_cvar_mean']:.3f}m",
                    "Oracle_Hit_Rate": f"{s['oracle_hit_rate_mean']*100:.2f}%",
                }
            writer.writerow(row)

    # Write summary.json
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "seeds": args.seeds,
        "total_methods": len(sorted_methods),
        "methods": dict(sorted_methods),
    }, indent=2), encoding="utf-8")

    print(f"[AGGREGATE OK] Processed {len(sorted_methods)} methods -> {lb_path}")


if __name__ == "__main__":
    main()
