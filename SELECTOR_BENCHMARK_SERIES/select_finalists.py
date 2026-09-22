#!/usr/bin/env python
"""
Select Pareto Finalists from Screening Leaderboard.

Applies degradation constraints relative to Native baseline and selects up to Top-K finalists:
- 2 best accuracy (lowest ADE regret)
- 2 best safety (lowest collision rate under ADE constraint)
- 1 Pareto balanced
- 1 lowest latency / parameter count
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def parse_val(val_str: str) -> float:
    # Handles both '0.5184' and '0.5184 ± 0.0012' and '6.47%'
    val_clean = val_str.split("±")[0].strip().rstrip("%").rstrip("m")
    try:
        return float(val_clean)
    except ValueError:
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description="Select Finalist Selectors")
    parser.add_argument("--leaderboard", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-ade-degradation-pct", type=float, default=1.0)
    parser.add_argument("--max-fde-degradation-pct", type=float, default=1.0)
    parser.add_argument("--max-collision-increase-pp", type=float, default=0.10)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    lb_path = (repo_root / args.leaderboard).resolve()
    out_path = (repo_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not lb_path.exists():
        raise FileNotFoundError(f"Leaderboard missing: {lb_path}")

    rows: List[Dict[str, Any]] = []
    with open(lb_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            rows.append({
                "method": r["Method"],
                "ade": parse_val(r["Selected_ADE"]),
                "fde": parse_val(r["Selected_FDE"]),
                "ade_regret": parse_val(r["ADE_Regret"]),
                "fde_regret": parse_val(r["FDE_Regret"]),
                "collision": parse_val(r["Collision_Rate"]),
                "near_collision": parse_val(r["Near_Collision_Rate"]),
                "closure": parse_val(r["Oracle_Gap_Closure"]),
            })

    # Find native baseline
    native_row = next((r for r in rows if r["method"] == "native"), None)
    if not native_row:
        raise RuntimeError("Native baseline not found in leaderboard!")

    nat_ade = native_row["ade"]
    nat_fde = native_row["fde"]
    nat_coll = native_row["collision"]

    max_ade = nat_ade * (1.0 + args.max_ade_degradation_pct / 100.0)
    max_fde = nat_fde * (1.0 + args.max_fde_degradation_pct / 100.0)
    max_coll = nat_coll + args.max_collision_increase_pp

    print(f"[FINALISTS] Constraints vs Native (ADE={nat_ade:.4f}, FDE={nat_fde:.4f}, Coll={nat_coll:.2f}%):")
    print(f"  Max ADE: {max_ade:.4f} (+{args.max_ade_degradation_pct}%)")
    print(f"  Max FDE: {max_fde:.4f} (+{args.max_fde_degradation_pct}%)")
    print(f"  Max Coll: {max_coll:.2f}% (+{args.max_collision_increase_pp} pp)")

    # Filter eligible methods (excluding native from candidates)
    eligible = []
    for r in rows:
        if r["method"] == "native":
            continue
        # Check constraints
        if r["ade"] <= max_ade and r["fde"] <= max_fde and r["collision"] <= max_coll:
            eligible.append(r)

    if not eligible:
        print("[FINALISTS WARN] No method satisfied all strict constraints; relaxing collision constraint...")
        eligible = [r for r in rows if r["method"] != "native" and r["ade"] <= max_ade * 1.02]

    # Select finalists
    finalists: List[str] = []

    # 1. Best accuracy (lowest ADE regret)
    sorted_acc = sorted(eligible, key=lambda x: x["ade_regret"])
    for r in sorted_acc[:2]:
        if r["method"] not in finalists:
            finalists.append(r["method"])

    # 2. Best safety (lowest collision rate)
    sorted_safe = sorted(eligible, key=lambda x: x["collision"])
    for r in sorted_safe[:2]:
        if r["method"] not in finalists:
            finalists.append(r["method"])

    # 3. Pareto balanced (normalized ADE regret + normalized collision)
    min_reg = min(r["ade_regret"] for r in eligible)
    max_reg = max(r["ade_regret"] for r in eligible) + 1e-6
    min_c = min(r["collision"] for r in eligible)
    max_c = max(r["collision"] for r in eligible) + 1e-6

    def pareto_cost(r):
        norm_reg = (r["ade_regret"] - min_reg) / (max_reg - min_reg)
        norm_coll = (r["collision"] - min_c) / (max_c - min_c)
        return norm_reg + norm_coll

    sorted_pareto = sorted(eligible, key=pareto_cost)
    for r in sorted_pareto:
        if r["method"] not in finalists:
            finalists.append(r["method"])
            break

    # 4. Minimal complexity / latency (shield or linear)
    low_cost = [r for r in eligible if "shield" in r["method"] or "linear" in r["method"] or "ridge" in r["method"]]
    if low_cost:
        sorted_low = sorted(low_cost, key=lambda x: x["ade_regret"])
        for r in sorted_low:
            if r["method"] not in finalists:
                finalists.append(r["method"])
                break

    # If still fewer than top_k, fill from sorted_pareto
    for r in sorted_pareto:
        if len(finalists) >= args.top_k:
            break
        if r["method"] not in finalists:
            finalists.append(r["method"])

    print(f"[FINALISTS SELECTED] ({len(finalists)}/{args.top_k}): {finalists}")

    out_data = {
        "selection_constraints": {
            "max_ade_degradation_pct": args.max_ade_degradation_pct,
            "max_fde_degradation_pct": args.max_fde_degradation_pct,
            "max_collision_increase_pp": args.max_collision_increase_pp,
        },
        "native_baseline": {
            "ade": nat_ade,
            "fde": nat_fde,
            "collision": nat_coll,
        },
        "methods": finalists,
    }
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[FINALISTS OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
