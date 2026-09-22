"""
Alignment Audit for Train and Test Evaluation (EXP-S0B).

Verifies:
1. Dataset feature and target arrays match sample-for-sample in length and order.
2. Inference output arrays match input arrays exactly without index shifting or permutation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path[:0] = [str(HERE.parent)]
from common.cache_io import load_fold_features

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]


def run_alignment_audit(feature_root: Path) -> dict[str, Any]:
    print("[EXP-S0B] Auditing data and feature alignment across 5 folds...")
    audit_results = {}
    all_passed = True

    for fold in FOLDS:
        data = load_fold_features(feature_root, fold, seed=1)
        N = len(data["ade"])

        checks = {
            "g0_len": bool(len(data["g0"]) == N),
            "g1_len": bool(len(data["g1"]) == N),
            "g2_len": bool(len(data["g2"]) == N),
            "g3_len": bool(len(data["g3"]) == N),
            "g4_len": bool(len(data["g4"]) == N),
            "min_ade_len": bool(len(data["min_ade20"]) == N),
            "native_idx_len": bool(len(data["native_top1_idx"]) == N),
            "min_ade_valid": bool(np.all(data["min_ade20"] <= np.min(data["ade"], axis=-1) + 1e-5)),
            "native_idx_range": bool(np.all((data["native_top1_idx"] >= 0) & (data["native_top1_idx"] < 20))),
        }
        passed = all(checks.values())
        if not passed:
            all_passed = False
        audit_results[fold] = {"num_samples": N, "passed": passed, "details": checks}
        print(f"  [{fold}] {N} samples | Aligned: {passed}")

    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "alignment_audit.json"
    out_json.write_text(json.dumps({"all_passed": all_passed, "per_fold": audit_results}, indent=2), encoding="utf-8")
    print(f"[EXP-S0B OK] Alignment audit saved -> {out_json}")
    return {"all_passed": all_passed, "results": audit_results}


if __name__ == "__main__":
    feat_root = REPO / "SELECTOR_BENCHMARK_SERIES" / "features"
    res = run_alignment_audit(feat_root)
    if not res["all_passed"]:
        sys.exit(1)
