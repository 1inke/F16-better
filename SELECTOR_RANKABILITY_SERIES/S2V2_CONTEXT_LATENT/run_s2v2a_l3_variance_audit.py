#!/usr/bin/env python
"""
S2-v2A: L3 Mode Latent Variance Audit.

Audits whether the previously extracted L3 latent (from decoder.mode_mlp) is sample-conditioned
or merely fixed/weakly-conditioned mode embeddings.

Metrics computed across modes and samples:
1. within-mode cross-sample variance
2. same-mode cross-sample cosine similarity
3. different-mode same-sample cosine similarity
4. between-mode centroid distance
5. between-mode variance
6. Variance ratio R = Var(between-mode) / Var(within-mode)

Outputs:
- results/l3_variance_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO / "SELECTOR_BENCHMARK_SERIES")]

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_a = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8
    norm_b = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8
    return np.sum((a / norm_a) * (b / norm_b), axis=-1)


def audit_l3_latents(latent_dir: Path) -> Dict[str, Any]:
    all_l3_list = []
    fold_stats = {}

    for fold in FOLDS:
        npz_file = latent_dir / fold / "latents.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Missing L3 latent cache: {npz_file}")
        data = np.load(npz_file)
        l3 = data["mode_latent"] # (N, 20, D)
        all_l3_list.append(l3)

        N, K, D = l3.shape

        # 1. Within-mode cross-sample variance
        var_within = float(np.mean(np.var(l3, axis=0))) # average over modes and feature dims

        # 2. Same-mode cross-sample cosine
        # Sample pairs
        np.random.seed(42)
        idx_a = np.random.choice(N, min(1000, N), replace=False)
        idx_b = np.random.choice(N, min(1000, N), replace=False)
        same_mode_cos = float(np.mean(cosine_sim(l3[idx_a], l3[idx_b])))

        # 3. Different-mode same-sample cosine
        diff_mode_cosines = []
        for i in range(min(500, N)):
            for m1 in range(min(5, K)):
                for m2 in range(m1 + 1, min(5, K)):
                    cos_val = cosine_sim(l3[i, m1:m1+1], l3[i, m2:m2+1])[0]
                    diff_mode_cosines.append(cos_val)
        diff_mode_cos = float(np.mean(diff_mode_cosines))

        # 4. Between-mode variance of sample-averaged centroids
        mode_centroids = np.mean(l3, axis=0) # (20, D)
        var_between = float(np.mean(np.var(mode_centroids, axis=0)))

        # 5. Centroid distances
        dists = []
        for m1 in range(K):
            for m2 in range(m1 + 1, K):
                d = np.linalg.norm(mode_centroids[m1] - mode_centroids[m2])
                dists.append(d)
        mean_centroid_dist = float(np.mean(dists))

        ratio = var_between / (var_within + 1e-9)

        fold_stats[fold] = {
            "num_samples": N,
            "latent_dim": D,
            "within_mode_variance": var_within,
            "between_mode_variance": var_between,
            "var_ratio_between_to_within": ratio,
            "same_mode_cross_sample_cosine": same_mode_cos,
            "diff_mode_same_sample_cosine": diff_mode_cos,
            "mean_centroid_distance": mean_centroid_dist,
        }

    # Macro aggregation
    macro_stats = {
        "within_mode_variance": float(np.mean([fold_stats[f]["within_mode_variance"] for f in FOLDS])),
        "between_mode_variance": float(np.mean([fold_stats[f]["between_mode_variance"] for f in FOLDS])),
        "var_ratio_between_to_within": float(np.mean([fold_stats[f]["var_ratio_between_to_within"] for f in FOLDS])),
        "same_mode_cross_sample_cosine": float(np.mean([fold_stats[f]["same_mode_cross_sample_cosine"] for f in FOLDS])),
        "diff_mode_same_sample_cosine": float(np.mean([fold_stats[f]["diff_mode_same_sample_cosine"] for f in FOLDS])),
        "mean_centroid_distance": float(np.mean([fold_stats[f]["mean_centroid_distance"] for f in FOLDS])),
    }

    # Assessment
    is_sample_conditioned = bool(
        macro_stats["same_mode_cross_sample_cosine"] < 0.95
        and macro_stats["within_mode_variance"] > 0.05
    )

    result = {
        "macro_summary": macro_stats,
        "is_sample_conditioned": is_sample_conditioned,
        "verdict": (
            "SAMPLE_CONDITIONED" if is_sample_conditioned else "WEAKLY_CONDITIONED_MODE_EMBEDDING"
        ),
        "per_fold": fold_stats,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="S2-v2A L3 Variance Audit")
    parser.add_argument("--latent-root", type=str, default="SELECTOR_RANKABILITY_SERIES/S2_LATENT_PROBE/latent_cache/seed_1")
    parser.add_argument("--output-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V2_CONTEXT_LATENT/results")
    args = parser.parse_args()

    latent_dir = (REPO / args.latent_root).resolve()
    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[S2-v2A] Auditing L3 mode latent variance from {latent_dir}...")
    stats = audit_l3_latents(latent_dir)

    out_file = out_dir / "l3_variance_audit.json"
    out_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[S2-v2A OK] Verdict: {stats['verdict']} -> {out_file}")
    print(f"  Macro same-mode cross-sample cosine: {stats['macro_summary']['same_mode_cross_sample_cosine']:.4f}")
    print(f"  Macro within-mode variance: {stats['macro_summary']['within_mode_variance']:.6f}")
    print(f"  Macro between-mode variance: {stats['macro_summary']['between_mode_variance']:.6f}")
    print(f"  Macro ratio (between/within): {stats['macro_summary']['var_ratio_between_to_within']:.2f}")


if __name__ == "__main__":
    main()
