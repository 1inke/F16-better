#!/usr/bin/env python
"""
EXP-C1: Centered Latent Construction.

Computes sample-centered mode latents:
  z'_{i, k} = z_{i, k} - (1/K) * sum_{j=1}^K z_{i, j}
for L3, L4, and L5 across all 5 folds.

Outputs per fold:
- l3_centered.npy: (N, 20, 64)
- l4_centered.npy: (N, 20, 128)
- l5_centered.npy: (N, 20, 70)
- manifest.json: metadata and validation stats

Outputs globally:
- centered_alignment.json: verification across all folds
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO / "SELECTOR_BENCHMARK_SERIES")]

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]


def center_modes(z: np.ndarray) -> np.ndarray:
    """Center latent representation across candidates for each sample in double precision."""
    # z: [N, K, D]
    mean64 = z.mean(axis=1, keepdims=True, dtype=np.float64)
    return (z.astype(np.float64) - mean64).astype(np.float32)


def build_centered_latents(
    src_dir: Path,
    dst_dir: Path,
    seed: int = 1,
    feature_root: Path | None = None,
) -> Dict[str, Any]:
    dst_seed_dir = dst_dir / f"seed_{seed}"
    dst_seed_dir.mkdir(parents=True, exist_ok=True)

    alignment_report: Dict[str, Any] = {
        "status": "PASS",
        "seed": seed,
        "folds": {},
    }

    for fold in FOLDS:
        src_npz = src_dir / f"seed_{seed}" / fold / "latents.npz"
        if not src_npz.exists():
            raise FileNotFoundError(f"Missing source latents: {src_npz}")

        raw = np.load(src_npz)
        l3 = raw["l3"] # (N, 20, 64)
        l4 = raw["l4"] # (N, 20, 128)
        l5 = raw["l5"] # (N, 20, 70)

        N, K, _ = l3.shape
        assert K == 20, f"Expected K=20, got {K}"

        # Check alignment with feature bank if provided
        if feature_root is not None:
            feat_npz = feature_root / f"seed_{seed}" / fold / "features.npz"
            if feat_npz.exists():
                feat_data = np.load(feat_npz)
                assert len(feat_data["ade"]) == N, f"Sample count mismatch: {N} vs {len(feat_data['ade'])}"

        l3_c = center_modes(l3)
        l4_c = center_modes(l4)
        l5_c = center_modes(l5)

        # Verification: abs(centered.mean(axis=1)).max() < 1e-6
        max_mean_l3 = float(np.abs(l3_c.mean(axis=1)).max())
        max_mean_l4 = float(np.abs(l4_c.mean(axis=1)).max())
        max_mean_l5 = float(np.abs(l5_c.mean(axis=1)).max())

        assert max_mean_l3 < 1e-6, f"L3 centering residual exceeds threshold: {max_mean_l3}"
        assert max_mean_l4 < 1e-6, f"L4 centering residual exceeds threshold: {max_mean_l4}"
        assert max_mean_l5 < 1e-6, f"L5 centering residual exceeds threshold: {max_mean_l5}"

        fold_dst = dst_seed_dir / fold
        fold_dst.mkdir(parents=True, exist_ok=True)

        np.save(fold_dst / "l3_centered.npy", l3_c)
        np.save(fold_dst / "l4_centered.npy", l4_c)
        np.save(fold_dst / "l5_centered.npy", l5_c)

        manifest = {
            "fold": fold,
            "seed": seed,
            "num_samples": N,
            "k_modes": K,
            "shapes": {
                "l3_centered": list(l3_c.shape),
                "l4_centered": list(l4_c.shape),
                "l5_centered": list(l5_c.shape),
            },
            "centering_residuals": {
                "l3": max_mean_l3,
                "l4": max_mean_l4,
                "l5": max_mean_l5,
            },
            "verification": "PASS",
        }

        with open(fold_dst / "manifest.json", "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)

        alignment_report["folds"][fold] = manifest
        print(f"  [{fold}] Processed {N} samples. Residuals: L3={max_mean_l3:.2e}, L4={max_mean_l4:.2e}, L5={max_mean_l5:.2e} -> PASS")

    return alignment_report


def main():
    parser = argparse.ArgumentParser(description="EXP-C1: Centered Latent Construction")
    parser.add_argument("--src-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V2_CONTEXT_LATENT/latent_cache_v2")
    parser.add_argument("--dst-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V3_CENTERED_LATENT/centered_cache")
    parser.add_argument("--feature-root", type=str, default="SELECTOR_BENCHMARK_SERIES/features")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V3_CENTERED_LATENT/results")
    args = parser.parse_args()

    src_dir = (REPO / args.src_dir).resolve()
    dst_dir = (REPO / args.dst_dir).resolve()
    feat_root = (REPO / args.feature_root).resolve()
    out_dir = (REPO / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXP-C1] Constructing Centered Latents for Seed {args.seed}...")
    report = build_centered_latents(src_dir, dst_dir, seed=args.seed, feature_root=feat_root)

    align_path = out_dir / "centered_alignment.json"
    with open(align_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(f"[EXP-C1 OK] Centered alignment report written to {align_path}")


if __name__ == "__main__":
    main()
