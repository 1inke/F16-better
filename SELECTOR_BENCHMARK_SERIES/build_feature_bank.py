#!/usr/bin/env python
"""
Build Feature Bank for Candidate Trajectories.

Extracts G0 (Native Confidence), G1 (Candidate Geometry), G2 (Set-Relative),
G3 (History Social Context), and G4 (Model-Predicted Neighbor Future Risk) features
from cached candidate shards. Strict partition between deployable features and privileged labels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

COLLISION_THRESHOLD = 0.20


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_sample_features(sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Compute G0, G1, G2, G3, G4 features for one sample (20 candidates)."""
    past = sample["past"].numpy()                  # (N_agents, 8, 2)
    nv = sample["num_valid"]
    focal_cands = sample["focal_candidates"].numpy()  # (20, 12, 2)
    focal_logits = sample["focal_logits"].numpy()    # (20,)
    valid_nbrs = max(0, nv - 1)

    # ------------------------------------------------------------------
    # G0: Native Confidence Features (20, 5)
    # ------------------------------------------------------------------
    max_logit = np.max(focal_logits)
    exp_logits = np.exp(focal_logits - max_logit)
    probs = exp_logits / (np.sum(exp_logits) + 1e-12)
    order = np.argsort(-focal_logits)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(focal_logits))
    margin_to_top1 = max_logit - focal_logits
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    g0 = np.stack([
        focal_logits,
        probs,
        ranks.astype(np.float32),
        margin_to_top1,
        np.full(20, entropy, dtype=np.float32),
    ], axis=-1)  # (20, 5)

    # ------------------------------------------------------------------
    # G1: Candidate Intrinsic Geometry (20, 10)
    # ------------------------------------------------------------------
    step_diffs = focal_cands[:, 1:] - focal_cands[:, :-1]  # (20, 11, 2)
    step_lengths = np.linalg.norm(step_diffs, axis=-1)      # (20, 11)
    endpoint_disp = np.linalg.norm(focal_cands[:, -1] - focal_cands[:, 0], axis=-1)  # (20,)
    path_len = np.sum(step_lengths, axis=-1)  # (20,)
    mean_speed = path_len / 4.4               # (20,) 11 intervals * 0.4s
    max_speed = np.max(step_lengths, axis=-1) / 0.4

    acc_diffs = step_diffs[:, 1:] - step_diffs[:, :-1]       # (20, 10, 2)
    acc_norms = np.linalg.norm(acc_diffs, axis=-1) / 0.16     # (20, 10)
    mean_acc = np.mean(acc_norms, axis=-1)
    max_acc = np.max(acc_norms, axis=-1)

    jerk_diffs = acc_diffs[:, 1:] - acc_diffs[:, :-1]        # (20, 9, 2)
    mean_jerk = np.mean(np.linalg.norm(jerk_diffs, axis=-1), axis=-1) / 0.064

    # Curvature / angle changes
    angles = np.arctan2(step_diffs[:, :, 1], step_diffs[:, :, 0])  # (20, 11)
    angle_diffs = np.abs(np.diff(angles, axis=-1))
    angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)
    curvature = np.mean(angle_diffs, axis=-1)

    # Deviation from past CV extrapolation
    focal_past = past[0]  # (8, 2)
    v_last = (focal_past[7] - focal_past[6])  # (2,)
    t_steps = np.arange(1, 13)[:, None]       # (12, 1)
    cv_future = focal_past[7:8] + v_last[None, :] * t_steps  # (12, 2)
    dev_cv = np.mean(np.linalg.norm(focal_cands - cv_future[None, :, :], axis=-1), axis=-1)  # (20,)
    smoothness = -np.std(step_lengths, axis=-1)

    g1 = np.stack([
        endpoint_disp, path_len, mean_speed, max_speed,
        mean_acc, max_acc, mean_jerk, curvature, dev_cv, smoothness
    ], axis=-1)  # (20, 10)

    # ------------------------------------------------------------------
    # G2: Candidate-Set Relative Features (20, 7)
    # ------------------------------------------------------------------
    # Pairwise candidate distances (20, 20)
    cand_dist_matrix = np.mean(
        np.linalg.norm(focal_cands[:, None, :, :] - focal_cands[None, :, :, :], axis=-1),
        axis=-1
    )  # (20, 20)
    np.fill_diagonal(cand_dist_matrix, np.nan)
    mean_dist_others = np.nanmean(cand_dist_matrix, axis=-1)
    min_dist_others = np.nanmin(cand_dist_matrix, axis=-1)
    np.fill_diagonal(cand_dist_matrix, 0.0)

    # Medoid
    sum_dists = np.sum(cand_dist_matrix, axis=-1)
    medoid_idx = np.argmin(sum_dists)
    dist_to_medoid = cand_dist_matrix[:, medoid_idx]

    # Barycenter
    weighted_barycenter = np.sum(probs[:, None, None] * focal_cands, axis=0)  # (12, 2)
    dist_to_barycenter = np.mean(np.linalg.norm(focal_cands - weighted_barycenter[None, :, :], axis=-1), axis=-1)

    # Local density
    local_density = np.sum((cand_dist_matrix < 0.5) & (cand_dist_matrix > 0), axis=-1).astype(np.float32)

    # MBR cost: sum_j p_j * ADE(cand_i, cand_j)
    mbr_cost = np.sum(cand_dist_matrix * probs[None, :], axis=-1)

    # Diversity to top 3
    top3_indices = order[:3]
    div_top3 = np.min(cand_dist_matrix[:, top3_indices], axis=-1)

    g2 = np.stack([
        mean_dist_others, min_dist_others, dist_to_medoid,
        dist_to_barycenter, local_density, mbr_cost, div_top3
    ], axis=-1)  # (20, 7)

    # ------------------------------------------------------------------
    # G3: History / Social Context Features (20, 8)
    # ------------------------------------------------------------------
    focal_speed_hist = np.linalg.norm(focal_past[7] - focal_past[6]) / 0.4
    focal_acc_hist = np.linalg.norm((focal_past[7] - focal_past[6]) - (focal_past[6] - focal_past[5])) / 0.16
    focal_turn_hist = abs(math.atan2(focal_past[7, 1] - focal_past[6, 1], focal_past[7, 0] - focal_past[6, 0]) -
                          math.atan2(focal_past[6, 1] - focal_past[5, 1], focal_past[6, 0] - focal_past[5, 0]))
    if focal_turn_hist > np.pi:
        focal_turn_hist = 2 * np.pi - focal_turn_hist

    if valid_nbrs > 0:
        nbrs_past = past[1:nv]  # (valid_nbrs, 8, 2)
        dists_t0 = np.linalg.norm(nbrs_past[:, 7] - focal_past[7], axis=-1)  # (valid_nbrs,)
        nearest_idx = np.argmin(dists_t0)
        nearest_obs_dist = dists_t0[nearest_idx]
        nbr_vel = (nbrs_past[nearest_idx, 7] - nbrs_past[nearest_idx, 6]) / 0.4
        focal_vel = (focal_past[7] - focal_past[6]) / 0.4
        rel_speed = float(np.linalg.norm(focal_vel - nbr_vel))

        # CV-TTC and CV-DCA
        rel_pos = nbrs_past[nearest_idx, 7] - focal_past[7]
        rel_v = nbr_vel - focal_vel
        v_sq = np.dot(rel_v, rel_v)
        if v_sq > 1e-6:
            ttc = -np.dot(rel_pos, rel_v) / v_sq
            ttc_clamped = float(np.clip(ttc, 0.0, 10.0))
            closest_pos = rel_pos + rel_v * ttc_clamped
            cv_dca = float(np.linalg.norm(closest_pos))
        else:
            ttc_clamped = 10.0
            cv_dca = float(nearest_obs_dist)
    else:
        nearest_obs_dist = 10.0
        rel_speed = 0.0
        ttc_clamped = 10.0
        cv_dca = 10.0

    g3_scalar = np.array([
        focal_speed_hist, focal_acc_hist, focal_turn_hist, float(valid_nbrs),
        nearest_obs_dist, rel_speed, ttc_clamped, cv_dca
    ], dtype=np.float32)
    g3 = np.tile(g3_scalar[None, :], (20, 1))  # (20, 8)

    # ------------------------------------------------------------------
    # G4: Model-Predicted Neighbor Future Risk (20, 7)
    # ------------------------------------------------------------------
    nbr_cands = sample["neighbor_candidates"].numpy() if isinstance(sample["neighbor_candidates"], torch.Tensor) else sample["neighbor_candidates"]
    nbr_logits = sample["neighbor_logits"].numpy() if isinstance(sample["neighbor_logits"], torch.Tensor) else sample["neighbor_logits"]

    if valid_nbrs > 0 and len(nbr_cands) > 0 and len(nbr_logits) > 0:
        # MPN top1 trajectories for each neighbor: (valid_nbrs, 12, 2)
        nbr_top1_idx = np.argmax(nbr_logits, axis=-1)  # (valid_nbrs,)
        mpn_top1 = nbr_cands[np.arange(valid_nbrs), nbr_top1_idx]  # (valid_nbrs, 12, 2)

        # Distances: (20, valid_nbrs, 12)
        dist_cands_mpn = np.linalg.norm(focal_cands[:, None, :, :] - mpn_top1[None, :, :, :], axis=-1)
        min_clearance_per_nbr = np.min(dist_cands_mpn, axis=-1)  # (20, valid_nbrs)
        min_clearance = np.min(min_clearance_per_nbr, axis=-1)   # (20,)
        mean_clearance = np.mean(dist_cands_mpn, axis=(1, 2))    # (20,)
        pred_coll_prob = (min_clearance < COLLISION_THRESHOLD).astype(np.float32)

        # Time of minimum separation
        t_min_sep = np.argmin(np.min(dist_cands_mpn, axis=1), axis=-1).astype(np.float32)

        # CVaR (mean of lowest 10% clearances across timesteps and neighbors)
        sorted_dists = np.sort(dist_cands_mpn.reshape(20, -1), axis=-1)
        cvar_k = max(1, int(sorted_dists.shape[-1] * 0.1))
        cvar_clearance = np.mean(sorted_dists[:, :cvar_k], axis=-1)

        worst_nbr_risk = 1.0 / (min_clearance + 0.05)

        # Expected clearance over neighbor modes
        exp_nbr_logits = np.exp(nbr_logits - np.max(nbr_logits, axis=-1, keepdims=True))
        nbr_probs = exp_nbr_logits / np.sum(exp_nbr_logits, axis=-1, keepdims=True)  # (valid_nbrs, 20)
        # For each candidate and neighbor: dist is (20, valid_nbrs, 20_modes, 12)
        all_modes_dist = np.min(
            np.linalg.norm(focal_cands[:, None, None, :, :] - nbr_cands[None, :, :, :, :], axis=-1),
            axis=-1
        )  # (20, valid_nbrs, 20_modes)
        expected_clearance = np.mean(np.sum(all_modes_dist * nbr_probs[None, :, :], axis=-1), axis=-1)
    else:
        min_clearance = np.full(20, 10.0, dtype=np.float32)
        mean_clearance = np.full(20, 10.0, dtype=np.float32)
        pred_coll_prob = np.zeros(20, dtype=np.float32)
        t_min_sep = np.zeros(20, dtype=np.float32)
        cvar_clearance = np.full(20, 10.0, dtype=np.float32)
        worst_nbr_risk = np.full(20, 0.1, dtype=np.float32)
        expected_clearance = np.full(20, 10.0, dtype=np.float32)

    g4 = np.stack([
        min_clearance, mean_clearance, pred_coll_prob,
        t_min_sep, cvar_clearance, worst_nbr_risk, expected_clearance
    ], axis=-1)  # (20, 7)

    # ------------------------------------------------------------------
    # Privileged Labels (Only for Training / Validation Supervision)
    # ------------------------------------------------------------------
    cand_ade = sample["candidate_ade"].numpy() if isinstance(sample["candidate_ade"], torch.Tensor) else sample["candidate_ade"]
    cand_fde = sample["candidate_fde"].numpy() if isinstance(sample["candidate_fde"], torch.Tensor) else sample["candidate_fde"]
    cand_clearance = sample["candidate_gt_min_clearance"].numpy() if isinstance(sample["candidate_gt_min_clearance"], torch.Tensor) else sample["candidate_gt_min_clearance"]
    cand_coll = sample["candidate_gt_collision"].numpy() if isinstance(sample["candidate_gt_collision"], torch.Tensor) else sample["candidate_gt_collision"]

    return {
        "g0": g0.astype(np.float32),
        "g1": g1.astype(np.float32),
        "g2": g2.astype(np.float32),
        "g3": g3.astype(np.float32),
        "g4": g4.astype(np.float32),
        "cand_ade": cand_ade.astype(np.float32),
        "cand_fde": cand_fde.astype(np.float32),
        "cand_clearance": cand_clearance.astype(np.float32),
        "cand_collision": cand_coll.astype(bool),
        "has_neighbours": bool(valid_nbrs > 0),
        "native_top1_idx": int(order[0]),
        "min_ade20": float(np.min(cand_ade)),
        "min_fde20": float(np.min(cand_fde)),
    }


def process_fold(cache_dir: Path, out_dir: Path, force: bool = False):
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found in {cache_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_manifest = out_dir / "manifest.json"
    if out_manifest.exists() and not force:
        try:
            m = json.loads(out_manifest.read_text(encoding="utf-8"))
            if m.get("status") == "complete":
                print(f"[FEATURES] Already complete: {out_dir}")
                return
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    shards = manifest["shards"]
    print(f"[FEATURES] Processing {len(shards)} shards from {cache_dir}...")

    all_g0 = []
    all_g1 = []
    all_g2 = []
    all_g3 = []
    all_g4 = []
    all_ade = []
    all_fde = []
    all_clearance = []
    all_collision = []
    all_has_nbr = []
    all_native_top1 = []
    all_min_ade20 = []
    all_min_fde20 = []

    t0 = time.time()
    for shard_file in shards:
        shard_path = cache_dir / shard_file
        samples = torch.load(shard_path, map_location="cpu", weights_only=False)
        for s in samples:
            feat = compute_sample_features(s)
            all_g0.append(feat["g0"])
            all_g1.append(feat["g1"])
            all_g2.append(feat["g2"])
            all_g3.append(feat["g3"])
            all_g4.append(feat["g4"])
            all_ade.append(feat["cand_ade"])
            all_fde.append(feat["cand_fde"])
            all_clearance.append(feat["cand_clearance"])
            all_collision.append(feat["cand_collision"])
            all_has_nbr.append(feat["has_neighbours"])
            all_native_top1.append(feat["native_top1_idx"])
            all_min_ade20.append(feat["min_ade20"])
            all_min_fde20.append(feat["min_fde20"])

    npz_path = out_dir / "features.npz"
    tmp_npz = out_dir / "features.tmp.npz"
    np.savez_compressed(
        tmp_npz,
        g0=np.array(all_g0, dtype=np.float32),
        g1=np.array(all_g1, dtype=np.float32),
        g2=np.array(all_g2, dtype=np.float32),
        g3=np.array(all_g3, dtype=np.float32),
        g4=np.array(all_g4, dtype=np.float32),
        ade=np.array(all_ade, dtype=np.float32),
        fde=np.array(all_fde, dtype=np.float32),
        clearance=np.array(all_clearance, dtype=np.float32),
        collision=np.array(all_collision, dtype=bool),
        has_neighbours=np.array(all_has_nbr, dtype=bool),
        native_top1_idx=np.array(all_native_top1, dtype=np.int32),
        min_ade20=np.array(all_min_ade20, dtype=np.float32),
        min_fde20=np.array(all_min_fde20, dtype=np.float32),
    )
    if npz_path.exists():
        os.remove(npz_path)
    os.replace(tmp_npz, npz_path)

    elapsed = time.time() - t0
    meta = {
        "fold": manifest.get("fold"),
        "seed": manifest.get("seed"),
        "total_samples": len(all_g0),
        "status": "complete",
        "elapsed_seconds": elapsed,
        "feature_shapes": {
            "g0": list(all_g0[0].shape),
            "g1": list(all_g1[0].shape),
            "g2": list(all_g2[0].shape),
            "g3": list(all_g3[0].shape),
            "g4": list(all_g4[0].shape),
        },
        "timestamp": now_iso(),
    }
    tmp_meta = out_dir / "manifest.tmp.json"
    tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if out_manifest.exists():
        os.remove(out_manifest)
    os.replace(tmp_meta, out_manifest)
    print(f"[FEATURES OK] {out_dir}: {len(all_g0)} samples in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Build Feature Bank")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache-root", type=str, default="SELECTOR_BENCHMARK_SERIES/cache")
    parser.add_argument("--feature-root", type=str, default="SELECTOR_BENCHMARK_SERIES/features")
    parser.add_argument("--feature-groups", type=str, default="g0,g1,g2,g3,g4")
    parser.add_argument("--model", type=str, default="f16")
    parser.add_argument("--folds", type=str, default="eth,hotel,univ,zara01,zara02")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cache_base = (repo_root / args.cache_root / args.model / f"seed_{args.seed}").resolve()
    feature_base = (repo_root / args.feature_root / f"seed_{args.seed}").resolve()

    fold_list = [f.strip() for f in args.folds.split(",") if f.strip()]
    for fold in fold_list:
        cache_fold_dir = cache_base / fold
        out_fold_dir = feature_base / fold
        if not cache_fold_dir.exists():
            raise FileNotFoundError(f"Cache folder missing: {cache_fold_dir}")
        process_fold(cache_fold_dir, out_fold_dir, force=args.force)

    print(f"\n[FEATURE BANK OK] All folds processed for seed {args.seed} -> {feature_base}")


if __name__ == "__main__":
    main()
