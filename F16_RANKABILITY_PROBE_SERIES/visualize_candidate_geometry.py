#!/usr/bin/env python3
"""
F16 Candidate Geometry & Selection Diagnostics Visualizer

Implements dual-panel trajectory space and quality-score diagnostic visualization
for the F16 Rankability Probe series, adhering to the diagnostic audit protocol
in quest.md.

Usage:
  python F16_RANKABILITY_PROBE_SERIES/visualize_candidate_geometry.py [--smoke] [--show-anchors] [--workers 8]
"""

import os
import sys
import argparse
import json
import concurrent.futures
import numpy as np

# Ensure matplotlib does not attempt to write font caches to protected user home in sandbox
current_script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_script_dir)
cache_mpl_dir = os.path.join(workspace_root, ".cache", "mpl_config")
os.makedirs(cache_mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_mpl_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="F16 Candidate Geometry and Selection Diagnostics Visualizer")
    parser.add_argument(
        "--cache-path",
        type=str,
        default="F16_RANKABILITY_PROBE_SERIES/results/cache/val_seed1.npz",
        help="Path to rankability feature cache NPZ file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="F16_RANKABILITY_PROBE_SERIES/results/visualizations",
        help="Destination directory for visualization panels"
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=20,
        help="Maximum images to render per non-top group (good_selection, regret_gt_20cm, regret_gt_50cm)"
    )
    parser.add_argument(
        "--top-k-regret",
        type=int,
        default=100,
        help="Number of images to render in top100_regret group"
    )
    parser.add_argument(
        "--sample-ids",
        type=str,
        default="",
        help="Comma-separated sample IDs to render specifically (ignores group quotas)"
    )
    parser.add_argument(
        "--show-anchors",
        action="store_true",
        help="Draw subtle arrows from source anchors to corrected anchors"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast smoke test mode: 2 samples per group, top 5 samples"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing valid image files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for rendering (1 for sandbox compatibility)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI resolution"
    )
    return parser.parse_args()


def compute_sample_diagnostics(cache_data):
    """
    Dynamically compute vectorized metrics across all samples without hardcoding.
    """
    sample_ids = cache_data['sample_id']
    N = len(sample_ids)
    K = cache_data['candidate_final'].shape[1]  # 20

    native_selected_idx = cache_data['native_selected_index']
    oracle_ade_idx = cache_data['oracle_final_index']  # minADE
    ADE_final = cache_data['ADE_final']                # [N, 20]
    FDE_final = cache_data['FDE_final']                # [N, 20]
    native_logits = cache_data['native_logits']        # [N, 20]
    candidate_final = cache_data['candidate_final']    # [N, 20, 12, 2]

    # minFDE index per sample
    min_fde_idx = np.argmin(FDE_final, axis=1)

    # Selected and oracle ADEs
    row_indices = np.arange(N)
    ade_selected = ADE_final[row_indices, native_selected_idx]
    fde_selected = FDE_final[row_indices, native_selected_idx]

    ade_min = ADE_final[row_indices, oracle_ade_idx]
    fde_at_min_ade = FDE_final[row_indices, oracle_ade_idx]

    fde_min = FDE_final[row_indices, min_fde_idx]
    ade_at_min_fde = ADE_final[row_indices, min_fde_idx]

    # Selection Regret
    regret = ade_selected - ade_min

    # Score ranks (1-indexed, rank 1 is highest logit)
    sorted_modes = np.argsort(-native_logits, axis=1)
    ranks = np.empty_like(sorted_modes)
    for i in range(N):
        ranks[i, sorted_modes[i]] = np.arange(1, K + 1)

    oracle_score_ranks = ranks[row_indices, oracle_ade_idx]
    selected_score_ranks = ranks[row_indices, native_selected_idx]
    min_fde_score_ranks = ranks[row_indices, min_fde_idx]

    # Normalized score (softmax over logits)
    exp_logits = np.exp(native_logits - np.max(native_logits, axis=1, keepdims=True))
    normalized_scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Endpoint spread: average pairwise euclidean distance among 20 candidate endpoints
    endpoints = candidate_final[:, :, -1, :]
    diff = endpoints[:, :, None, :] - endpoints[:, None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # [N, 20, 20]
    spread = np.sum(dist, axis=(1, 2)) / (K * (K - 1))

    # medianFDE - minFDE
    median_fde = np.median(FDE_final, axis=1)
    median_minus_min_fde = median_fde - fde_min

    return {
        "N": N,
        "K": K,
        "sample_ids": sample_ids,
        "native_selected_idx": native_selected_idx,
        "oracle_ade_idx": oracle_ade_idx,
        "min_fde_idx": min_fde_idx,
        "ade_selected": ade_selected,
        "fde_selected": fde_selected,
        "ade_min": ade_min,
        "fde_at_min_ade": fde_at_min_ade,
        "fde_min": fde_min,
        "ade_at_min_fde": ade_at_min_fde,
        "regret": regret,
        "oracle_score_ranks": oracle_score_ranks,
        "selected_score_ranks": selected_score_ranks,
        "min_fde_score_ranks": min_fde_score_ranks,
        "normalized_scores": normalized_scores,
        "endpoint_spread": spread,
        "median_minus_min_fde": median_minus_min_fde,
    }


def build_sample_payload(sample_idx, cache_data, diagnostics, show_anchors=False):
    """
    Builds an isolated, pickleable payload for a single sample render task.
    """
    sample_id = int(diagnostics["sample_ids"][sample_idx])
    past = cache_data["past_relative"][sample_idx]
    future = cache_data["future_relative"][sample_idx]
    candidates = cache_data["candidate_final"][sample_idx]
    ADE_modes = cache_data["ADE_final"][sample_idx]
    FDE_modes = cache_data["FDE_final"][sample_idx]
    norm_scores = diagnostics["normalized_scores"][sample_idx]

    src_anc = None
    cor_anc = None
    if show_anchors and "source_anchors_relative" in cache_data and "corrected_anchors_relative" in cache_data:
        src_anc = cache_data["source_anchors_relative"][sample_idx]
        cor_anc = cache_data["corrected_anchors_relative"][sample_idx]

    return {
        "sample_id": sample_id,
        "sample_idx": int(sample_idx),
        "past": past,
        "future": future,
        "candidates": candidates,
        "ADE_modes": ADE_modes,
        "FDE_modes": FDE_modes,
        "norm_scores": norm_scores,
        "source_anchors": src_anc,
        "corrected_anchors": cor_anc,
        "sel_idx": int(diagnostics["native_selected_idx"][sample_idx]),
        "ade_idx": int(diagnostics["oracle_ade_idx"][sample_idx]),
        "fde_idx": int(diagnostics["min_fde_idx"][sample_idx]),
        "regret_val": float(diagnostics["regret"][sample_idx]),
        "sel_rank": int(diagnostics["selected_score_ranks"][sample_idx]),
        "ade_rank": int(diagnostics["oracle_score_ranks"][sample_idx]),
        "fde_rank": int(diagnostics["min_fde_score_ranks"][sample_idx]),
        "spread_val": float(diagnostics["endpoint_spread"][sample_idx]),
        "min_fde_val": float(diagnostics["fde_min"][sample_idx]),
        "diff_med_min_fde": float(diagnostics["median_minus_min_fde"][sample_idx]),
        "ade_selected": float(diagnostics["ade_selected"][sample_idx]),
        "fde_selected": float(diagnostics["fde_selected"][sample_idx]),
        "ade_min": float(diagnostics["ade_min"][sample_idx]),
        "fde_at_min_ade": float(diagnostics["fde_at_min_ade"][sample_idx]),
        "ade_at_min_fde": float(diagnostics["ade_at_min_fde"][sample_idx]),
    }


def render_sample_payload(payload, output_filepath, dpi=150):
    """
    Renders one dual-panel diagnostic visualization from a single sample payload.
    Left: Trajectory space in relative physical coordinates.
    Right: ADE-FDE-Score quality scatter.
    """
    sample_id = payload["sample_id"]
    sample_idx = payload["sample_idx"]
    past = payload["past"]
    future = payload["future"]
    candidates = payload["candidates"]
    ADE_modes = payload["ADE_modes"]
    FDE_modes = payload["FDE_modes"]
    norm_scores = payload["norm_scores"]

    sel_idx = payload["sel_idx"]
    ade_idx = payload["ade_idx"]
    fde_idx = payload["fde_idx"]

    regret_val = payload["regret_val"]
    sel_rank = payload["sel_rank"]
    ade_rank = payload["ade_rank"]
    fde_rank = payload["fde_rank"]

    spread_val = payload["spread_val"]
    min_fde_val = payload["min_fde_val"]
    diff_med_min_fde = payload["diff_med_min_fde"]

    is_minade_minfde = (ade_idx == fde_idx)
    is_sel_minade = (sel_idx == ade_idx)
    is_sel_minfde = (sel_idx == fde_idx)

    fig, (ax_traj, ax_diag) = plt.subplots(1, 2, figsize=(16, 7.8), dpi=dpi)

    # -------------------------------------------------------------
    # Left Panel: Trajectory Geometry
    # -------------------------------------------------------------
    # 1. Reference origin (t=0)
    origin = past[-1] if len(past) > 0 else np.array([0.0, 0.0])
    ax_traj.plot(origin[0], origin[1], marker='+', markersize=10, markeredgewidth=2,
                 color='black', zorder=10, label='Current Pos (t=0)')

    # 2. Source anchors -> Corrected anchors (optional)
    if payload.get("source_anchors") is not None and payload.get("corrected_anchors") is not None:
        src_anc = payload["source_anchors"]
        cor_anc = payload["corrected_anchors"]
        for k in range(len(src_anc)):
            ax_traj.annotate(
                '',
                xy=(cor_anc[k, -1, 0], cor_anc[k, -1, 1]),
                xytext=(src_anc[k, -1, 0], src_anc[k, -1, 1]),
                arrowprops=dict(arrowstyle="->", color='#64748B', lw=0.7, alpha=0.35, shrinkA=0, shrinkB=0),
                zorder=2
            )

    # 3. Non-highlighted candidates (17~19 trajectories)
    special_indices = {sel_idx, ade_idx, fde_idx}
    other_label_added = False
    for k in range(len(candidates)):
        if k not in special_indices:
            lbl = "Other Candidates (17~19)" if not other_label_added else None
            ax_traj.plot(
                candidates[k, :, 0], candidates[k, :, 1],
                color='#94A3B8', alpha=0.45, lw=1.2, zorder=3, label=lbl
            )
            ax_traj.scatter(
                candidates[k, -1, 0], candidates[k, -1, 1],
                color='#64748B', alpha=0.55, s=28, zorder=3, edgecolors='none'
            )
            other_label_added = True

    # 4. Past History (black solid line)
    ax_traj.plot(
        past[:, 0], past[:, 1],
        color='#0F172A', lw=2.4, marker='o', markersize=3.5, zorder=5, label='Past History (8 steps)'
    )

    # 5. Ground Truth (black bold dashed line)
    ax_traj.plot(
        future[:, 0], future[:, 1],
        color='#000000', lw=2.8, linestyle='--', marker='s', markersize=3.5, zorder=6, label='Ground Truth (GT 12 steps)'
    )
    ax_traj.scatter(
        future[-1, 0], future[-1, 1],
        color='#000000', marker='s', s=55, zorder=7, edgecolors='white', linewidths=1.2
    )

    # 6. Highlighted Candidates with Coincidence Disambiguation (quest.md §5)
    if is_sel_minade and is_minade_minfde:
        lbl = f"Selected + minADE + minFDE (m{sel_idx})"
        ax_traj.plot(candidates[sel_idx, :, 0], candidates[sel_idx, :, 1],
                     color='#8A2BE2', lw=3.2, zorder=8, label=lbl)
        ax_traj.scatter(candidates[sel_idx, -1, 0], candidates[sel_idx, -1, 1],
                        color='#8A2BE2', s=85, zorder=9, edgecolors='black', linewidths=1.5)
    elif is_minade_minfde:
        lbl_ade_fde = f"minADE + minFDE (m{ade_idx})"
        ax_traj.plot(candidates[ade_idx, :, 0], candidates[ade_idx, :, 1],
                     color='#0284C7', lw=3.0, zorder=8, label=lbl_ade_fde)
        ax_traj.scatter(candidates[ade_idx, -1, 0], candidates[ade_idx, -1, 1],
                        color='#0284C7', s=80, zorder=9, edgecolors='black', linewidths=1.2)

        lbl_sel = f"Selected / Top-1 (m{sel_idx})"
        ax_traj.plot(candidates[sel_idx, :, 0], candidates[sel_idx, :, 1],
                     color='#DC2626', lw=3.0, zorder=8, label=lbl_sel)
        ax_traj.scatter(candidates[sel_idx, -1, 0], candidates[sel_idx, -1, 1],
                        color='#DC2626', s=80, zorder=9, edgecolors='black', linewidths=1.2)
    elif is_sel_minade:
        lbl_sel_ade = f"Selected + minADE (m{sel_idx})"
        ax_traj.plot(candidates[sel_idx, :, 0], candidates[sel_idx, :, 1],
                     color='#D97706', lw=3.0, zorder=8, label=lbl_sel_ade)
        ax_traj.scatter(candidates[sel_idx, -1, 0], candidates[sel_idx, -1, 1],
                        color='#D97706', s=80, zorder=9, edgecolors='black', linewidths=1.2)

        lbl_fde = f"minFDE (m{fde_idx})"
        ax_traj.plot(candidates[fde_idx, :, 0], candidates[fde_idx, :, 1],
                     color='#2563EB', lw=2.6, zorder=8, label=lbl_fde)
        ax_traj.scatter(candidates[fde_idx, -1, 0], candidates[fde_idx, -1, 1],
                        color='#2563EB', s=75, zorder=9, edgecolors='black', linewidths=1.2)
    elif is_sel_minfde:
        lbl_sel_fde = f"Selected + minFDE (m{sel_idx})"
        ax_traj.plot(candidates[sel_idx, :, 0], candidates[sel_idx, :, 1],
                     color='#C026D3', lw=3.0, zorder=8, label=lbl_sel_fde)
        ax_traj.scatter(candidates[sel_idx, -1, 0], candidates[sel_idx, -1, 1],
                        color='#C026D3', s=80, zorder=9, edgecolors='black', linewidths=1.2)

        lbl_ade = f"minADE (m{ade_idx})"
        ax_traj.plot(candidates[ade_idx, :, 0], candidates[ade_idx, :, 1],
                     color='#16A34A', lw=2.6, zorder=8, label=lbl_ade)
        ax_traj.scatter(candidates[ade_idx, -1, 0], candidates[ade_idx, -1, 1],
                        color='#16A34A', s=75, zorder=9, edgecolors='black', linewidths=1.2)
    else:
        lbl_ade = f"minADE (m{ade_idx})"
        ax_traj.plot(candidates[ade_idx, :, 0], candidates[ade_idx, :, 1],
                     color='#16A34A', lw=2.6, zorder=8, label=lbl_ade)
        ax_traj.scatter(candidates[ade_idx, -1, 0], candidates[ade_idx, -1, 1],
                        color='#16A34A', s=75, zorder=9, edgecolors='black', linewidths=1.2)

        lbl_fde = f"minFDE (m{fde_idx})"
        ax_traj.plot(candidates[fde_idx, :, 0], candidates[fde_idx, :, 1],
                     color='#2563EB', lw=2.6, zorder=8, label=lbl_fde)
        ax_traj.scatter(candidates[fde_idx, -1, 0], candidates[fde_idx, -1, 1],
                        color='#2563EB', s=75, zorder=9, edgecolors='black', linewidths=1.2)

        lbl_sel = f"Selected / Top-1 (m{sel_idx})"
        ax_traj.plot(candidates[sel_idx, :, 0], candidates[sel_idx, :, 1],
                     color='#DC2626', lw=3.0, zorder=8, label=lbl_sel)
        ax_traj.scatter(candidates[sel_idx, -1, 0], candidates[sel_idx, -1, 1],
                        color='#DC2626', s=85, zorder=9, edgecolors='black', linewidths=1.5)

    ax_traj.set_aspect('equal', adjustable='datalim')
    ax_traj.grid(True, linestyle=':', alpha=0.6)
    ax_traj.set_xlabel('Relative X (m)', fontsize=11, fontweight='semibold')
    ax_traj.set_ylabel('Relative Y (m)', fontsize=11, fontweight='semibold')
    ax_traj.set_title('Trajectory Geometry in Relative Coordinates', fontsize=12, fontweight='bold', pad=8)
    ax_traj.legend(loc='best', fontsize=8.5, framealpha=0.92, facecolor='white', edgecolor='#CBD5E1')

    # -------------------------------------------------------------
    # Right Panel: ADE-FDE-Score Scatter Diagnostics
    # -------------------------------------------------------------
    scatter_sizes = 35.0 + 345.0 * norm_scores

    for k in range(len(candidates)):
        if k not in special_indices:
            ax_diag.scatter(
                FDE_modes[k], ADE_modes[k],
                s=scatter_sizes[k], color='#94A3B8', alpha=0.55, edgecolors='#475569', linewidths=0.7, zorder=3
            )
            ax_diag.annotate(
                f"m{k}",
                (FDE_modes[k], ADE_modes[k]),
                textcoords="offset points", xytext=(4, 4),
                fontsize=7.5, color='#64748B', alpha=0.85
            )

    def plot_diag_highlight(idx, color, label_prefix, zorder=6):
        ax_diag.scatter(
            FDE_modes[idx], ADE_modes[idx],
            s=max(scatter_sizes[idx], 120.0),
            color=color, edgecolors='black', linewidths=1.5, zorder=zorder,
            label=f"{label_prefix} (m{idx}, ADE={ADE_modes[idx]:.2f}, FDE={FDE_modes[idx]:.2f})"
        )
        rnk = payload["sel_rank"] if idx == sel_idx else (ade_rank if idx == ade_idx else fde_rank)
        ax_diag.annotate(
            f"{label_prefix}\nm{idx} (r{rnk})",
            (FDE_modes[idx], ADE_modes[idx]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=8.5, fontweight='bold', color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
            zorder=zorder + 1
        )

    if is_sel_minade and is_minade_minfde:
        plot_diag_highlight(sel_idx, '#8A2BE2', 'Selected+minADE+minFDE')
    elif is_minade_minfde:
        plot_diag_highlight(ade_idx, '#0284C7', 'minADE+minFDE')
        plot_diag_highlight(sel_idx, '#DC2626', 'Selected')
    elif is_sel_minade:
        plot_diag_highlight(sel_idx, '#D97706', 'Selected+minADE')
        plot_diag_highlight(fde_idx, '#2563EB', 'minFDE')
    elif is_sel_minfde:
        plot_diag_highlight(sel_idx, '#C026D3', 'Selected+minFDE')
        plot_diag_highlight(ade_idx, '#16A34A', 'minADE')
    else:
        plot_diag_highlight(ade_idx, '#16A34A', 'minADE')
        plot_diag_highlight(fde_idx, '#2563EB', 'minFDE')
        plot_diag_highlight(sel_idx, '#DC2626', 'Selected')

    ax_diag.axvline(x=min_fde_val, color='#2563EB', linestyle=':', alpha=0.45, lw=1.2)
    ax_diag.axhline(y=payload["ade_min"], color='#16A34A', linestyle=':', alpha=0.45, lw=1.2)

    ax_diag.grid(True, linestyle=':', alpha=0.6)
    ax_diag.set_xlabel('Final Displacement Error - FDE (m)', fontsize=11, fontweight='semibold')
    ax_diag.set_ylabel('Average Displacement Error - ADE (m)', fontsize=11, fontweight='semibold')
    ax_diag.set_title('Quality vs. Normalized Score Distribution', fontsize=12, fontweight='bold', pad=8)
    ax_diag.legend(loc='best', fontsize=8.5, framealpha=0.92, facecolor='white', edgecolor='#CBD5E1')

    # -------------------------------------------------------------
    # Super Title / Diagnostic Banner Header (quest.md §3)
    # -------------------------------------------------------------
    header_line1 = (
        f"Sample {sample_id} (Index {sample_idx})  |  "
        f"Selected: mode={sel_idx} | ADE={payload['ade_selected']:.2f}m | FDE={payload['fde_selected']:.2f}m | ScoreRank={sel_rank}/20  |  "
        f"minADE: mode={ade_idx} | ADE={payload['ade_min']:.2f}m | FDE={payload['fde_at_min_ade']:.2f}m | ScoreRank={ade_rank}/20"
    )
    header_line2 = (
        f"minFDE: mode={fde_idx} | ADE={payload['ade_at_min_fde']:.2f}m | FDE={min_fde_val:.2f}m | ScoreRank={fde_rank}/20  |  "
        f"Selection Regret = {regret_val:.2f} m  |  minADE == minFDE: {is_minade_minfde}"
    )
    header_line3 = (
        f"Endpoint Spread = {spread_val:.2f} m  |  minFDE = {min_fde_val:.2f} m  |  "
        f"medianFDE - minFDE = {diff_med_min_fde:.2f} m  |  (Note: point size ~ normalized score = softmax(logit))"
    )

    full_header = f"{header_line1}\n{header_line2}\n{header_line3}"
    fig.suptitle(full_header, fontsize=10.5, family='monospace', fontweight='medium', y=0.985,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#F8FAFC", ec="#94A3B8", lw=1.0))

    plt.tight_layout(rect=[0, 0.02, 1, 0.90])
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def draw_dual_panel(sample_idx, cache_data, diagnostics, output_filepath, show_anchors=False, dpi=150):
    """
    Direct rendering entry point compatible with unit tests and single-item execution.
    """
    payload = build_sample_payload(sample_idx, cache_data, diagnostics, show_anchors=show_anchors)
    render_sample_payload(payload, output_filepath, dpi=dpi)


def _worker_task(task_args):
    payload, output_filepath, dpi = task_args
    render_sample_payload(payload, output_filepath, dpi=dpi)
    return output_filepath


def main():
    args = parse_args()
    print("================================================================================")
    print(" F16 CANDIDATE GEOMETRY & SELECTION DIAGNOSTICS VISUALIZER")
    print("================================================================================")
    print(f"Loading cache: {args.cache_path}")
    if not os.path.exists(args.cache_path):
        raise FileNotFoundError(f"Cache file not found at: {args.cache_path}")

    cache = np.load(args.cache_path)
    diagnostics = compute_sample_diagnostics(cache)
    N = diagnostics["N"]
    print(f"Total samples in cache: {N}")

    # Determine sample subsets for the 4 target directories (quest.md §4)
    # 1. good_selection: regret <= 0.05m
    good_mask = diagnostics["regret"] <= 0.05
    good_indices = np.where(good_mask)[0]
    good_sorted = good_indices[np.argsort(diagnostics["regret"][good_indices])]

    # 2. regret_gt_20cm: regret > 0.20m
    gt20_mask = diagnostics["regret"] > 0.20
    gt20_indices = np.where(gt20_mask)[0]
    gt20_sorted = gt20_indices[np.argsort(-diagnostics["regret"][gt20_indices])]

    # 3. regret_gt_50cm: regret > 0.50m
    gt50_mask = diagnostics["regret"] > 0.50
    gt50_indices = np.where(gt50_mask)[0]
    gt50_sorted = gt50_indices[np.argsort(-diagnostics["regret"][gt50_indices])]

    # 4. top100_regret: top N by regret descending
    all_sorted_by_regret = np.argsort(-diagnostics["regret"])

    if args.smoke:
        print("[SMOKE MODE ENABLED] Capping outputs for fast verification.")
        max_good = min(2, len(good_sorted))
        max_gt20 = min(2, len(gt20_sorted))
        max_gt50 = min(2, len(gt50_sorted))
        max_top = min(5, len(all_sorted_by_regret))
    else:
        max_good = min(args.max_per_group, len(good_sorted))
        max_gt20 = min(args.max_per_group, len(gt20_sorted))
        max_gt50 = min(args.max_per_group, len(gt50_sorted))
        max_top = min(args.top_k_regret, len(all_sorted_by_regret))

    group_selections = {
        "good_selection": good_sorted[:max_good],
        "regret_gt_20cm": gt20_sorted[:max_gt20],
        "regret_gt_50cm": gt50_sorted[:max_gt50],
        "top100_regret": all_sorted_by_regret[:max_top],
    }

    if args.sample_ids.strip():
        req_ids = [int(s.strip()) for s in args.sample_ids.split(",") if s.strip()]
        id_to_idx = {int(sid): idx for idx, sid in enumerate(diagnostics["sample_ids"])}
        matched_indices = [id_to_idx[sid] for sid in req_ids if sid in id_to_idx]
        group_selections = {"custom_samples": np.array(matched_indices, dtype=int)}

    print("\nTarget Sample Counts:")
    for grp, indices in group_selections.items():
        print(f"  - {grp:18s}: {len(indices)} samples to render")

    manifest = {
        "source_cache": os.path.abspath(args.cache_path),
        "total_cache_samples": N,
        "show_anchors": args.show_anchors,
        "smoke": args.smoke,
        "groups": {}
    }

    tasks = []
    for grp_name, sample_indices in group_selections.items():
        grp_dir = os.path.join(args.output_dir, grp_name)
        os.makedirs(grp_dir, exist_ok=True)
        manifest["groups"][grp_name] = []

        for rank_in_grp, idx in enumerate(sample_indices):
            sid = int(diagnostics["sample_ids"][idx])
            reg = float(diagnostics["regret"][idx])
            fname = f"sample_{sid:05d}_regret_{reg:.3f}m.png"
            fpath = os.path.join(grp_dir, fname)

            manifest["groups"][grp_name].append({
                "sample_id": sid,
                "cache_index": int(idx),
                "regret_m": reg,
                "selected_ade_m": float(diagnostics["ade_selected"][idx]),
                "selected_fde_m": float(diagnostics["fde_selected"][idx]),
                "oracle_min_ade_m": float(diagnostics["ade_min"][idx]),
                "oracle_min_fde_m": float(diagnostics["fde_min"][idx]),
                "endpoint_spread_m": float(diagnostics["endpoint_spread"][idx]),
                "filename": fname,
                "relative_path": os.path.relpath(fpath, args.output_dir)
            })

            # Check if file already exists and is valid
            if not args.overwrite and os.path.exists(fpath) and os.path.getsize(fpath) > 20000:
                continue

            payload = build_sample_payload(idx, cache, diagnostics, show_anchors=args.show_anchors)
            tasks.append((payload, fpath, args.dpi))

    print(f"\nPending images to render (after existing-file check): {len(tasks)}")

    if len(tasks) > 0:
        num_workers = max(1, min(args.workers, len(tasks)))
        print(f"Dispatching {len(tasks)} rendering jobs with {num_workers} parallel workers...")
        if num_workers == 1:
            for idx_task, task in enumerate(tasks, 1):
                _worker_task(task)
                if idx_task % 10 == 0 or idx_task == len(tasks):
                    print(f"  Render progress: {idx_task}/{len(tasks)} completed ({idx_task*100/len(tasks):.1f}%)")
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Use as_completed to display progress
                futures = [executor.submit(_worker_task, t) for t in tasks]
                completed = 0
                for f in concurrent.futures.as_completed(futures):
                    f.result()
                    completed += 1
                    if completed % 25 == 0 or completed == len(tasks):
                        print(f"  Render progress: {completed}/{len(tasks)} completed ({completed*100/len(tasks):.1f}%)")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "visualizations_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved overall manifest to: {manifest_path}")
    print("Diagnostic visualization pipeline completed successfully.")


if __name__ == "__main__":
    main()
