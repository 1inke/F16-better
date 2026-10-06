"""Same-sample visual comparison of the H3 source and C00 V2 predictions.

All summary panels use the complete locked ETH validation set.  The trajectory
plate shows four deterministically selected examples that represent a clear
oracle gain, the median change, a geometry/score mismatch, and an oracle loss.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTPUT = HERE / "results"
H3_CACHE = (
    REPO / "DIVERSITY_STAGE_ATTRIBUTION_SERIES" / "results"
    / "h3_interim_analysis" / "validation_candidate_cache" / "h3_w005.npz"
)
C00_CACHE = (
    REPO / "DIVERSITY_STAGE_ATTRIBUTION_SERIES" / "results"
    / "stage_attribution_analysis" / "validation_candidate_cache" / "c00.npz"
)
GEOMETRY_CACHE = (
    REPO / "F16_RANKABILITY_PROBE_SERIES" / "results" / "cache" / "val_seed1.npz"
)

os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".cache" / "mpl_config"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "history": "#24476B",
    "truth": "#111827",
    "other": "#A7B3C2",
    "selected": "#D1495B",
    "oracle": "#178A67",
    "combined": "#7A4EAB",
    "h3": "#718096",
    "c00": "#2F6B9A",
    "gain": "#178A67",
    "loss": "#C44E52",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def derive(cache: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    n = len(cache["sample_id"])
    row = np.arange(n)
    ade, fde, logits = cache["ade"], cache["fde"], cache["logits"]
    selected_index = logits.argmax(axis=1)
    oracle_index = ade.argmin(axis=1)
    order = np.argsort(-logits, axis=1)
    rank = np.empty_like(order)
    rank[row[:, None], order] = np.arange(order.shape[1])[None]
    endpoints = cache["candidate"][:, :, -1]
    left, right = np.triu_indices(endpoints.shape[1], k=1)
    spread = np.linalg.norm(endpoints[:, left] - endpoints[:, right], axis=-1).mean(axis=1)
    return {
        "minADE": ade.min(axis=1),
        "minFDE": fde.min(axis=1),
        "selected_ADE": ade[row, selected_index],
        "selected_FDE": fde[row, selected_index],
        "spread": spread,
        "selected_index": selected_index,
        "oracle_index": oracle_index,
        "oracle_rank": rank[row, oracle_index] + 1,
    }


def bootstrap_relative_change(
    before: np.ndarray,
    after: np.ndarray,
    repetitions: int = 10_000,
    seed: int = 20260922,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(0, repetitions, 500):
        count = min(500, repetitions - len(values))
        indices = rng.integers(0, len(before), size=(count, len(before)))
        before_mean = before[indices].mean(axis=1)
        after_mean = after[indices].mean(axis=1)
        values.extend((100.0 * (after_mean - before_mean) / before_mean).tolist())
    values = np.asarray(values)
    estimate = 100.0 * (after.mean() - before.mean()) / before.mean()
    low, high = np.quantile(values, (0.025, 0.975))
    return float(estimate), float(low), float(high)


def representative_indices(
    h3: dict[str, np.ndarray], c00: dict[str, np.ndarray]
) -> list[tuple[str, int, str]]:
    oracle_delta = c00["minADE"] - h3["minADE"]
    selected_delta = c00["selected_ADE"] - h3["selected_ADE"]
    result: list[tuple[str, int, str]] = []
    for quantile, name, note in (
        (0.10, "Clear oracle gain", "10th percentile of paired minADE change"),
        (0.50, "Typical case", "closest sample to the median paired change"),
    ):
        target = np.quantile(oracle_delta, quantile)
        index = int(np.argmin(np.abs(oracle_delta - target)))
        result.append((name, index, note))
    mismatch = (oracle_delta < -0.02) & (selected_delta > 0.20)
    if mismatch.any():
        candidates = np.where(mismatch)[0]
        index = int(candidates[np.argmax(selected_delta[candidates])])
    else:
        index = int(np.argmax(selected_delta - oracle_delta))
    result.append((
        "Geometry-score mismatch", index,
        "oracle improves but the selected trajectory becomes worse",
    ))
    target = np.quantile(oracle_delta, 0.90)
    index = int(np.argmin(np.abs(oracle_delta - target)))
    result.append(("Oracle degradation", index, "90th percentile of paired minADE change"))
    return result


def draw_trajectory(
    ax,
    cache: np.lib.npyio.NpzFile,
    metrics: dict[str, np.ndarray],
    geometry: np.lib.npyio.NpzFile,
    index: int,
    method: str,
    limits: tuple[float, float, float, float],
) -> None:
    candidates = cache["candidate"][index]
    selected = int(metrics["selected_index"][index])
    oracle = int(metrics["oracle_index"][index])
    past = geometry["past_relative"][index]
    future = geometry["future_relative"][index]
    for mode, trajectory in enumerate(candidates):
        if mode not in {selected, oracle}:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=PALETTE["other"], lw=0.75, alpha=0.38)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=8, color=PALETTE["other"], alpha=0.50)
    ax.plot(past[:, 0], past[:, 1], color=PALETTE["history"], lw=2.0, marker="o", ms=2.3, label="Observed")
    ax.plot(future[:, 0], future[:, 1], color=PALETTE["truth"], lw=2.2, ls="--", marker="s", ms=2.2, label="Ground truth")
    if selected == oracle:
        trajectory = candidates[selected]
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=PALETTE["combined"], lw=2.5, label="Selected = oracle")
        ax.scatter(*trajectory[-1], s=40, color=PALETTE["combined"], edgecolor="white", lw=0.8, zorder=5)
    else:
        selected_trajectory = candidates[selected]
        oracle_trajectory = candidates[oracle]
        ax.plot(selected_trajectory[:, 0], selected_trajectory[:, 1], color=PALETTE["selected"], lw=2.4, label="Selected")
        ax.plot(oracle_trajectory[:, 0], oracle_trajectory[:, 1], color=PALETTE["oracle"], lw=2.4, label="minADE oracle")
        ax.scatter(*selected_trajectory[-1], s=38, color=PALETTE["selected"], edgecolor="white", lw=0.8, zorder=5)
        ax.scatter(*oracle_trajectory[-1], s=38, color=PALETTE["oracle"], edgecolor="white", lw=0.8, zorder=5)
    ax.scatter(past[-1, 0], past[-1, 1], marker="+", s=55, color="black", lw=1.4, zorder=6)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", lw=0.5, color="#D5DCE5")
    ax.set_title(
        f"{method} | minADE {metrics['minADE'][index]:.3f} m | "
        f"selected {metrics['selected_ADE'][index]:.3f} m | "
        f"oracle rank {int(metrics['oracle_rank'][index])}/20",
        fontsize=8.4,
    )


def save_all(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    h3_cache, c00_cache = np.load(H3_CACHE), np.load(C00_CACHE)
    geometry = np.load(GEOMETRY_CACHE)
    if not (
        np.array_equal(h3_cache["sample_id"], c00_cache["sample_id"])
        and np.array_equal(c00_cache["sample_id"], geometry["sample_id"])
    ):
        raise RuntimeError("H3, C00, and geometry caches are not sample-aligned")
    if np.max(np.abs(c00_cache["candidate"] - geometry["candidate_final"])) > 5.0e-5:
        raise RuntimeError("C00 candidate cache does not match the visualizer cache")

    h3, c00 = derive(h3_cache), derive(c00_cache)
    metrics = ("minADE", "minFDE", "selected_ADE", "spread")
    paired = {}
    for metric_index, metric in enumerate(metrics):
        estimate, low, high = bootstrap_relative_change(
            h3[metric], c00[metric], seed=20260922 + metric_index
        )
        paired[metric] = {
            "h3_mean": float(h3[metric].mean()),
            "c00_mean": float(c00[metric].mean()),
            "relative_change_pct": estimate,
            "ci95_low_pct": low,
            "ci95_high_pct": high,
        }
    oracle_delta = c00["minADE"] - h3["minADE"]
    selected_delta = c00["selected_ADE"] - h3["selected_ADE"]
    paired["winner_switch_rate"] = float((h3["oracle_index"] != c00["oracle_index"]).mean())
    paired["c00_minade_better_fraction"] = float((oracle_delta < 0).mean())
    paired["c00_selected_better_fraction"] = float((selected_delta < 0).mean())
    paired["h3_oracle_rank_mean"] = float(h3["oracle_rank"].mean())
    paired["c00_oracle_rank_mean"] = float(c00["oracle_rank"].mean())
    paired["h3_oracle_rank_median"] = float(np.median(h3["oracle_rank"]))
    paired["c00_oracle_rank_median"] = float(np.median(c00["oracle_rank"]))

    # Quantitative overview: all 1,516 paired samples are displayed or summarized.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    ax = axes[0, 0]
    improvement = oracle_delta < 0
    ax.scatter(h3["minADE"][~improvement], c00["minADE"][~improvement], s=8, color=PALETTE["loss"], alpha=0.28, rasterized=True)
    ax.scatter(h3["minADE"][improvement], c00["minADE"][improvement], s=8, color=PALETTE["gain"], alpha=0.28, rasterized=True)
    bound = float(max(h3["minADE"].max(), c00["minADE"].max()))
    ax.plot([0, bound], [0, bound], color="#4B5563", ls="--", lw=1.0)
    ax.set(xlabel="H3 minADE (m)", ylabel="C00 minADE (m)", title="a  Paired oracle accuracy")
    ax.text(0.03, 0.96, f"C00 better: {(oracle_delta < 0).mean():.1%}", transform=ax.transAxes, va="top")

    ax = axes[0, 1]
    bins = np.linspace(
        np.quantile(np.concatenate((oracle_delta, selected_delta)), 0.01),
        np.quantile(np.concatenate((oracle_delta, selected_delta)), 0.99), 52,
    )
    ax.hist(oracle_delta, bins=bins, density=True, histtype="step", lw=2.0, color=PALETTE["oracle"], label="minADE")
    ax.hist(selected_delta, bins=bins, density=True, histtype="step", lw=2.0, color=PALETTE["selected"], label="Selected ADE")
    ax.axvline(0, color="#4B5563", ls="--", lw=1.0)
    ax.set(xlabel="Paired change: C00 − H3 (m)", ylabel="Density", title="b  Heterogeneous sample-level effects")
    ax.legend()

    ax = axes[1, 0]
    labels = ["minADE", "minFDE", "Selected ADE", "Endpoint spread"]
    keys = ["minADE", "minFDE", "selected_ADE", "spread"]
    estimates = np.array([paired[key]["relative_change_pct"] for key in keys])
    lows = np.array([paired[key]["ci95_low_pct"] for key in keys])
    highs = np.array([paired[key]["ci95_high_pct"] for key in keys])
    colors = [PALETTE["gain"] if value < 0 else PALETTE["loss"] for value in estimates]
    y = np.arange(len(keys))
    ax.barh(y, estimates, color=colors, alpha=0.86)
    ax.errorbar(estimates, y, xerr=np.vstack((estimates - lows, highs - estimates)), fmt="none", color="#111827", capsize=2.5, lw=0.9)
    ax.axvline(0, color="#4B5563", lw=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlabel="Relative change, C00 vs H3 (%)", title="c  Mean effect with paired-bootstrap 95% CI")
    ax.text(0.02, 0.02, "Negative = lower; desirable for error, contraction for spread", transform=ax.transAxes, fontsize=7, color="#4B5563")

    ax = axes[1, 1]
    bins = np.arange(0.5, 21.5, 1)
    ax.hist(h3["oracle_rank"], bins=bins, density=True, histtype="step", lw=2.0, color=PALETTE["h3"], label="H3")
    ax.hist(c00["oracle_rank"], bins=bins, density=True, histtype="step", lw=2.0, color=PALETTE["c00"], label="C00")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set(xlabel="Score rank of minADE candidate", ylabel="Fraction", title="d  Geometry improves more than score alignment")
    ax.legend()
    fig.suptitle("C00 versus its H3 source on the locked ETH validation set", fontsize=12, fontweight="bold")
    save_all(fig, OUTPUT / "c00_vs_h3_overview")
    plt.close(fig)

    representatives = representative_indices(h3, c00)
    figure, axes = plt.subplots(len(representatives), 2, figsize=(7.2, 10.2), constrained_layout=True)
    manifest_rows = []
    for row_index, (label, index, rule) in enumerate(representatives):
        all_points = np.concatenate((
            h3_cache["candidate"][index].reshape(-1, 2),
            c00_cache["candidate"][index].reshape(-1, 2),
            geometry["past_relative"][index], geometry["future_relative"][index],
        ))
        margin = max(0.25, 0.08 * np.ptp(all_points, axis=0).max())
        limits = (
            float(all_points[:, 0].min() - margin), float(all_points[:, 0].max() + margin),
            float(all_points[:, 1].min() - margin), float(all_points[:, 1].max() + margin),
        )
        draw_trajectory(axes[row_index, 0], h3_cache, h3, geometry, index, "H3 (before C00)", limits)
        draw_trajectory(axes[row_index, 1], c00_cache, c00, geometry, index, "C00", limits)
        axes[row_index, 0].set_ylabel(f"{label}\nRelative y (m)", fontweight="bold")
        for column in range(2):
            axes[row_index, column].set_xlabel("Relative x (m)")
        sample_id = int(h3_cache["sample_id"][index])
        axes[row_index, 0].text(
            0.01, 0.02, f"sample {sample_id} | {rule}", transform=axes[row_index, 0].transAxes,
            fontsize=7, color="#4B5563", va="bottom",
        )
        manifest_rows.append({
            "category": label, "selection_rule": rule, "cache_index": index,
            "sample_id": sample_id,
            "h3_minADE": float(h3["minADE"][index]),
            "c00_minADE": float(c00["minADE"][index]),
            "delta_minADE": float(oracle_delta[index]),
            "h3_selected_ADE": float(h3["selected_ADE"][index]),
            "c00_selected_ADE": float(c00["selected_ADE"][index]),
            "delta_selected_ADE": float(selected_delta[index]),
            "winner_changed": bool(h3["oracle_index"][index] != c00["oracle_index"][index]),
        })
    handles, legend_labels = axes[0, 1].get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    figure.legend(unique.values(), unique.keys(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.015))
    figure.suptitle("How C00 changes the 20-candidate geometry", fontsize=12, fontweight="bold", y=1.035)
    save_all(figure, OUTPUT / "c00_vs_h3_representative_trajectories")
    plt.close(figure)

    with (OUTPUT / "representative_samples.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "scope": "locked ETH validation set",
        "sample_count": int(len(h3_cache["sample_id"])),
        "sample_ids_exactly_aligned": True,
        "c00_visual_cache_max_abs_error": float(
            np.max(np.abs(c00_cache["candidate"] - geometry["candidate_final"]))
        ),
        "paired_summary": paired,
        "representatives": manifest_rows,
        "representative_selection_is_descriptive_not_inferential": True,
        "test_data_used": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
