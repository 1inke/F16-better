"""Zero-training marginal-utility audit for the locked C00 validation cache."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
C00_CACHE = (
    REPO / "DIVERSITY_STAGE_ATTRIBUTION_SERIES" / "results"
    / "stage_attribution_analysis" / "validation_candidate_cache" / "c00.npz"
)
GEOMETRY_CACHE = REPO / "F16_RANKABILITY_PROBE_SERIES" / "results" / "cache" / "val_seed1.npz"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def marginal_contribution(quality: np.ndarray) -> np.ndarray:
    """Per-sample loss increase after removing each candidate."""
    order = np.argsort(quality, axis=1)
    best, second = quality[np.arange(len(quality)), order[:, 0]], quality[np.arange(len(quality)), order[:, 1]]
    out = np.zeros_like(quality, dtype=np.float64)
    out[np.arange(len(quality)), order[:, 0]] = second - best
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "mode_utility_audit")
    args = parser.parse_args()
    for path in (C00_CACHE, GEOMETRY_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw, geometry = np.load(C00_CACHE), np.load(GEOMETRY_CACHE)
    if not np.array_equal(raw["sample_id"], geometry["sample_id"]):
        raise RuntimeError("C00 and geometry cache sample IDs are not aligned")
    ade = raw["ade"].astype(np.float64)
    fde = raw["fde"].astype(np.float64)
    quality = ade + 0.5 * fde
    logits = raw["logits"].astype(np.float64)
    candidate = raw["candidate"].astype(np.float64)
    n, modes = ade.shape
    row = np.arange(n)
    contribution = marginal_contribution(quality)
    score_order = np.argsort(-logits, axis=1)
    score_rank = np.empty_like(score_order)
    score_rank[row[:, None], score_order] = np.arange(1, modes + 1)[None]
    past_last = geometry["past_relative"][:, -1].astype(np.float64)
    radius = np.linalg.norm(candidate[:, :, -1] - past_last[:, None], axis=-1)
    correction = np.linalg.norm(
        geometry["corrected_anchors_relative"] - geometry["source_anchors_relative"],
        axis=-1,
    ).mean(axis=-1)

    mode_rows = []
    for mode in range(modes):
        values = contribution[:, mode]
        mode_rows.append({
            "mode": mode,
            "mean_marginal_J": float(values.mean()),
            "nonzero_rate": float((values > 0).mean()),
            "mean_score_rank": float(score_rank[:, mode].mean()),
            "mean_endpoint_radius": float(radius[:, mode].mean()),
            "mean_correction": float(correction[:, mode].mean()),
        })
    rank_rows = []
    for rank in range(1, modes + 1):
        mask = score_rank == rank
        values = contribution[mask]
        rank_rows.append({
            "native_score_rank": rank,
            "mean_marginal_J": float(values.mean()),
            "oracle_winner_rate": float((values > 0).mean()),
            "mean_endpoint_radius": float(radius[mask].mean()),
        })
    radius_edges = np.quantile(radius, np.linspace(0.0, 1.0, 6))
    radius_rows = []
    for index in range(5):
        mask = (radius >= radius_edges[index]) & (
            radius <= radius_edges[index + 1] if index == 4 else radius < radius_edges[index + 1]
        )
        values = contribution[mask]
        radius_rows.append({
            "radius_quintile": index + 1,
            "radius_low": float(radius_edges[index]),
            "radius_high": float(radius_edges[index + 1]),
            "mean_marginal_J": float(values.mean()),
            "oracle_winner_rate": float((values > 0).mean()),
        })

    curve_rows = []
    globally_useful = np.argsort(-np.array([item["mean_marginal_J"] for item in mode_rows]))
    for k in (1, 3, 5, 10, 15, 20):
        native_ids = score_order[:, :k]
        native_quality = np.take_along_axis(quality, native_ids, axis=1).min(axis=1)
        global_quality = quality[:, globally_useful[:k]].min(axis=1)
        native_ade = np.take_along_axis(ade, native_ids, axis=1).min(axis=1)
        curve_rows.append({
            "K": k,
            "native_topK_mean_ADE": float(native_ade.mean()),
            "native_topK_mean_J": float(native_quality.mean()),
            "global_mode_utility_topK_mean_J": float(global_quality.mean()),
            "full_oracle_mean_J": float(quality.min(axis=1).mean()),
        })

    oracle = quality.argmin(axis=1)
    far = radius >= np.quantile(radius, 0.8)
    far_oracle_rate = float(far[row, oracle].mean())
    zero_modes = [int(item["mode"]) for item in mode_rows if item["nonzero_rate"] < 0.005]
    summary = {
        "protocol": "c00_mode_leave_one_out_validation_diagnostic_v1",
        "sample_count": n,
        "test_accessed": False,
        "full_oracle_mean_ADE": float(ade.min(axis=1).mean()),
        "full_oracle_mean_J": float(quality.min(axis=1).mean()),
        "far_radius_oracle_rate_top20pct": far_oracle_rate,
        "modes_with_oracle_rate_below_0p5pct": zero_modes,
        "interpretation_guard": (
            "Marginal utility uses validation futures only as a diagnostic. It must not be used "
            "as an inference-time gate or as a test-tuned allocation rule."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mode_marginal_utility.csv", mode_rows)
    write_csv(args.output_dir / "score_rank_marginal_utility.csv", rank_rows)
    write_csv(args.output_dir / "radius_marginal_utility.csv", radius_rows)
    write_csv(args.output_dir / "candidate_budget_curves.csv", curve_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# C00 mode marginal-utility audit", "",
        f"- Samples: {n}",
        f"- Full-set oracle minADE: {summary['full_oracle_mean_ADE']:.6f}",
        f"- Far-radius (top 20%) oracle rate: {100 * far_oracle_rate:.2f}%",
        f"- Modes with <0.5% oracle contribution: {zero_modes}", "",
        "This is a validation-only allocation diagnostic, not an inference rule. Detailed results are in the CSV files.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

