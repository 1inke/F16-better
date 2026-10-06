#!/usr/bin/env python3
"""Render D1 outputs for the legacy curated samples and stack old/new panels.

The legacy images use the same ETH development sample IDs as the formal
regime-adaptive experiment, but they do not use the same candidate generator.
This script therefore keeps only the matching history/ground truth from the
legacy cache and renders the complete fold-local V2 candidate set and D1
reranked scores from the formal cache.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD_SERIES = ROOT / "F16_RANKABILITY_PROBE_SERIES"
OLD_CACHE = OLD_SERIES / "results" / "cache" / "val_seed1.npz"
OLD_IMAGES = (
    OLD_SERIES / "results" / "visualizations" / "regret_gt_50cm_curated_samples"
)
OLD_MANIFEST = OLD_IMAGES / "curated_manifest.json"
FORMAL_CACHE = HERE / "results" / "cache" / "train_dev" / "eth_dev.npz"
FORMAL_CONFIG = HERE / "formal_config.json"
FORMAL_RUNNER = HERE / "run_formal_regime_selection.py"
DEFAULT_OUTPUT = (
    OLD_SERIES
    / "results"
    / "visualizations"
    / "regime_adaptive_d1_vs_old_curated_samples"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--export-vector", action="store_true",
        help="Also write editable SVG/PDF copies of each new-method panel.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def descending_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks + 0


def curated_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, payload in manifest["categories"].items():
        for sample in payload["samples"]:
            row = dict(sample)
            row["category"] = category
            rows.append(row)
    if len(rows) != 25:
        raise RuntimeError(f"Expected 25 curated image entries, found {len(rows)}")
    return rows


def verify_alignment(old: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> dict[int, int]:
    old_ids = np.asarray(old["sample_id"], dtype=np.int64)
    new_ids = np.asarray(new["sample_id"], dtype=np.int64)
    if not np.array_equal(old_ids, new_ids):
        raise RuntimeError("Legacy and formal ETH-dev sample_id arrays are not exactly aligned")
    candidates = new["candidate"][..., :24].reshape(-1, 20, 12, 2)
    errors = np.linalg.norm(candidates - old["future_relative"][:, None], axis=-1)
    ade_diff = float(np.max(np.abs(errors.mean(-1) - new["ade"])))
    fde_diff = float(np.max(np.abs(errors[..., -1] - new["fde"])))
    if ade_diff > 2e-5 or fde_diff > 2e-5:
        raise RuntimeError(
            f"Coordinate audit failed: ADE max diff={ade_diff:.3g}, FDE max diff={fde_diff:.3g}"
        )
    return {int(sample_id): index for index, sample_id in enumerate(new_ids)}


def draw_new_method(payload: dict[str, Any], output: Path, dpi: int) -> None:
    # Matplotlib is imported only after MPLCONFIGDIR is assigned by main().
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    past = payload["past"]
    future = payload["future"]
    candidates = payload["candidates"]
    ade = payload["ade"]
    fde = payload["fde"]
    probabilities = payload["probabilities"]
    native_idx = payload["native_idx"]
    selected_idx = payload["selected_idx"]
    minade_idx = payload["minade_idx"]
    minfde_idx = payload["minfde_idx"]
    ranks = payload["ranks"]

    fig, (ax_traj, ax_score) = plt.subplots(1, 2, figsize=(16, 7.8), dpi=dpi)

    special = {native_idx, selected_idx, minade_idx, minfde_idx}
    other_label = True
    for mode in range(len(candidates)):
        if mode in special:
            continue
        ax_traj.plot(
            candidates[mode, :, 0], candidates[mode, :, 1],
            color="#94A3B8", alpha=0.45, lw=1.2, zorder=2,
            label="Other candidates" if other_label else None,
        )
        ax_traj.scatter(
            candidates[mode, -1, 0], candidates[mode, -1, 1],
            color="#64748B", alpha=0.55, s=28, edgecolors="none", zorder=2,
        )
        other_label = False

    origin = past[-1]
    ax_traj.plot(
        past[:, 0], past[:, 1], color="#0F172A", lw=2.4, marker="o",
        markersize=3.5, zorder=4, label="Past history (8 steps)",
    )
    ax_traj.plot(
        origin[0], origin[1], marker="+", markersize=10, markeredgewidth=2,
        color="black", zorder=10, label="Current position (t=0)",
    )
    ax_traj.plot(
        future[:, 0], future[:, 1], color="black", lw=2.8, linestyle="--",
        marker="s", markersize=3.5, zorder=5, label="Ground truth (12 steps)",
    )

    if native_idx != selected_idx:
        ax_traj.plot(
            candidates[native_idx, :, 0], candidates[native_idx, :, 1],
            color="#D97706", lw=2.3, linestyle="--", zorder=6,
            label=f"Current native top-1 (m{native_idx})",
        )
        ax_traj.scatter(
            *candidates[native_idx, -1], color="#D97706", marker="D", s=65,
            edgecolors="black", linewidths=1.0, zorder=7,
        )

    if minfde_idx != minade_idx:
        ax_traj.plot(
            candidates[minfde_idx, :, 0], candidates[minfde_idx, :, 1],
            color="#2563EB", lw=2.5, zorder=7, label=f"minFDE (m{minfde_idx})",
        )
        ax_traj.scatter(
            *candidates[minfde_idx, -1], color="#2563EB", s=72,
            edgecolors="black", linewidths=1.1, zorder=8,
        )

    oracle_label = (
        f"minADE + minFDE (m{minade_idx})"
        if minade_idx == minfde_idx
        else f"minADE (m{minade_idx})"
    )
    if minade_idx != selected_idx:
        ax_traj.plot(
            candidates[minade_idx, :, 0], candidates[minade_idx, :, 1],
            color="#0284C7", lw=3.0, zorder=8, label=oracle_label,
        )
        ax_traj.scatter(
            *candidates[minade_idx, -1], color="#0284C7", s=82,
            edgecolors="black", linewidths=1.2, zorder=9,
        )

    selected_color = "#8A2BE2" if selected_idx == minade_idx else "#DC2626"
    selected_label = (
        f"D1 selected + minADE (m{selected_idx})"
        if selected_idx == minade_idx
        else f"D1 selected / top-1 (m{selected_idx})"
    )
    ax_traj.plot(
        candidates[selected_idx, :, 0], candidates[selected_idx, :, 1],
        color=selected_color, lw=3.2, zorder=9, label=selected_label,
    )
    ax_traj.scatter(
        *candidates[selected_idx, -1], color=selected_color, s=88,
        edgecolors="black", linewidths=1.4, zorder=10,
    )

    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.grid(True, linestyle=":", alpha=0.6)
    ax_traj.set_xlabel("Relative X (m)", fontsize=11, fontweight="semibold")
    ax_traj.set_ylabel("Relative Y (m)", fontsize=11, fontweight="semibold")
    ax_traj.set_title("Fold-local V2 Candidate Geometry", fontsize=12, fontweight="bold", pad=8)
    ax_traj.legend(loc="best", fontsize=8.2, framealpha=0.92, facecolor="white")

    sizes = 35.0 + 345.0 * probabilities
    for mode in range(len(candidates)):
        ax_score.scatter(
            fde[mode], ade[mode], s=sizes[mode], color="#94A3B8", alpha=0.55,
            edgecolors="#475569", linewidths=0.7, zorder=2,
        )
        ax_score.annotate(
            f"m{mode}", (fde[mode], ade[mode]), textcoords="offset points",
            xytext=(4, 4), fontsize=7.5, color="#64748B", alpha=0.85,
        )

    if native_idx != selected_idx:
        ax_score.scatter(
            fde[native_idx], ade[native_idx], s=max(sizes[native_idx], 105),
            color="#F59E0B", marker="D", edgecolors="black", linewidths=1.2,
            zorder=5,
            label=(f"Current native top-1 (m{native_idx}, ADE={ade[native_idx]:.2f}, "
                   f"FDE={fde[native_idx]:.2f})"),
        )

    if minfde_idx != minade_idx:
        ax_score.scatter(
            fde[minfde_idx], ade[minfde_idx], s=max(sizes[minfde_idx], 115),
            color="#2563EB", edgecolors="black", linewidths=1.3, zorder=6,
            label=f"minFDE (m{minfde_idx}, ADE={ade[minfde_idx]:.2f}, FDE={fde[minfde_idx]:.2f})",
        )

    if minade_idx != selected_idx:
        ax_score.scatter(
            fde[minade_idx], ade[minade_idx], s=max(sizes[minade_idx], 120),
            color="#0284C7", edgecolors="black", linewidths=1.4, zorder=7,
            label=f"minADE (m{minade_idx}, ADE={ade[minade_idx]:.2f}, FDE={fde[minade_idx]:.2f})",
        )
        ax_score.annotate(
            f"minADE\nm{minade_idx} (r{ranks[minade_idx]})",
            (fde[minade_idx], ade[minade_idx]), textcoords="offset points", xytext=(6, 6),
            fontsize=8.5, fontweight="bold", color="#0284C7",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#0284C7", alpha=0.85),
            zorder=8,
        )

    ax_score.scatter(
        fde[selected_idx], ade[selected_idx], s=max(sizes[selected_idx], 125),
        color=selected_color, edgecolors="black", linewidths=1.5, zorder=9,
        label=(f"D1 selected (m{selected_idx}, ADE={ade[selected_idx]:.2f}, "
               f"FDE={fde[selected_idx]:.2f})"),
    )
    selected_text = "D1 selected+minADE" if selected_idx == minade_idx else "D1 selected"
    ax_score.annotate(
        f"{selected_text}\nm{selected_idx} (r1)",
        (fde[selected_idx], ade[selected_idx]), textcoords="offset points", xytext=(6, 6),
        fontsize=8.5, fontweight="bold", color=selected_color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=selected_color, alpha=0.85),
        zorder=10,
    )
    ax_score.axvline(float(np.min(fde)), color="#2563EB", linestyle=":", alpha=0.45, lw=1.2)
    ax_score.axhline(float(np.min(ade)), color="#16A34A", linestyle=":", alpha=0.45, lw=1.2)
    ax_score.grid(True, linestyle=":", alpha=0.6)
    ax_score.set_xlabel("Final Displacement Error - FDE (m)", fontsize=11, fontweight="semibold")
    ax_score.set_ylabel("Average Displacement Error - ADE (m)", fontsize=11, fontweight="semibold")
    ax_score.set_title("D1 Reranked Score Distribution", fontsize=12, fontweight="bold", pad=8)
    ax_score.legend(loc="best", fontsize=8.0, framealpha=0.92, facecolor="white")

    delta_ade = ade[selected_idx] - ade[native_idx]
    delta_fde = fde[selected_idx] - fde[native_idx]
    changed = selected_idx != native_idx
    header = (
        "NEW METHOD OUTPUT | ETH development split | fold-local V2 candidates | "
        f"D1 social-gated reranker | best checkpoint | head seed {payload['seed']}\n"
        f"Sample {payload['sample_id']} (Index {payload['sample_idx']}) | "
        f"p(change)=gate={payload['p_change']:.3f} | Native m{native_idx}: "
        f"ADE={ade[native_idx]:.2f}m, FDE={fde[native_idx]:.2f}m -> D1 m{selected_idx}: "
        f"ADE={ade[selected_idx]:.2f}m, FDE={fde[selected_idx]:.2f}m | changed={changed}\n"
        f"Delta vs current native: ADE={delta_ade:+.2f}m, FDE={delta_fde:+.2f}m | "
        f"minADE=m{minade_idx} ({ade[minade_idx]:.2f}m, rank={ranks[minade_idx]}/20) | "
        f"D1 regret={ade[selected_idx] - ade[minade_idx]:.2f}m | point size ~ softmax(D1 logit)"
    )
    fig.suptitle(
        header, fontsize=9.7, family="monospace", fontweight="medium", y=0.985,
        bbox=dict(boxstyle="round,pad=0.5", fc="#EFF6FF", ec="#2563EB", lw=1.0),
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.89])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if payload.get("export_vector", False):
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def stack_old_new(old_path: Path, new_path: Path, output: Path, seed: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(old_path) as old_source, Image.open(new_path) as new_source:
        old = old_source.convert("RGB")
        new = new_source.convert("RGB")
        width = max(old.width, new.width)
        label_height = 48
        separator = 14
        canvas = Image.new(
            "RGB", (width, label_height + old.height + separator + label_height + new.height), "white"
        )
        font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arialbd.ttf"
        try:
            font = ImageFont.truetype(str(font_path), 23)
        except OSError:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, width, label_height), fill="#E2E8F0")
        draw.text((18, 11), "OLD | prior F16 validation visualization", fill="#0F172A", font=font)
        old_x = (width - old.width) // 2
        canvas.paste(old, (old_x, label_height))
        second_y = label_height + old.height + separator
        draw.rectangle((0, second_y, width, second_y + label_height), fill="#DBEAFE")
        draw.text(
            (18, second_y + 11),
            f"NEW | fold-local ETH V2 + D1 social-gated reranker (best, head seed {seed})",
            fill="#1D4ED8", font=font,
        )
        new_x = (width - new.width) // 2
        canvas.paste(new, (new_x, second_y + label_height))
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, format="PNG", optimize=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No metrics rows")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    new_dir = output / "new_method"
    combined_dir = output / "old_top_new_bottom"
    new_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    mpl_dir = ROOT / ".cache" / "mpl_regime_curated_comparison"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib
    matplotlib.use("Agg")

    for required in (OLD_CACHE, OLD_MANIFEST, FORMAL_CACHE, FORMAL_CONFIG, FORMAL_RUNNER):
        if not required.is_file():
            raise FileNotFoundError(required)

    config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    if args.seed not in config["head_seeds"]:
        raise ValueError(f"Seed {args.seed} is not in the formal protocol")
    model_dir = HERE / "results" / "models" / "eth" / f"seed_{args.seed}"
    detector_path = model_dir / "d1_social.pth"
    head_path = model_dir / "d1_social_gate_best.pth"
    if not detector_path.is_file() or not head_path.is_file():
        raise FileNotFoundError(f"Missing D1 artifacts under {model_dir}")

    with np.load(OLD_CACHE, allow_pickle=False) as source:
        old = {key: source[key] for key in source.files}
    with np.load(FORMAL_CACHE, allow_pickle=False) as source:
        formal = {key: source[key] for key in source.files}
    id_to_index = verify_alignment(old, formal)

    runner = load_module("formal_regime_visualization_runner", FORMAL_RUNNER)
    device = torch.device(args.device)
    p_change = runner.detector_probability(formal, detector_path, device)
    head, saved = runner._load_head(head_path, config, device)
    if saved.get("arm") != "d1_social_gate" or saved.get("fold") != "eth" or saved.get("seed") != args.seed:
        raise RuntimeError("D1 head checkpoint metadata does not match requested fold/seed/arm")
    d1_logits = runner._predict_head(head, saved, formal, p_change, device)
    if d1_logits.shape != formal["logits"].shape or not np.isfinite(d1_logits).all():
        raise RuntimeError("Invalid D1 logits")

    source_manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    rows = curated_rows(source_manifest)
    manifest_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    selection_changed = 0
    improved = 0
    worsened = 0
    tied = 0

    for number, row in enumerate(rows, 1):
        sample_id = int(row["sample_id"])
        if sample_id not in id_to_index:
            raise RuntimeError(f"Curated sample {sample_id} absent from formal ETH-dev cache")
        index = id_to_index[sample_id]
        old_path = OLD_IMAGES / row["filename"]
        if not old_path.is_file():
            raise FileNotFoundError(old_path)

        candidates = formal["candidate"][index, :, :24].reshape(20, 12, 2)
        native_logits = formal["logits"][index]
        new_logits = d1_logits[index]
        native_idx = int(np.argmax(native_logits))
        selected_idx = int(np.argmax(new_logits))
        minade_idx = int(np.argmin(formal["ade"][index]))
        minfde_idx = int(np.argmin(formal["fde"][index]))
        ranks = descending_ranks(new_logits)
        probabilities = softmax(new_logits)
        delta_ade = float(formal["ade"][index, selected_idx] - formal["ade"][index, native_idx])
        delta_fde = float(formal["fde"][index, selected_idx] - formal["fde"][index, native_idx])
        selection_changed += int(selected_idx != native_idx)
        if delta_ade < -1e-7:
            improved += 1
        elif delta_ade > 1e-7:
            worsened += 1
        else:
            tied += 1

        safe_stem = re.sub(r"\.png$", "", row["filename"], flags=re.IGNORECASE)
        new_name = f"{safe_stem}__NEW_D1_best_seed{args.seed}.png"
        comparison_name = f"{safe_stem}__OLD_TOP__NEW_BOTTOM.png"
        new_path = new_dir / new_name
        comparison_path = combined_dir / comparison_name
        if args.overwrite or not new_path.is_file():
            draw_new_method(
                {
                    "sample_id": sample_id,
                    "sample_idx": index,
                    "seed": args.seed,
                    "p_change": float(p_change[index]),
                    "past": old["past_relative"][index],
                    "future": old["future_relative"][index],
                    "candidates": candidates,
                    "ade": formal["ade"][index],
                    "fde": formal["fde"][index],
                    "probabilities": probabilities,
                    "native_idx": native_idx,
                    "selected_idx": selected_idx,
                    "minade_idx": minade_idx,
                    "minfde_idx": minfde_idx,
                    "ranks": ranks,
                    "export_vector": args.export_vector,
                },
                new_path,
                args.dpi,
            )
        if args.overwrite or not comparison_path.is_file():
            stack_old_new(old_path, new_path, comparison_path, args.seed)

        metric = {
            "category": row["category"],
            "sample_id": sample_id,
            "cache_index": index,
            "p_change": float(p_change[index]),
            "old_selected_ade": float(row["selected_ade_m"]),
            "old_oracle_minade": float(row["oracle_min_ade_m"]),
            "current_native_mode": native_idx,
            "current_native_ade": float(formal["ade"][index, native_idx]),
            "current_native_fde": float(formal["fde"][index, native_idx]),
            "d1_selected_mode": selected_idx,
            "d1_selected_ade": float(formal["ade"][index, selected_idx]),
            "d1_selected_fde": float(formal["fde"][index, selected_idx]),
            "d1_delta_ade_vs_current_native": delta_ade,
            "d1_delta_fde_vs_current_native": delta_fde,
            "selection_changed": bool(selected_idx != native_idx),
            "minade_mode": minade_idx,
            "minade": float(formal["ade"][index, minade_idx]),
            "minade_d1_score_rank": int(ranks[minade_idx]),
        }
        metrics_rows.append(metric)
        manifest_rows.append(
            {
                **metric,
                "old_image": str(old_path.resolve()),
                "new_image": str(new_path.resolve()),
                "comparison_image": str(comparison_path.resolve()),
            }
        )
        print(f"[{number:02d}/25] sample {sample_id}: native m{native_idx} -> D1 m{selected_idx}, dADE={delta_ade:+.3f}")

    write_csv(output / "paired_metrics.csv", metrics_rows)
    summary = {
        "purpose": "Visual comparison of legacy curated validation images and formal D1 outputs",
        "source_old_images": str(OLD_IMAGES.resolve()),
        "source_old_cache": str(OLD_CACHE.resolve()),
        "source_formal_cache": str(FORMAL_CACHE.resolve()),
        "fold": "eth",
        "split": "development",
        "arm": "d1_social_gate",
        "checkpoint": "best",
        "head_seed": args.seed,
        "candidate_generator": "fold-local EXP-156B F16 V2 seed-1 best_selected",
        "entries": len(manifest_rows),
        "unique_sample_ids": len({row["sample_id"] for row in manifest_rows}),
        "selection_changed_entries": selection_changed,
        "d1_ade_improved_entries": improved,
        "d1_ade_worsened_entries": worsened,
        "d1_ade_tied_entries": tied,
        "note": (
            "Counts are per image entry; duplicated sample IDs occur across curated categories. "
            "This is a dev-set diagnostic, not a held-out test estimate."
        ),
        "images": manifest_rows,
    }
    (output / "comparison_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = (
        "# Old/New curated visualization comparison\n\n"
        "- Top panel in every composite: unchanged legacy curated PNG.\n"
        "- Bottom panel: fold-local ETH V2 candidates scored by the D1 social-gated reranker, "
        f"best checkpoint, head seed {args.seed}.\n"
        "- The matched sample key is `sample_id`; cache index equality was also verified.\n"
        "- The formal cache candidates were independently checked against the legacy cache ground truth: "
        "recomputed ADE/FDE matched the stored formal metrics within 2e-5 m.\n"
        "- These samples belong to the ETH development split used for checkpoint selection, not held-out test.\n\n"
        f"Rendered {len(manifest_rows)} image pairs ({summary['unique_sample_ids']} unique samples). "
        f"D1 changed the current fold-local native top-1 in {selection_changed} entries; "
        f"ADE improved/worsened/tied in {improved}/{worsened}/{tied} entries.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "entries", "unique_sample_ids", "selection_changed_entries",
        "d1_ade_improved_entries", "d1_ade_worsened_entries", "d1_ade_tied_entries",
    )}, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
