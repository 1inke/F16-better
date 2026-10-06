"""Validation-only analysis of H3 oracle-coverage propagation through V2."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [
    str(REPO),
    str(REPO / "SELECTOR_BENCHMARK_SERIES"),
    str(REPO / "FRONT_ENCODER_SERIES"),
    str(REPO / "DIVERSITY_STAGE_ATTRIBUTION_SERIES"),
    str(REPO / "H3_OBJECTIVE_ATTRIBUTION_SERIES"),
]

if sys.platform == "win32":
    torch_lib = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import analyze_h3_objective_attribution as h3_analysis
import analyze_stage_attribution as base
import run_v2_coverage_propagation as runner
from build_candidate_cache import configure_astra_env, load_module


CONDITIONS = {
    "h3_on": base.Condition(
        "h3_on", "H3 O=.20", "h3", 0.05, None, runner.H3_ON_RUN
    ),
    "h3_off": base.Condition(
        "h3_off", "H3 O=0", "h3", 0.05, None, runner.H3_OFF_RUN
    ),
    "v2_on": base.Condition(
        "v2_on", "H3 O=.20 -> V2 O=.20", "v2", 0.05, 0.05,
        runner.V2_ON_RUN, runner.H3_ON_RUN / "best_selected.pth",
    ),
    "v2_off": base.Condition(
        "v2_off", "H3 O=0 -> V2 O=.20", "v2", 0.05, 0.05,
        runner.NEW_RUN, runner.H3_OFF_RUN / "best_selected.pth",
    ),
}
STAGE_PAIRS = {"h3": ("h3_on", "h3_off"), "v2": ("v2_on", "v2_off")}
ORDER = ["h3_on", "h3_off", "v2_on", "v2_off"]
METRICS = [
    "j", "minade", "minfde", "samepath_fde", "selected_ade", "selected_fde",
    "regret", "endpoint_spread", "support_0p5", "support_1p0",
    "ade_lt_0p5", "ade_lt_1p0", "oracle_recall_1", "oracle_recall_3",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def derive(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = base.derive(raw)
    values["j"] = values["minade"] + 0.5 * values["minfde"]
    return values


def paired_rows(
    values_by_checkpoint: dict[str, dict[str, dict[str, np.ndarray]]],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint_set in enumerate(("best", "final")):
        values_by_key = values_by_checkpoint[checkpoint_set]
        for stage_index, (stage, (on_key, off_key)) in enumerate(STAGE_PAIRS.items()):
            on, off = values_by_key[on_key], values_by_key[off_key]
            if not np.array_equal(on["sample_id"], off["sample_id"]):
                raise RuntimeError(f"Paired IDs do not align for {checkpoint_set}/{stage}")
            for metric_index, metric in enumerate(METRICS):
                delta = on[metric] - off[metric]
                used_seed = seed + checkpoint_index * 10_000 + stage_index * 100 + metric_index
                estimate, low, high = base.bootstrap_mean_ci(
                    delta, repetitions=repetitions, seed=used_seed
                )
                off_mean = float(np.mean(off[metric]))
                relative = estimate / off_mean if abs(off_mean) > 1e-12 else float("nan")
                rows.append({
                    "checkpoint_set": checkpoint_set,
                    "stage": stage,
                    "contrast": "coverage_on_source_minus_off_source",
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "off_reference_mean": off_mean,
                    "relative_effect": relative,
                    "relative_effect_pct": 100.0 * relative,
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": used_seed,
                })
    return rows


def row_lookup(
    rows: list[dict[str, Any]], checkpoint_set: str, stage: str, metric: str
) -> dict[str, Any]:
    for row in rows:
        if (
            row["checkpoint_set"] == checkpoint_set
            and row["stage"] == stage
            and row["metric"] == metric
        ):
            return row
    raise KeyError((checkpoint_set, stage, metric))


def effect_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for checkpoint_set in ("best", "final"):
        output[checkpoint_set] = {
            "j_rel": float(row_lookup(rows, checkpoint_set, "v2", "j")["relative_effect"]),
            "minade_rel": float(row_lookup(rows, checkpoint_set, "v2", "minade")["relative_effect"]),
            "minfde_rel": float(row_lookup(rows, checkpoint_set, "v2", "minfde")["relative_effect"]),
            "support_1p0": float(row_lookup(rows, checkpoint_set, "v2", "support_1p0")["estimate"]),
            "selected_ade_rel": float(row_lookup(rows, checkpoint_set, "v2", "selected_ade")["relative_effect"]),
        }
    return output


def classify(effects: dict[str, dict[str, float]]) -> tuple[str, str]:
    def quality_gate(item: dict[str, float]) -> bool:
        joint = item["j_rel"] <= -0.02
        one_strong = (
            item["minade_rel"] <= -0.02 and item["minfde_rel"] <= 0.005
        ) or (
            item["minfde_rel"] <= -0.02 and item["minade_rel"] <= 0.005
        )
        return joint or one_strong

    quality = all(quality_gate(effects[key]) for key in ("best", "final"))
    local_support = all(effects[key]["support_1p0"] >= 0.5 for key in ("best", "final"))
    selection_guard = all(
        effects[key]["selected_ade_rel"] <= 0.005 for key in ("best", "final")
    )
    direction_consistent = all(
        effects[key]["j_rel"] < 0.0 and effects[key]["support_1p0"] > 0.0
        for key in ("best", "final")
    )
    if quality and local_support and selection_guard and direction_consistent:
        return (
            "H3_COVERAGE_PROPAGATES_THROUGH_V2",
            "The H3 coverage-on source retains an engineering-meaningful oracle and local-support advantage after matched V2 training.",
        )

    small = all(
        abs(effects[key]["j_rel"]) < 0.01
        and abs(effects[key]["support_1p0"]) < 0.5
        for key in ("best", "final")
    )
    reversed_effect = all(
        effects[key]["j_rel"] >= 0.0 or effects[key]["support_1p0"] <= 0.0
        for key in ("best", "final")
    )
    if small or reversed_effect:
        return (
            "H3_COVERAGE_GAIN_ABSORBED_BY_V2",
            "The H3 coverage advantage is below the preregistered V2 propagation scale or reverses after V2 training.",
        )
    return (
        "MIXED_OR_CHECKPOINT_SENSITIVE",
        "The propagation evidence does not jointly pass quality, local-support, selection-guard, and checkpoint-consistency gates.",
    )


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def render_report(
    path: Path,
    summaries: dict[str, dict[str, dict[str, Any]]],
    paired: list[dict[str, Any]],
    verdict: tuple[str, str],
    provenance: dict[str, Any],
) -> None:
    lines = [
        "# H3 Coverage → V2 Propagation Experiment",
        "",
        "- Scope: ETH, seed 1, fixed 5% scene-tail validation split (N=1,516).",
        "- New training: one V2 run from H3 D=.05/O=0; V2 itself remains D=.05/O=.20.",
        "- Reference: existing EXP-173 C00 from H3 D=.05/O=.20; this analysis reads validation data only.",
        "- Sensitivity: oracle-combo best checkpoint and matched epoch-20 final model.",
        "- Paired bootstrap is conditional on the trained models and does not estimate training-seed uncertainty.",
        f"- Verdict: **{verdict[0]}** — {verdict[1]}",
    ]
    for checkpoint_set in ("best", "final"):
        lines += [
            "",
            f"## {checkpoint_set.title()} checkpoint summary",
            "",
            "| Condition | Stage | H3 coverage | V2 coverage | J | minADE | minFDE | Same-path FDE | Selected ADE | Spread | Support@1m | ADE<1m |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key in ORDER:
            s = summaries[checkpoint_set][key]
            v2_coverage = (
                "NA" if s["v2_coverage_weight"] is None
                else f"{s['v2_coverage_weight']:.2f}"
            )
            lines.append(
                f"| {s['label']} | {s['stage']} | {s['h3_coverage_weight']:.2f} | "
                f"{v2_coverage} | "
                f"{fmt(s['j'])} | {fmt(s['minade'])} | {fmt(s['minfde'])} | "
                f"{fmt(s['samepath_fde'])} | {fmt(s['selected_ade'])} | "
                f"{fmt(s['endpoint_spread'], 3)} | {fmt(s['support_1p0'], 2)} | "
                f"{fmt(s['ade_lt_1p0'], 2)} |"
            )

    lines += [
        "",
        "## Coverage-source effect before and after V2",
        "",
        "Delta is coverage-on source minus coverage-off source. Negative error delta and positive support delta are improvements.",
        "",
        "| Checkpoint | Stage | Metric | Delta | Relative | 95% CI |",
        "|---|---|---|---:|---:|---:|",
    ]
    for checkpoint_set in ("best", "final"):
        for stage in ("h3", "v2"):
            for metric in (
                "j", "minade", "minfde", "samepath_fde", "selected_ade",
                "endpoint_spread", "support_1p0", "ade_lt_1p0",
            ):
                row = row_lookup(paired, checkpoint_set, stage, metric)
                lines.append(
                    f"| {checkpoint_set} | {stage} | {metric} | {fmt(row['estimate'], 5)} | "
                    f"{fmt(row['relative_effect_pct'], 2)}% | "
                    f"[{fmt(row['ci95_low'], 5)}, {fmt(row['ci95_high'], 5)}] |"
                )

    effects = effect_summary(paired)
    lines += [
        "",
        "## Preregistered advancement gates",
        "",
        "- Same direction at best and matched epoch 20.",
        "- V2 J improves by at least 2%, or minADE/minFDE has one ≥2% gain while the other degrades by no more than 0.5%.",
        "- Support@1m increases by at least 0.5 candidate/sample.",
        "- Selected ADE does not degrade by more than 0.5%.",
        "- If |J change| <1% and |Support@1m change| <0.5 at both checkpoints, treat the H3 gain as absorbed by V2.",
        "",
        "| Checkpoint | V2 J rel. | V2 minADE rel. | V2 minFDE rel. | Support@1m delta | Selected ADE rel. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for checkpoint_set in ("best", "final"):
        e = effects[checkpoint_set]
        lines.append(
            f"| {checkpoint_set} | {fmt(100*e['j_rel'], 2)}% | "
            f"{fmt(100*e['minade_rel'], 2)}% | {fmt(100*e['minfde_rel'], 2)}% | "
            f"{fmt(e['support_1p0'], 3)} | {fmt(100*e['selected_ade_rel'], 2)}% |"
        )
    lines += [
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(provenance, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astra-root", type=Path, default=runner.ASTRA_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260922)
    args = parser.parse_args()

    audit = runner.preflight(args.astra_root.resolve(), require_new_v2=True)
    for condition in CONDITIONS.values():
        for filename in ("best_selected.pth", "final_model.pth", "split_indices.npz"):
            path = condition.run_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)

    sample_ids = np.load(CONDITIONS["h3_on"].split_indices)["validation"].astype(np.int64)
    for key in ORDER[1:]:
        current = np.load(CONDITIONS[key].split_indices)["validation"].astype(np.int64)
        if not np.array_equal(sample_ids, current):
            raise RuntimeError(f"Validation IDs differ for {key}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = configure_astra_env(REPO, args.astra_root.resolve(), device)
    trainer = load_module(
        "v2_coverage_propagation_trainer",
        REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py",
    )
    data_dir = args.astra_root.resolve() / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer.TrajectoryDataset(
        data_dir / "eth_data_train.npy", data_dir / "eth_num_train.npy", cfg
    )
    loader = DataLoader(
        Subset(dataset, sample_ids.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    output = HERE / "results" / "analysis"
    cache_root = output / "validation_candidate_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    values_by_checkpoint: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "best": {}, "final": {}
    }
    summaries: dict[str, dict[str, dict[str, Any]]] = {"best": {}, "final": {}}
    summary_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {
        "fold": "eth",
        "seed": 1,
        "validation_only": True,
        "test_accessed_by_this_analysis": False,
        "validation_sample_count": int(len(sample_ids)),
        "validation_id_sha256": hashlib.sha256(sample_ids.tobytes()).hexdigest(),
        "preflight": audit,
        "conditions": {},
    }
    coverage = {
        "h3_on": (0.20, None),
        "h3_off": (0.0, None),
        "v2_on": (0.20, 0.20),
        "v2_off": (0.0, 0.20),
    }
    for checkpoint_set, checkpoint_name in (
        ("best", "best_selected.pth"), ("final", "final_model.pth")
    ):
        for key in ORDER:
            condition = CONDITIONS[key]
            print(f"[EVAL] {checkpoint_set}/{key}")
            raw = h3_analysis.evaluate_checkpoint(
                condition, checkpoint_name, copy.deepcopy(cfg), loader,
                sample_ids, trainer, device,
            )
            np.savez_compressed(cache_root / f"{checkpoint_set}_{key}.npz", **raw)
            values = derive(raw)
            values_by_checkpoint[checkpoint_set][key] = values
            summary = base.summarize(condition, values)
            summary["j"] = float(values["j"].mean())
            summary["h3_coverage_weight"] = coverage[key][0]
            summary["v2_coverage_weight"] = coverage[key][1]
            summary["checkpoint_set"] = checkpoint_set
            summaries[checkpoint_set][key] = summary
            summary_rows.append(summary)
            checkpoint = condition.run_dir / checkpoint_name
            provenance["conditions"].setdefault(key, {})[checkpoint_set] = {
                "checkpoint": str(checkpoint),
                "sha256": runner.sha256(checkpoint),
                "h3_source": str(condition.h3_source) if condition.h3_source else None,
                "h3_source_sha256": (
                    runner.sha256(condition.h3_source) if condition.h3_source else None
                ),
            }

    paired = paired_rows(
        values_by_checkpoint,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    effects = effect_summary(paired)
    verdict = classify(effects)
    write_csv(output / "condition_summary.csv", summary_rows)
    write_csv(output / "paired_bootstrap.csv", paired)
    (output / "effect_summary.json").write_text(
        json.dumps(effects, indent=2), encoding="utf-8"
    )
    (output / "verdict.json").write_text(
        json.dumps({"code": verdict[0], "explanation": verdict[1]}, indent=2),
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = output / "V2_COVERAGE_PROPAGATION_REPORT.md"
    render_report(report, summaries, paired, verdict, provenance)
    print(f"[DONE] {report}")


if __name__ == "__main__":
    main()
