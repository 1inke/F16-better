"""Analyze H3 Diversity x Oracle-Coverage attribution on validation only."""
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
    str(REPO), str(REPO / "SELECTOR_BENCHMARK_SERIES"),
    str(REPO / "FRONT_ENCODER_SERIES"),
    str(REPO / "DIVERSITY_STAGE_ATTRIBUTION_SERIES"),
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

import analyze_stage_attribution as base
import run_h3_objective_attribution as runner
from build_candidate_cache import configure_astra_env, load_module


CONDITIONS = {
    "d1_o1": base.Condition("d1_o1", "D=.05, O=.20", "h3", 0.05, None, runner.H3_BASE),
    "d0_o1": base.Condition("d0_o1", "D=0, O=.20", "h3", 0.0, None, runner.H3_NODIV),
    "d1_o0": base.Condition("d1_o0", "D=.05, O=0", "h3", 0.05, None, runner.new_run_dir("d1_o0")),
    "d0_o0": base.Condition("d0_o0", "D=0, O=0", "h3", 0.0, None, runner.new_run_dir("d0_o0")),
}
ORACLE_WEIGHT = {"d1_o1": 0.20, "d0_o1": 0.20, "d1_o0": 0.0, "d0_o0": 0.0}
ORDER = ["d1_o1", "d0_o1", "d1_o0", "d0_o0"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(
    condition: base.Condition,
    checkpoint_name: str,
    cfg,
    loader: DataLoader,
    sample_ids: np.ndarray,
    trainer,
    device: torch.device,
) -> dict[str, np.ndarray]:
    local_cfg = copy.deepcopy(cfg)
    model = base.load_model(condition, local_cfg, device)
    checkpoint = condition.run_dir / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint_name != "best_selected.pth":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
    candidates, ades, fdes, logits = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            past, future, num_valid, *_ = batch
            past, future = past.to(device), future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(
                past - center, None, None, mode="testing",
                num_valid=num_valid, graph_loc=past,
            )
            raw_prediction = output[2] if isinstance(output, (tuple, list)) else output
            focal = (raw_prediction + center.unsqueeze(-3))[:, 0]
            ground_truth = future[:, 0]
            error = torch.linalg.vector_norm(focal - ground_truth[:, None], dim=-1)
            candidates.append(focal.cpu().numpy())
            ades.append(error.mean(dim=-1).cpu().numpy())
            fdes.append(error[..., -1].cpu().numpy())
            logits.append(model.last_mode_logits[:, 0].cpu().numpy())
    return {
        "sample_id": sample_ids.copy(),
        "candidate": np.concatenate(candidates).astype(np.float32),
        "ade": np.concatenate(ades).astype(np.float32),
        "fde": np.concatenate(fdes).astype(np.float32),
        "logits": np.concatenate(logits).astype(np.float32),
    }


def bootstrap(delta: np.ndarray, repetitions: int, seed: int) -> tuple[float, float, float]:
    return base.bootstrap_mean_ci(delta, repetitions=repetitions, seed=seed)


def factorial_rows(
    values: dict[str, dict[str, np.ndarray]],
    checkpoint_set: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    metrics = [
        "endpoint_spread", "support_0p5", "support_1p0", "ade_lt_0p5",
        "ade_lt_1p0", "delta_ade_1_2", "minade", "minfde",
        "selected_ade", "regret", "active_pair_rate_sample", "active_sample",
        "diversity_penalty",
    ]
    rows = []
    for metric_index, metric in enumerate(metrics):
        d1o1, d0o1, d1o0, d0o0 = (values[key][metric] for key in ORDER)
        contrasts = {
            "coverage_on_minus_off_given_div_on": d1o1 - d1o0,
            "coverage_on_minus_off_given_div_off": d0o1 - d0o0,
            "coverage_main_effect": 0.5 * ((d1o1 - d1o0) + (d0o1 - d0o0)),
            "diversity_on_minus_off_given_coverage_on": d1o1 - d0o1,
            "diversity_on_minus_off_given_coverage_off": d1o0 - d0o0,
            "diversity_main_effect": 0.5 * ((d1o1 - d0o1) + (d1o0 - d0o0)),
            "interaction": d1o1 - d1o0 - d0o1 + d0o0,
        }
        for contrast_index, (name, delta) in enumerate(contrasts.items()):
            used_seed = seed + metric_index * 20 + contrast_index
            estimate, low, high = bootstrap(delta, repetitions, used_seed)
            rows.append({
                "checkpoint_set": checkpoint_set,
                "metric": metric,
                "contrast": name,
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": used_seed,
            })
    return rows


def contrast_lookup(rows: list[dict[str, Any]], checkpoint_set: str, metric: str, contrast: str) -> dict[str, Any]:
    for row in rows:
        if row["checkpoint_set"] == checkpoint_set and row["metric"] == metric and row["contrast"] == contrast:
            return row
    raise KeyError((checkpoint_set, metric, contrast))


def classify(factorial: list[dict[str, Any]], summaries: dict[str, dict[str, dict[str, Any]]]) -> tuple[str, str]:
    best_support = contrast_lookup(factorial, "best", "support_1p0", "coverage_main_effect")["estimate"]
    final_support = contrast_lookup(factorial, "final", "support_1p0", "coverage_main_effect")["estimate"]
    best_spread = contrast_lookup(factorial, "best", "endpoint_spread", "coverage_main_effect")["estimate"]
    final_spread = contrast_lookup(factorial, "final", "endpoint_spread", "coverage_main_effect")["estimate"]
    best_minade = contrast_lookup(factorial, "best", "minade", "coverage_main_effect")["estimate"]
    final_minade = contrast_lookup(factorial, "final", "minade", "coverage_main_effect")["estimate"]
    off_best = 0.5 * (summaries["best"]["d1_o0"]["minade"] + summaries["best"]["d0_o0"]["minade"])
    off_final = 0.5 * (summaries["final"]["d1_o0"]["minade"] + summaries["final"]["d0_o0"]["minade"])
    best_rel = best_minade / off_best
    final_rel = final_minade / off_final
    useful_support = best_support >= 0.5 and final_support >= 0.5
    useful_oracle = best_rel <= -0.02 and final_rel <= -0.02
    dispersion_only = (
        best_spread >= 0.10 and final_spread >= 0.10
        and not useful_support and not useful_oracle
    )
    if useful_support or useful_oracle:
        return "COVERAGE_IS_USEFUL_ALLOCATOR", "Coverage materially improves GT-local allocation and/or oracle quality in both checkpoint views."
    if dispersion_only:
        return "COVERAGE_IS_DISPERSION_DRIVER", "Coverage increases global spread without a material local-coverage or oracle benefit."
    if abs(best_support) < 0.5 and abs(final_support) < 0.5 and abs(best_spread) < 0.10 and abs(final_spread) < 0.10 and abs(best_rel) < 0.02 and abs(final_rel) < 0.02:
        return "NO_LARGE_COVERAGE_EFFECT", "Coverage does not cross any preregistered engineering threshold."
    return "MIXED_CHECKPOINT_SENSITIVE", "Coverage effects are mixed or differ between best and matched epoch-30 checkpoints."


def format_float(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def render_report(
    path: Path,
    summaries: dict[str, dict[str, dict[str, Any]]],
    factorial: list[dict[str, Any]],
    verdict: tuple[str, str],
    provenance: dict[str, Any],
) -> None:
    lines = [
        "# H3 Diversity × Oracle-Coverage Objective Attribution",
        "",
        "- Scope: ETH, seed 1, fixed 5% scene-tail validation split; no test access.",
        "- Primary: each condition's oracle-combo best checkpoint.",
        "- Sensitivity: all conditions at matched epoch 30 (`final_model.pth`).",
        "- Paired bootstrap conditions on the trained models and does not measure training-seed uncertainty.",
        f"- Verdict: **{verdict[0]}** — {verdict[1]}",
    ]
    for checkpoint_set in ("best", "final"):
        lines += [
            "", f"## {checkpoint_set.title()} checkpoint summary", "",
            "| Condition | Diversity | Coverage | Selected ADE | minADE | minFDE | Spread | Support@0.5 | Support@1m | ADE<1m |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key in ORDER:
            row = summaries[checkpoint_set][key]
            lines.append(
                f"| {key.upper()} | {row['h3_weight']:.2f} | {ORACLE_WEIGHT[key]:.2f} | "
                f"{format_float(row['selected_ade'])} | {format_float(row['minade'])} | "
                f"{format_float(row['minfde'])} | {format_float(row['endpoint_spread'], 3)} | "
                f"{format_float(row['support_0p5'], 2)} | {format_float(row['support_1p0'], 2)} | "
                f"{format_float(row['ade_lt_1p0'], 2)} |"
            )
    lines += [
        "", "## Coverage main effect", "",
        "Positive delta means coverage=.20 minus coverage=0. For ADE/FDE, negative is better; for support, positive is better.",
        "",
        "| Checkpoint | Metric | Delta | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for checkpoint_set in ("best", "final"):
        for metric in ("endpoint_spread", "support_0p5", "support_1p0", "ade_lt_1p0", "minade", "minfde", "selected_ade"):
            row = contrast_lookup(factorial, checkpoint_set, metric, "coverage_main_effect")
            lines.append(
                f"| {checkpoint_set} | {metric} | {format_float(row['estimate'], 5)} | "
                f"[{format_float(row['ci95_low'], 5)}, {format_float(row['ci95_high'], 5)}] |"
            )
    lines += [
        "", "## Decision scales", "",
        "- Endpoint spread: |Δ| ≥ 0.10 m.",
        "- Support@1m or ADE<1m count: |Δ| ≥ 0.5 candidate/sample.",
        "- minADE/minFDE: relative |Δ| ≥ 2%.",
        "- A mechanism claim requires the same direction at best checkpoint and matched epoch 30.",
        "", "## Provenance", "", "```json",
        json.dumps(provenance, indent=2, ensure_ascii=False), "```", "",
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

    audit = runner.preflight(args.astra_root.resolve())
    for key, condition in CONDITIONS.items():
        for name in ("best_selected.pth", "final_model.pth", "split_indices.npz"):
            path = condition.run_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
    sample_ids = np.load(CONDITIONS["d1_o1"].split_indices)["validation"].astype(np.int64)
    for key in ORDER[1:]:
        current = np.load(CONDITIONS[key].split_indices)["validation"].astype(np.int64)
        if not np.array_equal(sample_ids, current):
            raise RuntimeError(f"Validation IDs differ for {key}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = configure_astra_env(REPO, args.astra_root.resolve(), device)
    trainer = load_module("h3_objective_trainer", REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py")
    data_dir = args.astra_root.resolve() / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer.TrajectoryDataset(data_dir / "eth_data_train.npy", data_dir / "eth_num_train.npy", cfg)
    loader = DataLoader(
        Subset(dataset, sample_ids.tolist()), batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )

    output = HERE / "results" / "analysis"
    cache_root = output / "validation_candidate_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    all_values: dict[str, dict[str, dict[str, np.ndarray]]] = {"best": {}, "final": {}}
    all_summaries: dict[str, dict[str, dict[str, Any]]] = {"best": {}, "final": {}}
    summary_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {
        "fold": "eth", "seed": 1, "validation_only": True,
        "test_accessed_by_this_analysis": False,
        "validation_sample_count": len(sample_ids),
        "validation_id_sha256": hashlib.sha256(sample_ids.tobytes()).hexdigest(),
        "preflight": audit, "conditions": {},
    }
    for checkpoint_set, checkpoint_name in (("best", "best_selected.pth"), ("final", "final_model.pth")):
        for key in ORDER:
            condition = CONDITIONS[key]
            print(f"[EVAL] {checkpoint_set}/{key}")
            raw = evaluate_checkpoint(condition, checkpoint_name, cfg, loader, sample_ids, trainer, device)
            np.savez_compressed(cache_root / f"{checkpoint_set}_{key}.npz", **raw)
            values = base.derive(raw)
            summary = base.summarize(condition, values)
            summary["oracle_coverage_weight"] = ORACLE_WEIGHT[key]
            summary["checkpoint_set"] = checkpoint_set
            all_values[checkpoint_set][key] = values
            all_summaries[checkpoint_set][key] = summary
            summary_rows.append(summary)
            checkpoint_path = condition.run_dir / checkpoint_name
            provenance["conditions"].setdefault(key, {})[checkpoint_set] = {
                "checkpoint": str(checkpoint_path),
                "sha256": runner.sha256(checkpoint_path),
            }
    factorial = []
    factorial.extend(factorial_rows(all_values["best"], "best", args.bootstrap_repetitions, args.bootstrap_seed))
    factorial.extend(factorial_rows(all_values["final"], "final", args.bootstrap_repetitions, args.bootstrap_seed + 10_000))
    verdict = classify(factorial, all_summaries)

    write_csv(output / "condition_summary.csv", summary_rows)
    write_csv(output / "factorial_paired_bootstrap.csv", factorial)
    (output / "verdict.json").write_text(
        json.dumps({"code": verdict[0], "explanation": verdict[1]}, indent=2), encoding="utf-8"
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = output / "H3_OBJECTIVE_ATTRIBUTION_REPORT.md"
    render_report(report, all_summaries, factorial, verdict, provenance)
    print(f"[DONE] {report}")


if __name__ == "__main__":
    main()
