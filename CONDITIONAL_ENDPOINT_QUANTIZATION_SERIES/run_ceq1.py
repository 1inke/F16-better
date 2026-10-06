"""CEQ1 validation-only conditional endpoint quantization experiment."""
from __future__ import annotations

import argparse
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
    str(HERE),
    str(REPO),
    str(REPO / "SELECTOR_BENCHMARK_SERIES"),
    str(REPO / "DIVERSITY_STAGE_ATTRIBUTION_SERIES"),
    str(REPO / "FRONT_ENCODER_SERIES"),
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

import analyze_stage_attribution as stage_analysis
from build_candidate_cache import build_f16_model, configure_astra_env, load_module
from ceq_core import (
    ConditionalFutureQuantizer,
    Standardizer,
    build_observation_features,
    calibration_objective,
    candidate_duplicate_rate,
    observation_motion_ratio,
    parameter_grid,
    raw_metrics,
    split_fit_calibration,
)


ASTRA_DEFAULT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
EXP173 = REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
BASELINE_RUN = (
    EXP173 / "results" / "eth" / "v2" / "e3"
    / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"
)
H3_RUN = (
    EXP173 / "results" / "eth" / "h3" / "e3"
    / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"
)
METRICS = (
    "j", "minade", "minfde", "samepath_fde", "selected_ade",
    "selected_fde", "regret", "endpoint_spread", "support_0p5",
    "support_1p0", "ade_lt_0p5", "ade_lt_1p0", "oracle_recall_1",
    "oracle_recall_3", "oracle_rank", "duplicate_pair_rate",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_ids() -> tuple[np.ndarray, np.ndarray]:
    with np.load(BASELINE_RUN / "split_indices.npz") as split:
        return split["train"].astype(np.int64), split["validation"].astype(np.int64)


def preflight(astra_root: Path) -> dict[str, Any]:
    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    required = [
        Path(__file__), HERE / "ceq_core.py",
        data_dir / "eth_data_train.npy", data_dir / "eth_num_train.npy",
        BASELINE_RUN / "best_selected.pth", BASELINE_RUN / "split_indices.npz",
        BASELINE_RUN / "metrics.json", BASELINE_RUN / "locked_backend_contract.json",
        H3_RUN / "best_selected.pth",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("CEQ1 missing inputs:\n  " + "\n  ".join(missing))
    contract = json.loads(
        (BASELINE_RUN / "locked_backend_contract.json").read_text(encoding="utf-8")
    )
    if contract.get("decoder_class") != "JointTrainableV2Decoder":
        raise RuntimeError("baseline decoder identity changed")
    if int(contract["parameter_counts"]["total"]) != 321988:
        raise RuntimeError("baseline parameter identity changed")
    train, validation = split_ids()
    if len(validation) != 1516 or len(np.unique(validation)) != len(validation):
        raise RuntimeError("validation identity is not the locked 1,516-sample split")
    if np.intersect1d(train, validation).size:
        raise RuntimeError("train and validation IDs overlap")
    return {
        "experiment": "CEQ1_CONDITIONAL_EMPIRICAL_FUTURE_QUANTIZATION",
        "scope": "ETH seed 1 validation-only feasibility pilot",
        "new_neural_training": False,
        "test_paths_read_or_passed": False,
        "candidate_family": (
            "training-only conditional empirical future distribution; weighted "
            "full-trajectory Lloyd components; component mass is score"
        ),
        "not_used": [
            "pairwise repulsion", "fixed global endpoint prototypes",
            "hard candidate-family quota", "validation future in retrieval features",
        ],
        "split": {
            "train_count": int(len(train)),
            "validation_count": int(len(validation)),
            "validation_id_sha256": hashlib.sha256(validation.tobytes()).hexdigest(),
            "split_file_sha256": sha256(BASELINE_RUN / "split_indices.npz"),
        },
        "baseline": {
            "checkpoint": str(BASELINE_RUN / "best_selected.pth"),
            "checkpoint_sha256": sha256(BASELINE_RUN / "best_selected.pth"),
            "contract": contract,
        },
    }


def load_coordinates(dataset) -> tuple[np.ndarray, np.ndarray]:
    total = dataset.past_frames + dataset.future_frames
    coordinates = np.asarray(
        dataset.data[:, : dataset.max_agents, :total, -2:], dtype=np.float32
    ) / float(dataset.scale)
    counts = np.minimum(np.asarray(dataset.num, dtype=np.int64), dataset.max_agents)
    return coordinates, counts


def evaluate_baseline(model, loader, sample_ids: np.ndarray, trainer) -> dict[str, np.ndarray]:
    candidates, ades, fdes, logits = [], [], [], []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for batch in loader:
            past, future, num_valid, *_ = batch
            past, future = past.to(device), future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            prediction = model(
                past - center, None, None, mode="testing",
                num_valid=num_valid, graph_loc=past,
            )[2] + center.unsqueeze(-3)
            candidate = prediction[:, 0]
            target = future[:, 0]
            error = torch.linalg.vector_norm(candidate - target[:, None], dim=-1)
            candidates.append(candidate.cpu().numpy())
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


def derive(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = stage_analysis.derive(raw)
    values["j"] = values["minade"] + 0.5 * values["minfde"]
    values["duplicate_pair_rate"] = candidate_duplicate_rate(raw["candidate"])
    return values


def summary(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float | int]:
    out: dict[str, float | int] = {"N": int(mask.sum())}
    for metric in METRICS:
        out[metric] = float(values[metric][mask].mean())
    out["oracle_rank_median"] = float(np.median(values["oracle_rank"][mask]))
    return out


def paired_rows(
    baseline: dict[str, np.ndarray],
    treatment: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset_index, (subset, mask) in enumerate(masks.items()):
        for metric_index, metric in enumerate(METRICS):
            delta = treatment[metric][mask] - baseline[metric][mask]
            used_seed = seed + subset_index * 100 + metric_index
            estimate, low, high = stage_analysis.bootstrap_mean_ci(
                delta, repetitions=repetitions, seed=used_seed
            )
            reference = float(baseline[metric][mask].mean())
            relative = estimate / reference if abs(reference) > 1.0e-12 else float("nan")
            rows.append({
                "subset": subset, "contrast": "ceq1_minus_c00", "metric": metric,
                "estimate": estimate, "ci95_low": low, "ci95_high": high,
                "baseline_mean": reference, "relative_effect": relative,
                "relative_effect_pct": 100.0 * relative,
                "bootstrap_repetitions": repetitions, "bootstrap_seed": used_seed,
            })
    return rows


def lookup(rows: list[dict[str, Any]], subset: str, metric: str) -> dict[str, Any]:
    for row in rows:
        if row["subset"] == subset and row["metric"] == metric:
            return row
    raise KeyError((subset, metric))


def classify(rows: list[dict[str, Any]], ceq_summary: dict[str, dict[str, Any]]) -> tuple[str, str, dict[str, bool]]:
    all_j = float(lookup(rows, "all", "j")["relative_effect"])
    all_minade = float(lookup(rows, "all", "minade")["relative_effect"])
    all_minfde = float(lookup(rows, "all", "minfde")["relative_effect"])
    selected = float(lookup(rows, "all", "selected_ade")["relative_effect"])
    support = float(lookup(rows, "all", "support_1p0")["estimate"])
    high_j = float(lookup(rows, "high_motion", "j")["relative_effect"])
    duplicate = float(ceq_summary["all"]["duplicate_pair_rate"])
    generation = all_j <= -0.02 or (
        all_minade <= -0.02 and all_minfde <= 0.005
    ) or (all_minfde <= -0.02 and all_minade <= 0.005)
    gates = {
        "generation_quality": generation,
        "selected_ade_improves_2pct": selected <= -0.02,
        "support_1m_increases_0p25": support >= 0.25,
        "high_motion_j_not_worse_1pct": high_j <= 0.01,
        "duplicate_pair_rate_below_10pct": duplicate <= 0.10,
    }
    if all(gates.values()):
        return (
            "CEQ1_ADVANCE_TO_NEURAL_DISTILLATION",
            "The training-only conditional distribution improves oracle geometry and mass-based selection without endpoint collapse.",
            gates,
        )
    if generation and gates["duplicate_pair_rate_below_10pct"]:
        return (
            "CEQ1_ORACLE_SIGNAL_ONLY",
            "The empirical distribution improves candidate geometry but its cluster mass is not yet a strong selector; distillation is blocked pending score analysis.",
            gates,
        )
    return (
        "STOP_CEQ_ROUTE",
        "The conditional empirical future family does not clear the geometry gate or collapses locally; do not build a neural distillation head.",
        gates,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def render_report(
    path: Path,
    summaries: dict[str, dict[str, dict[str, Any]]],
    rows: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    selected_parameters: dict[str, Any],
    verdict: tuple[str, str, dict[str, bool]],
    provenance: dict[str, Any],
) -> None:
    lines = [
        "# CEQ1 条件端点分布量化实验", "",
        "- 范围：ETH seed 1，固定 validation IDs，不读取测试集。",
        "- 候选族：从训练折相似历史对应的真实未来构成条件经验分布，再按概率质量量化为 20 条候选。",
        "- 分配原则：weighted Lloyd 最小化条件分布的量化误差；没有候选间排斥项、固定方向模板或硬配额。",
        "- 评分：每个候选的 cluster probability mass；最高分与候选生成来自同一分布。",
        f"- 校准锁定参数：`{json.dumps(selected_parameters, ensure_ascii=False)}`。",
        f"- 判定：**{verdict[0]}** — {verdict[1]}", "",
        "## Validation 指标", "",
        "| Family | Subset | N | J | minADE | minFDE | Selected ADE | Regret | Spread | Support@1m | R@1 | R@3 | Duplicate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ("c00", "ceq1"):
        for subset in ("all", "high_motion", "non_high_motion"):
            item = summaries[family][subset]
            lines.append(
                f"| {family} | {subset} | {item['N']} | {fmt(item['j'])} | "
                f"{fmt(item['minade'])} | {fmt(item['minfde'])} | "
                f"{fmt(item['selected_ade'])} | {fmt(item['regret'])} | "
                f"{fmt(item['endpoint_spread'])} | {fmt(item['support_1p0'], 3)} | "
                f"{fmt(item['oracle_recall_1'], 3)} | {fmt(item['oracle_recall_3'], 3)} | "
                f"{fmt(item['duplicate_pair_rate'], 4)} |"
            )
    lines += ["", "## CEQ1 minus C00 paired bootstrap", "",
              "| Subset | Metric | Delta | Relative | 95% CI |",
              "|---|---|---:|---:|---:|"]
    for subset in ("all", "high_motion", "non_high_motion"):
        for metric in (
            "j", "minade", "minfde", "selected_ade", "regret",
            "endpoint_spread", "support_1p0", "oracle_recall_1",
            "oracle_recall_3", "duplicate_pair_rate",
        ):
            row = lookup(rows, subset, metric)
            lines.append(
                f"| {subset} | {metric} | {fmt(row['estimate'], 5)} | "
                f"{fmt(row['relative_effect_pct'], 2)}% | "
                f"[{fmt(row['ci95_low'], 5)}, {fmt(row['ci95_high'], 5)}] |"
            )
    lines += ["", "## Train-only calibration", "",
              "| K neighbours | Temperature | Objective | minADE | minFDE | Selected ADE |",
              "|---:|---:|---:|---:|---:|---:|"]
    for item in calibration:
        lines.append(
            f"| {item['neighbours']} | {item['temperature']} | {fmt(item['objective'], 6)} | "
            f"{fmt(item['minade'])} | {fmt(item['minfde'])} | {fmt(item['selected_ade'])} |"
        )
    lines += ["", "## 判定门槛", "", *[
        f"- `{key}`: {value}" for key, value in verdict[2].items()
    ], "", "## Provenance", "", "```json",
        json.dumps(provenance, indent=2, ensure_ascii=False), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    astra_root = args.astra_root.resolve()
    audit = preflight(astra_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = configure_astra_env(REPO, astra_root, device)
    trainer = load_module(
        "ceq1_trainer", REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    )
    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer.TrajectoryDataset(
        data_dir / "eth_data_train.npy", data_dir / "eth_num_train.npy", cfg
    )
    coordinates, counts = load_coordinates(dataset)
    features, futures = build_observation_features(
        coordinates, counts, dataset.past_frames, dataset.future_frames
    )
    train_ids, validation_ids = split_ids()
    fit_ids, calibration_ids = split_fit_calibration(
        train_ids, args.calibration_seed, args.calibration_count
    )
    if args.smoke:
        calibration_ids = calibration_ids[:64]
        fit_ids = fit_ids[:2048]
        validation_ids = validation_ids[:64]
        grid = parameter_grid((64,), (1.0,))
        output = HERE / "smoke_output"
    else:
        grid = parameter_grid()
        output = HERE / "results" / "ceq1_eth_seed1"
    output.mkdir(parents=True, exist_ok=True)

    standardizer = Standardizer.fit(features[fit_ids])
    fit_quantizer = ConditionalFutureQuantizer(
        standardizer.transform(features[fit_ids]), futures[fit_ids], fit_ids, device
    )
    calibration_rows: list[dict[str, Any]] = []
    for parameters in grid:
        print(f"[CALIBRATE] {parameters}")
        candidates, logits = fit_quantizer.generate(
            standardizer.transform(features[calibration_ids]), calibration_ids,
            neighbours=int(parameters["neighbours"]),
            temperature=float(parameters["temperature"]),
            batch_size=args.batch_size,
        )
        values = derive(raw_metrics(calibration_ids, candidates, futures[calibration_ids], logits))
        calibration_rows.append({
            **parameters,
            "objective": calibration_objective(values),
            "minade": float(values["minade"].mean()),
            "minfde": float(values["minfde"].mean()),
            "selected_ade": float(values["selected_ade"].mean()),
        })
    selected = min(calibration_rows, key=lambda row: (row["objective"], row["neighbours"], row["temperature"]))
    selected_parameters = {
        "neighbours": int(selected["neighbours"]),
        "temperature": float(selected["temperature"]),
        "components": 20, "lloyd_iterations": 8,
    }
    print(f"[LOCKED] {selected_parameters}")

    final_standardizer = Standardizer.fit(features[train_ids])
    quantizer = ConditionalFutureQuantizer(
        final_standardizer.transform(features[train_ids]), futures[train_ids], train_ids, device
    )
    ceq_candidates, ceq_logits = quantizer.generate(
        final_standardizer.transform(features[validation_ids]), validation_ids,
        neighbours=selected_parameters["neighbours"],
        temperature=selected_parameters["temperature"],
        batch_size=args.batch_size,
    )
    ceq_raw = raw_metrics(validation_ids, ceq_candidates, futures[validation_ids], ceq_logits)

    validation_loader = DataLoader(
        Subset(dataset, validation_ids.tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )
    model = build_f16_model(REPO, "eth", 1, cfg, device)
    baseline_raw = evaluate_baseline(model, validation_loader, validation_ids, trainer)
    del model
    if not np.array_equal(baseline_raw["sample_id"], ceq_raw["sample_id"]):
        raise RuntimeError("paired sample identity failure")
    baseline_values, ceq_values = derive(baseline_raw), derive(ceq_raw)

    train_motion = observation_motion_ratio(coordinates[train_ids], dataset.past_frames)
    validation_motion = observation_motion_ratio(
        coordinates[validation_ids], dataset.past_frames
    )
    motion_threshold = float(np.quantile(train_motion, 0.75))
    high = validation_motion >= motion_threshold
    masks = {
        "all": np.ones(len(validation_ids), dtype=bool),
        "high_motion": high,
        "non_high_motion": ~high,
    }
    summaries = {
        "c00": {name: summary(baseline_values, mask) for name, mask in masks.items()},
        "ceq1": {name: summary(ceq_values, mask) for name, mask in masks.items()},
    }
    paired = paired_rows(
        baseline_values, ceq_values, masks,
        repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed,
    )
    verdict = classify(paired, summaries["ceq1"])
    if args.smoke:
        verdict = (
            "SMOKE_ONLY_NO_SCIENTIFIC_VERDICT",
            "This 64-sample execution checks the pipeline only; run the fixed full validation protocol for a scientific decision.",
            verdict[2],
        )
    provenance = {
        **audit,
        "smoke": bool(args.smoke),
        "device": str(device),
        "feature_dimension": int(features.shape[1]),
        "calibration": {
            "fit_count": int(len(fit_ids)),
            "calibration_count": int(len(calibration_ids)),
            "seed": args.calibration_seed,
            "objective": "mean(J) + 0.25 * mean(selected_ADE)",
            "selected": selected_parameters,
        },
        "validation_count_evaluated": int(len(validation_ids)),
        "motion_threshold_train_q75": motion_threshold,
        "verdict": {"code": verdict[0], "explanation": verdict[1], "gates": verdict[2]},
    }
    np.savez_compressed(output / "baseline_validation_cache.npz", **baseline_raw)
    np.savez_compressed(output / "ceq1_validation_cache.npz", **ceq_raw)
    write_csv(output / "calibration_grid.csv", calibration_rows)
    write_csv(output / "paired_bootstrap.csv", paired)
    summary_rows = [
        {"family": family, "subset": subset, **values}
        for family, family_values in summaries.items()
        for subset, values in family_values.items()
    ]
    write_csv(output / "condition_summary.csv", summary_rows)
    (output / "verdict.json").write_text(
        json.dumps(provenance["verdict"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = output / "CEQ1_REPORT.md"
    render_report(
        report, summaries, paired, calibration_rows, selected_parameters,
        verdict, provenance,
    )
    print(f"[DONE] {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astra-root", type=Path, default=ASTRA_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-count", type=int, default=768)
    parser.add_argument("--calibration-seed", type=int, default=20260922)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260922)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    audit = preflight(args.astra_root.resolve())
    if args.preflight_only:
        output = HERE / "preflight_audit.json"
        output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        print(f"[PASS] {output}")
        return
    run(args)


if __name__ == "__main__":
    main()
