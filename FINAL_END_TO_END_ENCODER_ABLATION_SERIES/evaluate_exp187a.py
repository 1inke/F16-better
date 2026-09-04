"""Aggregate EXP-187A and matched EXP-173 reference results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "EXP_187A_END_TO_END_ENCODER_ABLATION"
EXP173 = ROOT.parent / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
STAGES = ("h3", "v2")
VARIANTS = {
    "e0": "f0_baseline",
    "e1": "f3_factorized",
    "e2": "f4_relation_tokens",
    "f16": "f16_temporal_social_cross",
}
REFERENCE_LABELS = {"e2": "e2", "f16": "e3"}
METRICS = ("minADE20", "minFDE20", "same_path_FDE", "J", "top1_ADE", "top1_FDE")


def own_run(experiment: Path, fold: str, stage: str, label: str, seed: int) -> Path:
    return experiment / "results" / fold / stage / label / VARIANTS[label] / f"seed_{seed}" / "val_scene_-1"


def reference_run(fold: str, stage: str, label: str, seed: int) -> Path:
    old = REFERENCE_LABELS[label]
    return EXP173 / "results" / fold / stage / old / VARIANTS[label] / f"seed_{seed}" / "val_scene_-1"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_row(path: Path, label: str, fold: str, stage: str, seed: int, source: str) -> dict | None:
    metrics_path = path / "metrics.json"
    if not metrics_path.is_file():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    evaluation = payload.get("evaluation_test_best_selected")
    if not evaluation:
        return None
    oracle, original = evaluation["oracle"], evaluation["original"]
    counts = payload["locked_backend"]["parameter_counts"]
    ade, fde = float(oracle["min_ade_20"]), float(oracle["min_fde_20"])
    return {
        "encoder": label, "variant": VARIANTS[label], "fold": fold, "seed": seed,
        "stage": stage, "source": source, "samples": int(evaluation["samples"]),
        "minADE20": ade, "minFDE20": fde,
        "same_path_FDE": float(oracle["same_path_fde"]), "J": ade + 0.5 * fde,
        "top1_ADE": float(original["ade"]), "top1_FDE": float(original["fde"]),
        "parameters_total": int(counts["total"]),
        "parameters_trainable": int(counts["trainable"]),
        "decoder_parameters": int(counts["decoder"]),
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
    }


def mean_metrics(rows: list[dict]) -> dict:
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in METRICS}


def summarize(experiment: Path = EXPERIMENT, seeds: tuple[int, ...] = (1,)) -> dict:
    rows: list[dict] = []
    for seed in seeds:
        for label in VARIANTS:
            for fold in FOLDS:
                for stage in STAGES:
                    if label in REFERENCE_LABELS:
                        path, source = reference_run(fold, stage, label, seed), "EXP-173 reference"
                    else:
                        path, source = own_run(experiment, fold, stage, label, seed), "EXP-187A trained"
                    row = metric_row(path, label, fold, stage, seed, source)
                    if row:
                        rows.append(row)
    rows.sort(key=lambda r: (r["seed"], tuple(VARIANTS).index(r["encoder"]), FOLDS.index(r["fold"]), STAGES.index(r["stage"])))
    write_csv(experiment / "all_runs.csv", rows)

    per_fold = []
    for seed in seeds:
        for label in VARIANTS:
            for stage in STAGES:
                for fold in FOLDS:
                    group = [r for r in rows if r["seed"] == seed and r["encoder"] == label and r["stage"] == stage and r["fold"] == fold]
                    if group:
                        per_fold.append({"seed": seed, "encoder": label, "variant": VARIANTS[label], "stage": stage, "fold": fold, **mean_metrics(group), "samples": group[0]["samples"], "parameters_total": group[0]["parameters_total"], "source": group[0]["source"]})
    write_csv(experiment / "per_fold.csv", per_fold)

    macro: dict[str, dict] = {}
    for seed in seeds:
        macro[str(seed)] = {}
        for label in VARIANTS:
            macro[str(seed)][label] = {}
            for stage in STAGES:
                group = [r for r in per_fold if r["seed"] == seed and r["encoder"] == label and r["stage"] == stage]
                if len(group) == len(FOLDS):
                    macro[str(seed)][label][stage] = {**mean_metrics(group), "parameters_total": group[0]["parameters_total"]}

    paired_rows, stage_rows = [], []
    for seed in seeds:
        for fold in FOLDS:
            pair = {r["encoder"]: r for r in per_fold if r["seed"] == seed and r["fold"] == fold and r["stage"] == "v2"}
            if "e0" in pair and "e1" in pair:
                gain = 100.0 * (pair["e0"]["J"] - pair["e1"]["J"]) / pair["e0"]["J"]
                paired_rows.append({"seed": seed, "fold": fold, "E0_J": pair["e0"]["J"], "E1_J": pair["e1"]["J"], "E1_vs_E0_J_gain_pct": gain, "winner": "E1" if gain > 0 else ("E0" if gain < 0 else "TIE")})
            for label in VARIANTS:
                stage = {r["stage"]: r for r in per_fold if r["seed"] == seed and r["fold"] == fold and r["encoder"] == label}
                if set(stage) == set(STAGES):
                    stage_rows.append({"seed": seed, "fold": fold, "encoder": label, "H3_J": stage["h3"]["J"], "V2_J": stage["v2"]["J"], "V2_gain_pct": 100.0 * (stage["h3"]["J"] - stage["v2"]["J"]) / stage["h3"]["J"]})
    write_csv(experiment / "paired_e0_e1.csv", paired_rows)
    write_csv(experiment / "stage_attribution.csv", stage_rows)

    decision = None
    if all("v2" in macro.get("1", {}).get(label, {}) for label in ("e0", "e1")):
        e0, e1 = macro["1"]["e0"]["v2"], macro["1"]["e1"]["v2"]
        gain = 100.0 * (e0["J"] - e1["J"]) / e0["J"]
        positive = sum(r["seed"] == 1 and r["winner"] == "E1" for r in paired_rows)
        if gain >= 2.0 and positive >= 4:
            verdict, next_action = "PASS_TO_MULTI_SEED", "RUN_E0_E1_SEEDS_2_3"
        elif gain < 1.0 or positive <= 2:
            verdict, next_action = "FLAG_ENCODER_ATTRIBUTION_RISK", "REVIEW_SPARSE_SOCIAL_CLAIM"
        else:
            verdict, next_action = "INCONCLUSIVE", "REVIEW_FOLD_STABILITY_BEFORE_MULTI_SEED"
        decision = {"seed": 1, "E0_macro_J": e0["J"], "E1_macro_J": e1["J"], "E1_vs_E0_J_gain_pct": gain, "positive_folds": positive, "verdict": verdict, "next_action": next_action}

    expected = len(seeds) * len(VARIANTS) * len(FOLDS) * len(STAGES)
    payload = {
        "experiment": "EXP-187A End-to-End Encoder Contribution Ablation",
        "complete": len(rows) == expected, "available_stage_runs": len(rows), "expected_stage_runs": expected,
        "protocol": {"seeds": list(seeds), "folds": list(FOLDS), "stages": list(STAGES), "variants": VARIANTS, "macro": "five-fold equal weight", "decoder": "matched H3 then fresh jointly trainable Joint-V2", "references": "E2/F16 from EXP-173 exact protocol"},
        "macro": macro, "decision": decision,
    }
    experiment.mkdir(parents=True, exist_ok=True)
    (experiment / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# EXP-187A End-to-End Encoder Ablation", "", "实验尚未完整。"]
    if decision:
        lines = ["# EXP-187A End-to-End Encoder Ablation", "", f"Seed1 判定：**{decision['verdict']}**", "", f"- E0 macro J: {decision['E0_macro_J']:.6f}", f"- E1 macro J: {decision['E1_macro_J']:.6f}", f"- E1 相对 E0: {decision['E1_vs_E0_J_gain_pct']:+.2f}%", f"- E1 胜出场景: {decision['positive_folds']}/5", f"- 下一步: {decision['next_action']}"]
    (experiment / "RESULT_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    args = parser.parse_args()
    result = summarize(args.experiment_dir.resolve(), tuple(args.seeds))
    print(json.dumps({"complete": result["complete"], "available": result["available_stage_runs"], "decision": result["decision"]}, ensure_ascii=False), flush=True)

