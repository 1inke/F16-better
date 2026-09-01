"""Aggregate EXP-173 H3/V2 results and make the preregistered seed-1 decision."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = ROOT / "EXP_173_E2_F16_END_TO_END"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
VARIANTS = {"e2": "f4_relation_tokens", "e3": "f16_temporal_social_cross"}
STAGES = ("h3", "v2")
METRICS = ("minADE20", "minFDE20", "same_path_FDE", "J", "top1_ADE", "top1_FDE")


def run_dir(experiment: Path, fold: str, stage: str, label: str, seed: int) -> Path:
    return experiment / "results" / fold / stage / label / VARIANTS[label] / f"seed_{seed}" / "val_scene_-1"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(experiment: Path, seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    for label, variant in VARIANTS.items():
        for fold in FOLDS:
            for seed in seeds:
                for stage in STAGES:
                    path = run_dir(experiment, fold, stage, label, seed) / "metrics.json"
                    if not path.is_file():
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    evaluation = payload.get("evaluation_test_best_selected")
                    if not evaluation:
                        continue
                    oracle, original = evaluation["oracle"], evaluation["original"]
                    counts = payload["locked_backend"]["parameter_counts"]
                    ade, fde = oracle["min_ade_20"], oracle["min_fde_20"]
                    rows.append({
                        "experiment": "EXP-173", "encoder_label": label,
                        "encoder_variant": variant, "fold": fold, "seed": seed,
                        "stage": stage, "samples": evaluation["samples"],
                        "minADE20": ade, "minFDE20": fde,
                        "same_path_FDE": oracle["same_path_fde"],
                        "J": ade + 0.5 * fde,
                        "top1_ADE": original["ade"], "top1_FDE": original["fde"],
                        "parameters_total": counts["total"],
                        "parameters_trainable": counts["trainable"],
                        "decoder_parameters": counts["decoder"],
                        "elapsed_seconds": payload["elapsed_seconds"],
                    })
    rows.sort(key=lambda x: (x["seed"], tuple(VARIANTS).index(x["encoder_label"]), FOLDS.index(x["fold"]), STAGES.index(x["stage"])))
    return rows


def mean_metrics(group: list[dict]) -> dict:
    return {key: float(np.mean([float(row[key]) for row in group])) for key in METRICS}


def summarize(experiment: Path, seeds: tuple[int, ...]) -> dict:
    rows = collect(experiment, seeds)
    write_csv(experiment / "all_runs.csv", rows)
    per_fold = []
    for seed in seeds:
        for label, variant in VARIANTS.items():
            for stage in STAGES:
                for fold in FOLDS:
                    group = [x for x in rows if x["seed"] == seed and x["encoder_label"] == label and x["stage"] == stage and x["fold"] == fold]
                    if group:
                        per_fold.append({"seed": seed, "encoder_label": label, "encoder_variant": variant, "stage": stage, "fold": fold, **mean_metrics(group), "samples": group[0]["samples"], "parameters_total": group[0]["parameters_total"], "parameters_trainable": group[0]["parameters_trainable"]})
    write_csv(experiment / "per_fold_summary.csv", per_fold)

    paired = []
    for seed in seeds:
        for stage in STAGES:
            for fold in FOLDS:
                pair = {x["encoder_label"]: x for x in per_fold if x["seed"] == seed and x["stage"] == stage and x["fold"] == fold}
                if set(pair) != set(VARIANTS):
                    continue
                delta = {key: pair["e2"][key] - pair["e3"][key] for key in METRICS}
                paired.append({"seed": seed, "stage": stage, "fold": fold, **{f"delta_E2_minus_F16_{key}": value for key, value in delta.items()}, "winner_J": "E2" if delta["J"] < 0 else ("F16" if delta["J"] > 0 else "TIE")})
    write_csv(experiment / "paired_comparison.csv", paired)

    attribution = []
    for seed in seeds:
        for label in VARIANTS:
            for fold in FOLDS:
                stage = {x["stage"]: x for x in per_fold if x["seed"] == seed and x["encoder_label"] == label and x["fold"] == fold}
                if set(stage) != set(STAGES):
                    continue
                attribution.append({"seed": seed, "encoder_label": label, "fold": fold, "H3_J": stage["h3"]["J"], "V2_J": stage["v2"]["J"], "V2_gain_pct": 100.0 * (stage["h3"]["J"] - stage["v2"]["J"]) / stage["h3"]["J"]})
    write_csv(experiment / "stage_attribution.csv", attribution)

    macro = {}
    weighted = {}
    for seed in seeds:
        macro[str(seed)] = {}
        weighted[str(seed)] = {}
        for label in VARIANTS:
            macro[str(seed)][label] = {}
            weighted[str(seed)][label] = {}
            for stage in STAGES:
                group = [x for x in per_fold if x["seed"] == seed and x["encoder_label"] == label and x["stage"] == stage]
                if len(group) != len(FOLDS):
                    continue
                macro[str(seed)][label][stage] = mean_metrics(group)
                total = sum(int(x["samples"]) for x in group)
                weighted[str(seed)][label][stage] = {key: sum(float(x[key]) * int(x["samples"]) for x in group) / total for key in METRICS}

    decision = None
    if seeds == (1,) and all("v2" in macro.get("1", {}).get(label, {}) for label in VARIANTS):
        e2, f16 = macro["1"]["e2"]["v2"], macro["1"]["e3"]["v2"]
        delta_j = e2["J"] - f16["J"]
        winner = "E2" if delta_j < 0 else ("F16" if delta_j > 0 else "TIE")
        loser_j = f16["J"] if winner == "E2" else e2["J"]
        relative = 100.0 * abs(delta_j) / loser_j if loser_j else 0.0
        v2_pairs = [x for x in paired if x["seed"] == 1 and x["stage"] == "v2"]
        wins_e2 = sum(x["winner_J"] == "E2" for x in v2_pairs)
        wins_f16 = sum(x["winner_J"] == "F16" for x in v2_pairs)
        if relative < 0.5:
            winner_seed1, next_action, category = "TIE", "RUN_EFFICIENCY", "CASE_C"
        elif relative < 1.0:
            winner_seed1, next_action, category = winner, "RUN_SEEDS_2_3", "CASE_D"
        elif (winner == "E2" and wins_e2 >= 4) or (winner == "F16" and wins_f16 >= 4):
            winner_seed1, next_action, category = winner, "RUN_SEEDS_2_3", "CASE_A" if winner == "E2" else "CASE_B"
        else:
            winner_seed1, next_action, category = "TIE", "RUN_SEEDS_2_3", "REVIEW_FOLD_INCONSISTENCY"
        decision = {"experiment": "EXP-173", "winner_seed1": winner_seed1, "macro_J_E2": e2["J"], "macro_J_F16": f16["J"], "relative_gap_pct": relative, "fold_wins_E2": wins_e2, "fold_wins_F16": wins_f16, "decision_case": category, "next_action": next_action}
    elif set(seeds) == {1, 2, 3} and all(
        "v2" in macro.get(str(seed), {}).get(label, {})
        for seed in seeds for label in VARIANTS
    ):
        e2_j = float(np.mean([macro[str(seed)]["e2"]["v2"]["J"] for seed in seeds]))
        f16_j = float(np.mean([macro[str(seed)]["e3"]["v2"]["J"] for seed in seeds]))
        delta_j = e2_j - f16_j
        winner = "E2" if delta_j < 0 else ("F16" if delta_j > 0 else "TIE")
        worse_j = f16_j if winner == "E2" else e2_j
        relative = 100.0 * abs(delta_j) / worse_j if worse_j else 0.0
        v2_pairs = [x for x in paired if x["stage"] == "v2"]
        wins_e2 = sum(x["winner_J"] == "E2" for x in v2_pairs)
        wins_f16 = sum(x["winner_J"] == "F16" for x in v2_pairs)
        fold_mean_delta = {
            fold: float(np.mean([
                x["delta_E2_minus_F16_J"] for x in v2_pairs if x["fold"] == fold
            ]))
            for fold in FOLDS
        }
        persistent_loss_folds = []
        for fold in FOLDS:
            fold_pairs = [x for x in v2_pairs if x["fold"] == fold]
            if winner == "F16" and all(x["winner_J"] == "E2" for x in fold_pairs):
                persistent_loss_folds.append(fold)
            elif winner == "E2" and all(x["winner_J"] == "F16" for x in fold_pairs):
                persistent_loss_folds.append(fold)
        winner_wins = wins_f16 if winner == "F16" else wins_e2
        if winner != "TIE" and relative >= 0.8 and winner_wins >= 10 and not persistent_loss_folds:
            category = f"THREE_SEED_{winner}_FINAL_CANDIDATE"
            next_action = "KEEP_F16_ETH_UCY_RETAIN_E2_FOR_SDD" if winner == "F16" else "KEEP_E2_ETH_UCY_RETAIN_F16_FOR_SDD"
        elif relative < 0.5:
            category, next_action = "THREE_SEED_ACCURACY_TIE", "RUN_EFFICIENCY"
        else:
            category, next_action = "THREE_SEED_INCONCLUSIVE", "REVIEW_PAIRED_FOLDS"
        paired_delta = np.asarray([x["delta_E2_minus_F16_J"] for x in v2_pairs], dtype=np.float64)
        decision = {
            "experiment": "EXP-173",
            "winner_three_seed": winner,
            "macro_J_E2": e2_j,
            "macro_J_F16": f16_j,
            "relative_gap_pct": relative,
            "paired_mean_delta_E2_minus_F16_J": float(paired_delta.mean()),
            "paired_median_delta_E2_minus_F16_J": float(np.median(paired_delta)),
            "paired_std_delta_E2_minus_F16_J": float(paired_delta.std(ddof=1)),
            "paired_wins_E2": wins_e2,
            "paired_wins_F16": wins_f16,
            "fold_mean_delta_E2_minus_F16_J": fold_mean_delta,
            "persistent_loss_folds": persistent_loss_folds,
            "decision_case": category,
            "next_action": next_action,
        }

    expected = len(seeds) * len(VARIANTS) * len(FOLDS) * len(STAGES)
    payload = {"experiment": "EXP-173 E2 vs F16 End-to-End Matched Comparison", "complete": len(rows) == expected, "expected_stage_runs": expected, "available_stage_runs": len(rows), "protocol": {"folds": list(FOLDS), "seeds": list(seeds), "variants": VARIANTS, "stages": list(STAGES), "macro": "five-fold equal weight", "decoder": "H3 then jointly trainable H3+V2"}, "macro": macro, "sample_weighted_diagnostic": weighted, "paired": paired, "stage_attribution": attribution, "decision": decision}
    experiment.mkdir(parents=True, exist_ok=True)
    (experiment / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    args = parser.parse_args()
    payload = summarize(args.experiment_dir.resolve(), tuple(args.seeds))
    print(json.dumps({"complete": payload["complete"], "available": payload["available_stage_runs"], "decision": payload["decision"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
