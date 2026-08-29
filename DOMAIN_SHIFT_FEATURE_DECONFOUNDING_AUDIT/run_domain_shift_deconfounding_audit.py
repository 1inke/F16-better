"""DSA: zero-main-training domain shift and matched social-signal audit."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
TNPA_ROOT = WORKSPACE / "TAIL_NEED_PREDICTABILITY_AUDIT"
TNPA_AUDIT = TNPA_ROOT / "audit"
OUTPUT = ROOT / "audit"
SPLITS = ("general", "challenge")

INDIVIDUAL_NAMES = [
    "last_speed", "mean_speed", "speed_std", "mean_acceleration",
    "last_acceleration", "heading_change_sum", "mean_curvature",
    "observation_displacement", "observation_nonlinearity", "velocity_variance",
]
SOCIAL_NAMES = [
    "neighbour_count", "has_neighbour", "nearest_distance", "density_r2",
    "density_r4", "density_r6", "maximum_closing", "mean_closing",
    "minimum_ttc", "minimum_dca", "sparse_edge_count", "social_delta_norm",
    "fused_feature_norm",
]


def load_cache(split: str, partition: str, seed: int):
    path = TNPA_AUDIT / "cache" / f"{split}_{partition}_seed_{seed}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing TNPA cache: {path}")
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def representation_names(width: int):
    if width % 3:
        raise ValueError(f"Unexpected representation width: {width}")
    dim = width // 3
    return (
        [f"encoder_individual_{i}" for i in range(dim)]
        + [f"encoder_social_delta_{i}" for i in range(dim)]
        + [f"encoder_fused_{i}" for i in range(dim)]
    )


def scene_features(train, validation):
    scenes = sorted(set(train["scene"].tolist()) | set(validation["scene"].tolist()))
    train_speed = train["individual"][:, 1]
    global_mean, global_std = float(train_speed.mean()), float(train_speed.std() + 1.0e-6)
    stats = {}
    for scene in scenes:
        value = train_speed[train["scene"] == scene]
        stats[scene] = (
            float(value.mean()) if len(value) else global_mean,
            float(value.std() + 1.0e-6) if len(value) else global_std,
        )

    def build(values):
        one_hot = np.stack([(values["scene"] == scene) for scene in scenes], axis=1).astype(np.float32)
        normalized = np.asarray([
            (speed - stats[int(scene)][0]) / stats[int(scene)][1]
            for speed, scene in zip(values["individual"][:, 1], values["scene"])
        ], dtype=np.float32)[:, None]
        return np.concatenate((one_hot, normalized), axis=1)

    return build(train), build(validation), [f"scene_{scene}" for scene in scenes] + ["scene_normalized_speed"]


def base_feature_sets(train, validation):
    train_scene, val_scene, scene_names = scene_features(train, validation)
    train_scene_only, val_scene_only = train_scene[:, :-1], val_scene[:, :-1]
    scene_only_names = scene_names[:-1]
    rep_names = representation_names(train["representation"].shape[1])
    return {
        "P0_scene_only": (train_scene_only, val_scene_only, scene_only_names),
        "P1_individual": (train["individual"], validation["individual"], INDIVIDUAL_NAMES),
        "P2_social": (train["social"], validation["social"], SOCIAL_NAMES),
        "P3_individual_social": (
            np.concatenate((train["individual"], train["social"]), axis=1),
            np.concatenate((validation["individual"], validation["social"]), axis=1),
            INDIVIDUAL_NAMES + SOCIAL_NAMES,
        ),
        "P4_encoder_representation": (train["representation"], validation["representation"], rep_names),
        "P5_individual_social_scene": (
            np.concatenate((train["individual"], train["social"], train_scene), axis=1),
            np.concatenate((validation["individual"], validation["social"], val_scene), axis=1),
            INDIVIDUAL_NAMES + SOCIAL_NAMES + scene_names,
        ),
        "P6_all": (
            np.concatenate((train["individual"], train["social"], train_scene, train["representation"]), axis=1),
            np.concatenate((validation["individual"], validation["social"], val_scene, validation["representation"]), axis=1),
            INDIVIDUAL_NAMES + SOCIAL_NAMES + scene_names + rep_names,
        ),
    }


def prepare_sets(train, validation):
    sets = base_feature_sets(train, validation)
    return {
        "D0_scene_only": sets["P0_scene_only"],
        "D1_individual": sets["P1_individual"],
        "D2_social": sets["P2_social"],
        "D3_individual_social": sets["P3_individual_social"],
        "D4_encoder_representation": sets["P4_encoder_representation"],
        "D5_all_observable": sets["P6_all"],
    }, sets


def ece(labels, probability, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if mask.any():
            value += mask.mean() * abs(probability[mask].mean() - labels[mask].mean())
    return float(value)


def bootstrap_metric(labels, probability, metric, samples, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        index = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[index]).size == 2:
            values.append(metric(labels[index], probability[index]))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if values else None


def domain_classification(split, domain_sets, bootstrap, seed):
    rng = np.random.default_rng(seed + 17)
    result = {}
    for name, (train_x, val_x, feature_names) in domain_sets.items():
        train_index = rng.choice(len(train_x), min(len(train_x), len(val_x)), replace=False)
        x = np.concatenate((train_x[train_index], val_x), axis=0)
        y = np.concatenate((np.zeros(len(train_index), dtype=np.int64), np.ones(len(val_x), dtype=np.int64)))
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2500, solver="lbfgs", random_state=seed),
        )
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probability = cross_val_predict(estimator, x, y, cv=folds, method="predict_proba", n_jobs=1)[:, 1]
        fitted = clone(estimator).fit(x, y)
        coefficient = fitted[-1].coef_[0]
        order = np.argsort(-np.abs(coefficient))[:15]
        result[name] = {
            "balanced_samples": int(len(y)),
            "oof_auroc": float(roc_auc_score(y, probability)),
            "oof_auroc_ci95": bootstrap_metric(y, probability, roc_auc_score, bootstrap, seed + len(name)),
            "oof_auprc": float(average_precision_score(y, probability)),
            "brier": float(brier_score_loss(y, probability)),
            "ece_10bin": ece(y, probability),
            "top_standardized_coefficients": [
                {"feature": feature_names[index], "coefficient": float(coefficient[index])}
                for index in order
            ],
        }
        print(f"[domain] {split}/{name} auroc={result[name]['oof_auroc']:.3f}", flush=True)
    return result


def pooled_smd(train, validation):
    denominator = np.sqrt((np.var(train) + np.var(validation)) / 2.0 + 1.0e-12)
    return float((np.mean(validation) - np.mean(train)) / denominator)


def feature_shift(train, validation):
    scalar = {}
    for prefix, names in (("individual", INDIVIDUAL_NAMES), ("social", SOCIAL_NAMES)):
        for index, name in enumerate(names):
            scalar[f"{prefix}.{name}"] = (train[prefix][:, index], validation[prefix][:, index])
    dim = train["representation"].shape[1] // 3
    for index, name in enumerate(("encoder_individual_norm", "encoder_social_delta_norm", "encoder_fused_norm")):
        start = index * dim
        scalar[name] = (
            np.linalg.norm(train["representation"][:, start:start + dim], axis=1),
            np.linalg.norm(validation["representation"][:, start:start + dim], axis=1),
        )
    rows = []
    for name, (source, target) in scalar.items():
        rows.append({
            "feature": name,
            "train_mean": float(source.mean()), "train_median": float(np.median(source)),
            "validation_mean": float(target.mean()), "validation_median": float(np.median(target)),
            "standardized_mean_difference": pooled_smd(source, target),
            "absolute_smd": abs(pooled_smd(source, target)),
            "wasserstein": float(wasserstein_distance(source, target)),
            "train_p10": float(np.quantile(source, .10)), "train_p90": float(np.quantile(source, .90)),
            "validation_p10": float(np.quantile(target, .10)), "validation_p90": float(np.quantile(target, .90)),
        })
    return sorted(rows, key=lambda row: row["absolute_smd"], reverse=True)


def quantile_bin(train_values, validation_values, bins=4):
    edges = np.unique(np.quantile(train_values, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.digitize(train_values, edges), np.digitize(validation_values, edges), edges.tolist()


def matched_indices(train, validation, seed):
    speed_train, speed_val, speed_edges = quantile_bin(train["individual"][:, 1], validation["individual"][:, 1])
    curve_train, curve_val, curve_edges = quantile_bin(train["individual"][:, 6], validation["individual"][:, 6])
    density_train = np.clip(np.rint(train["social"][:, 4]), 0, 3).astype(np.int64)
    density_val = np.clip(np.rint(validation["social"][:, 4]), 0, 3).astype(np.int64)
    train_key = list(zip(train["scene"].astype(int), speed_train, curve_train, density_train))
    val_key = list(zip(validation["scene"].astype(int), speed_val, curve_val, density_val))
    train_groups, val_groups = defaultdict(list), defaultdict(list)
    for index, key in enumerate(train_key):
        train_groups[key].append(index)
    for index, key in enumerate(val_key):
        val_groups[key].append(index)
    rng = np.random.default_rng(seed + 101)
    selected_train, selected_val = [], []
    cell_counts = []
    for key in sorted(val_groups):
        if key not in train_groups:
            continue
        count = min(len(train_groups[key]), len(val_groups[key]))
        if count == 0:
            continue
        selected_train.extend(rng.choice(train_groups[key], count, replace=False).tolist())
        selected_val.extend(rng.choice(val_groups[key], count, replace=False).tolist())
        cell_counts.append({"stratum": list(map(int, key)), "pairs": int(count)})
    return (
        np.asarray(selected_train, dtype=np.int64), np.asarray(selected_val, dtype=np.int64),
        {"speed_edges": speed_edges, "curvature_edges": curve_edges, "cells": cell_counts},
    )


def matching_balance(train, validation, train_index, val_index):
    confounders = {
        "mean_speed": (train["individual"][:, 1], validation["individual"][:, 1]),
        "mean_curvature": (train["individual"][:, 6], validation["individual"][:, 6]),
        "density_r4": (train["social"][:, 4], validation["social"][:, 4]),
        "neighbour_count": (train["social"][:, 0], validation["social"][:, 0]),
    }
    result = {}
    for name, (source, target) in confounders.items():
        result[name] = {
            "smd_before": pooled_smd(source, target),
            "smd_after": pooled_smd(source[train_index], target[val_index]),
        }
    return result


def paired_delta_ci(labels, first, second, samples, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        index = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[index]).size < 2:
            continue
        values.append(roc_auc_score(labels[index], second[index]) - roc_auc_score(labels[index], first[index]))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if values else None


def matched_tail_probe(split, train, validation, sets, bootstrap, seed):
    train_index, val_index, strata = matched_indices(train, validation, seed)
    if len(train_index) < 300:
        raise RuntimeError(f"Insufficient common support for {split}: {len(train_index)} pairs")
    mapping = {
        "M1_individual": sets["P1_individual"],
        "M2_individual_social": sets["P3_individual_social"],
        "M3_individual_social_scene": sets["P5_individual_social_scene"],
    }
    result = {
        "matched_pairs": int(len(train_index)),
        "validation_retained_fraction": float(len(val_index) / len(validation["scene"])),
        "strata": strata,
        "balance": matching_balance(train, validation, train_index, val_index),
        "labels": {},
    }
    for label in ("ade", "fde", "union"):
        y_train = train[f"label_{label}"][train_index]
        y_val = validation[f"label_{label}"][val_index]
        models, probabilities = {}, {}
        for name, (x_train, x_val, _feature_names) in mapping.items():
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2500, solver="lbfgs", random_state=seed),
            ).fit(x_train[train_index], y_train)
            probability = estimator.predict_proba(x_val[val_index])[:, 1]
            probabilities[name] = probability
            models[name] = {
                "auroc": float(roc_auc_score(y_val, probability)),
                "auprc": float(average_precision_score(y_val, probability)),
            }
        base = probabilities["M1_individual"]
        for name in ("M2_individual_social", "M3_individual_social_scene"):
            models[name]["delta_auroc_vs_M1"] = models[name]["auroc"] - models["M1_individual"]["auroc"]
            models[name]["delta_auroc_ci95"] = paired_delta_ci(y_val, base, probabilities[name], bootstrap, seed + len(label) + len(name))
            models[name]["delta_auprc_vs_M1"] = models[name]["auprc"] - models["M1_individual"]["auprc"]
        result["labels"][label] = {
            "train_prevalence": float(y_train.mean()),
            "validation_prevalence": float(y_val.mean()),
            "models": models,
        }
    print(f"[matched] {split} pairs={len(train_index)} delta={result['labels']['union']['models']['M2_individual_social']['delta_auroc_vs_M1']:.3f}", flush=True)
    return result


def encoder_robustness(split, domain, tnpa_summary):
    classification = tnpa_summary["classification"]
    tail = {
        "raw_P3": classification["P3_individual_social"]["models"]["union"]["logistic"]["auroc"],
        "raw_P5": classification["P5_individual_social_scene"]["models"]["union"]["logistic"]["auroc"],
        "encoder_P4": classification["P4_encoder_representation"]["models"]["union"]["logistic"]["auroc"],
        "all_P6": classification["P6_all"]["models"]["union"]["logistic"]["auroc"],
    }
    domain_auc = {
        "raw_D3": domain["D3_individual_social"]["oof_auroc"],
        "encoder_D4": domain["D4_encoder_representation"]["oof_auroc"],
        "all_D5": domain["D5_all_observable"]["oof_auroc"],
    }
    robust = tail["encoder_P4"] >= tail["raw_P5"] and domain_auc["encoder_D4"] <= domain_auc["raw_D3"]
    return {"split": split, "tail_auroc": tail, "domain_auroc": domain_auc, "encoder_more_task_informative_and_domain_invariant": bool(robust)}


def decision(results):
    lookup = {item["split"]: item for item in results}
    deltas = {
        split: lookup[split]["matched_tail_probe"]["labels"]["union"]["models"]["M2_individual_social"]
        for split in SPLITS
    }
    social_pass = (
        deltas["challenge"]["delta_auroc_vs_M1"] >= 0.03
        and deltas["challenge"]["delta_auroc_ci95"][0] > 0.0
        and deltas["general"]["delta_auroc_vs_M1"] >= 0.0
    )
    challenge_domain = lookup["challenge"]["domain_classification"]
    individual_auc = challenge_domain["D1_individual"]["oof_auroc"]
    social_auc = challenge_domain["D2_social"]["oof_auroc"]
    scene_auc = challenge_domain["D0_scene_only"]["oof_auroc"]
    if abs(individual_auc - social_auc) < 0.02:
        strongest = ("mixed_motion_social", max(individual_auc, social_auc))
    else:
        strongest = max(("individual", individual_auc), ("social", social_auc), key=lambda row: row[1])
    motion_pass = individual_auc >= 0.70 and individual_auc >= social_auc + 0.03 and not social_pass
    scene_shortcut = scene_auc >= 0.70 and scene_auc >= max(individual_auc, social_auc) - 0.03
    if social_pass:
        recommendation = "DRS1_domain_robust_social_representation"
    elif motion_pass:
        recommendation = "DRM1_dual_scale_motion_representation"
    elif scene_shortcut:
        recommendation = "scene_balancing_and_normalization"
    else:
        recommendation = "freeze_tail_specific_modeling_and_focus_general_encoder_robustness"
    return {
        "social_branch_gate": {
            "rule": "Challenge matched M2-M1 AUROC>=0.03, CI lower>0, General delta>=0",
            "passed": bool(social_pass), "deltas": deltas,
        },
        "strongest_challenge_shift_source": {"group": strongest[0], "domain_auroc": strongest[1]},
        "motion_branch_gate_passed": bool(motion_pass),
        "scene_shortcut_detected": bool(scene_shortcut),
        "recommended_next_experiment": recommendation,
    }


def write_shift_csv(path, results):
    rows = []
    for item in results:
        for row in item["feature_shift"]:
            rows.append({"split": item["split"], **row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    tnpa_combined = json.loads((TNPA_AUDIT / "combined_summary.json").read_text(encoding="utf-8"))
    tnpa_lookup = {item["split"]: item for item in tnpa_combined["results"]}
    results = []
    for split in SPLITS:
        train, validation = load_cache(split, "train", args.seed), load_cache(split, "validation", args.seed)
        domain_sets, tnpa_sets = prepare_sets(train, validation)
        domain = domain_classification(split, domain_sets, args.bootstrap, args.seed)
        shift = feature_shift(train, validation)
        matched = matched_tail_probe(split, train, validation, tnpa_sets, args.bootstrap, args.seed)
        robustness = encoder_robustness(split, domain, tnpa_lookup[split])
        result = {
            "split": split, "domain_classification": domain,
            "feature_shift": shift, "matched_tail_probe": matched,
            "encoder_robustness": robustness,
        }
        (output / f"{split}_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(result)
    verdict = decision(results)
    combined = {
        "main_model_training_performed": False,
        "source": "TNPA frozen-B0 cache",
        "domain_task": "0=corresponding training fold, 1=corresponding locked validation fold",
        "matching": "exact common-support strata on scene, train-defined speed quartile, train-defined curvature quartile, and density_r4 bin",
        "results": results, "decision": verdict,
    }
    (output / "combined_summary.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    write_shift_csv(output / "feature_shift.csv", results)
    print(f"[done] recommendation={verdict['recommended_next_experiment']} output={output}", flush=True)


if __name__ == "__main__":
    main()
