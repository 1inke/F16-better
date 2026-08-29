"""TNPA: predict score-ranked tail need from information visible at inference time."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
TCA = WORKSPACE / "TAIL_CANDIDATE_FAMILY_AUDIT" / "run_tail_candidate_audit.py"
OUTPUT = ROOT / "audit"
SPLITS = ("general", "challenge")
LABEL_THRESHOLDS = {"ade": 0.05, "fde": 0.10}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_float(value) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def individual_features(past: torch.Tensor) -> tuple[np.ndarray, list[str]]:
    focal = past[:, 0]
    velocity = torch.diff(focal, dim=1)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    acceleration = torch.diff(velocity, dim=1)
    accel_norm = torch.linalg.vector_norm(acceleration, dim=-1)
    cross = velocity[:, :-1, 0] * velocity[:, 1:, 1] - velocity[:, :-1, 1] * velocity[:, 1:, 0]
    dot = (velocity[:, :-1] * velocity[:, 1:]).sum(dim=-1)
    turn = torch.atan2(cross, dot + 1.0e-8).abs()
    line_weight = torch.linspace(0, 1, focal.shape[1], device=focal.device, dtype=focal.dtype)[None, :, None]
    line = focal[:, :1] + line_weight * (focal[:, -1:] - focal[:, :1])
    values = torch.stack(
        (
            speed[:, -1], speed.mean(1), speed.std(1, unbiased=False),
            accel_norm.mean(1), accel_norm[:, -1], turn.sum(1), turn.mean(1),
            torch.linalg.vector_norm(focal[:, -1] - focal[:, 0], dim=-1),
            torch.linalg.vector_norm(focal - line, dim=-1).mean(1),
            velocity.var(1, unbiased=False).sum(-1),
        ), dim=-1,
    )
    names = [
        "last_speed", "mean_speed", "speed_std", "mean_acceleration",
        "last_acceleration", "heading_change_sum", "mean_curvature",
        "observation_displacement", "observation_nonlinearity", "velocity_variance",
    ]
    return values.cpu().numpy().astype(np.float32), names


def social_features(
    past: torch.Tensor,
    num_valid: torch.Tensor,
    social_delta: torch.Tensor,
    fused: torch.Tensor,
    allowed: torch.Tensor,
) -> tuple[np.ndarray, list[str]]:
    batch, agents = past.shape[:2]
    focal_pos = past[:, :1, -1]
    focal_vel = past[:, :1, -1] - past[:, :1, -2]
    relative = past[:, 1:, -1] - focal_pos
    neighbour_vel = past[:, 1:, -1] - past[:, 1:, -2]
    relative_velocity = neighbour_vel - focal_vel
    valid = torch.arange(1, agents, device=past.device)[None] < num_valid[:, None]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    distance_safe = distance.clamp_min(1.0e-5)
    closing = -(relative * relative_velocity).sum(-1) / distance_safe
    velocity_sq = relative_velocity.square().sum(-1).clamp_min(1.0e-6)
    ttc = -(relative * relative_velocity).sum(-1) / velocity_sq
    valid_ttc = valid & (ttc > 0)
    ttc_clamped = torch.where(valid_ttc, ttc.clamp_max(12.0), torch.full_like(ttc, 12.0))
    closest = relative + ttc_clamped[..., None] * relative_velocity
    dca = torch.linalg.vector_norm(closest, dim=-1)
    inf = torch.full_like(distance, float("inf"))
    nearest = torch.where(valid, distance, inf).min(1).values
    nearest = torch.where(torch.isfinite(nearest), nearest, torch.zeros_like(nearest))
    max_closing = torch.where(valid, closing, torch.full_like(closing, -1.0e4)).max(1).values
    max_closing = torch.where(valid.any(1), max_closing, torch.zeros_like(max_closing))
    mean_closing = torch.where(valid, closing, torch.zeros_like(closing)).sum(1) / valid.sum(1).clamp_min(1)
    min_ttc = ttc_clamped.min(1).values
    min_dca = torch.where(valid_ttc, dca, inf).min(1).values
    min_dca = torch.where(torch.isfinite(min_dca), min_dca, torch.zeros_like(min_dca))
    focal_edges = allowed[:, 0].clone()
    focal_edges[:, 0] = False
    values = torch.stack(
        (
            valid.sum(1).float(), valid.any(1).float(), nearest,
            (valid & (distance < 2.0)).sum(1).float(),
            (valid & (distance < 4.0)).sum(1).float(),
            (valid & (distance < 6.0)).sum(1).float(),
            max_closing, mean_closing, min_ttc, min_dca,
            focal_edges.sum(1).float(),
            torch.linalg.vector_norm(social_delta, dim=-1),
            torch.linalg.vector_norm(fused, dim=-1),
        ), dim=-1,
    )
    names = [
        "neighbour_count", "has_neighbour", "nearest_distance", "density_r2",
        "density_r4", "density_r6", "maximum_closing", "mean_closing",
        "minimum_ttc", "minimum_dca", "sparse_edge_count", "social_delta_norm",
        "fused_feature_norm",
    ]
    return values.cpu().numpy().astype(np.float32), names


@torch.inference_mode()
def extract(model, loader, device, tca):
    from train_front_variant import normalize_to_focal_heading

    keys = ("index", "scene", "prediction", "target", "score", "individual", "social", "representation")
    parts = {key: [] for key in keys}
    individual_names = social_names = None
    captured = {}

    def temporal_query_hook(_module, _inputs, output):
        captured["individual"] = output[1].detach()

    encoder = model.agent_aware_trans_encoder
    hook = encoder.temporal_queries.register_forward_hook(temporal_query_hook)
    try:
        for past, future, num_valid, indices, scene_ids in loader:
            past = past.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            num_valid = num_valid.to(device, non_blocking=True).int()
            past, future = normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            prediction = model(
                past - center, None, None, mode="testing",
                num_valid=num_valid, graph_loc=past,
            )[2] + center.unsqueeze(-3)
            if "individual" not in captured:
                raise RuntimeError("Temporal-query representation hook did not fire")
            individual_rep = captured["individual"][:, 0]
            fused_rep = encoder.last_agent_features[:, 0]
            social_delta = fused_rep - individual_rep
            indiv, individual_names = individual_features(past)
            social, social_names = social_features(
                past, num_valid, social_delta, fused_rep, encoder.last_allowed,
            )
            representation = torch.cat((individual_rep, social_delta, fused_rep), dim=-1)
            values = {
                "index": indices.numpy(), "scene": scene_ids.numpy(),
                "prediction": prediction[:, 0].cpu().numpy(),
                "target": future[:, 0].cpu().numpy(),
                "score": model.last_mode_logits[:, 0].cpu().numpy(),
                "individual": indiv, "social": social,
                "representation": representation.cpu().numpy().astype(np.float32),
            }
            for key, value in values.items():
                parts[key].append(value)
    finally:
        hook.remove()
    result = {key: np.concatenate(value) for key, value in parts.items()}
    tail = tca.tail_arrays(result)
    result.update({"gain_ade": tail["gain_ade"], "gain_fde": tail["gain_fde"]})
    result["label_ade"] = result["gain_ade"] >= LABEL_THRESHOLDS["ade"]
    result["label_fde"] = result["gain_fde"] >= LABEL_THRESHOLDS["fde"]
    result["label_union"] = result["label_ade"] | result["label_fde"]
    return result, individual_names, social_names


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

    names = [f"scene_{scene}" for scene in scenes] + ["scene_normalized_speed"]
    return build(train), build(validation), names


def feature_sets(train, validation, individual_names, social_names):
    train_scene, val_scene, scene_names = scene_features(train, validation)
    rep_names = (
        [f"encoder_individual_{i}" for i in range(train["representation"].shape[1] // 3)]
        + [f"encoder_social_delta_{i}" for i in range(train["representation"].shape[1] // 3)]
        + [f"encoder_fused_{i}" for i in range(train["representation"].shape[1] // 3)]
    )
    return {
        "P0_scene_only": (train_scene, val_scene, scene_names),
        "P1_individual": (train["individual"], validation["individual"], individual_names),
        "P2_social": (train["social"], validation["social"], social_names),
        "P3_individual_social": (
            np.concatenate((train["individual"], train["social"]), axis=1),
            np.concatenate((validation["individual"], validation["social"]), axis=1),
            individual_names + social_names,
        ),
        "P4_encoder_representation": (train["representation"], validation["representation"], rep_names),
        "P5_individual_social_scene": (
            np.concatenate((train["individual"], train["social"], train_scene), axis=1),
            np.concatenate((validation["individual"], validation["social"], val_scene), axis=1),
            individual_names + social_names + scene_names,
        ),
        "P6_all": (
            np.concatenate((train["individual"], train["social"], train_scene, train["representation"]), axis=1),
            np.concatenate((validation["individual"], validation["social"], val_scene, validation["representation"]), axis=1),
            individual_names + social_names + scene_names + rep_names,
        ),
    }


def expected_calibration_error(labels, probability, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if mask.any():
            result += mask.mean() * abs(probability[mask].mean() - labels[mask].mean())
    return float(result)


def bootstrap_ci(labels, probability, metric, samples, seed):
    if samples <= 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        index = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[index]).size < 2:
            continue
        values.append(metric(labels[index], probability[index]))
    if not values:
        return None
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def classification_metrics(train_labels, val_labels, train_probability, val_probability, bootstrap, seed):
    prevalence = float(train_labels.mean())
    threshold = float(np.quantile(train_probability, 1.0 - prevalence))
    predicted = val_probability >= threshold
    tn, fp, fn, tp = confusion_matrix(val_labels, predicted, labels=[False, True]).ravel()
    auroc = float(roc_auc_score(val_labels, val_probability))
    auprc = float(average_precision_score(val_labels, val_probability))
    budgets = {}
    for budget in (0.10, 0.20, 0.35):
        count = max(1, int(round(len(val_probability) * budget)))
        selected = np.argsort(-val_probability)[:count]
        budgets[f"top_{int(100 * budget)}pct"] = {
            "positive_recall": float(val_labels[selected].sum() / max(val_labels.sum(), 1)),
            "precision": float(val_labels[selected].mean()),
        }
    return {
        "train_prevalence": prevalence,
        "validation_prevalence": float(val_labels.mean()),
        "auroc": auroc,
        "auroc_ci95": bootstrap_ci(val_labels, val_probability, roc_auc_score, bootstrap, seed),
        "auprc": auprc,
        "auprc_ci95": bootstrap_ci(val_labels, val_probability, average_precision_score, bootstrap, seed + 1),
        "auprc_lift_over_prevalence": float(auprc / max(val_labels.mean(), 1.0e-8)),
        "train_prevalence_matched_threshold": threshold,
        "positive_recall": float(tp / max(tp + fn, 1)),
        "negative_recall": float(tn / max(tn + fp, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "brier": float(brier_score_loss(val_labels, val_probability)),
        "ece_10bin": expected_calibration_error(val_labels, val_probability),
        "risk_budgets": budgets,
    }


def classifier_factories(seed):
    return {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
        ),
        "mlp32": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32,), activation="relu", alpha=1.0e-3,
                batch_size=256, learning_rate_init=1.0e-3, max_iter=150,
                early_stopping=True, validation_fraction=0.10, n_iter_no_change=10,
                random_state=seed,
            ),
        ),
    }


def evaluate_probes(split, train, validation, sets, bootstrap, seed):
    result = {"split": split, "classification": {}, "continuous": {}}
    for feature_name, (x_train, x_val, names) in sets.items():
        result["classification"][feature_name] = {"feature_count": len(names), "models": {}}
        for label_name in ("ade", "fde", "union"):
            y_train = train[f"label_{label_name}"]
            y_val = validation[f"label_{label_name}"]
            label_results = {}
            for model_name, estimator in classifier_factories(seed).items():
                model = clone(estimator).fit(x_train, y_train)
                train_probability = model.predict_proba(x_train)[:, 1]
                val_probability = model.predict_proba(x_val)[:, 1]
                label_results[model_name] = classification_metrics(
                    y_train, y_val, train_probability, val_probability,
                    bootstrap, seed + len(feature_name) + len(label_name) + len(model_name),
                )
            result["classification"][feature_name]["models"][label_name] = label_results
        continuous = {}
        for target in ("gain_ade", "gain_fde"):
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(x_train, train[target])
            prediction = model.predict(x_val)
            continuous[target] = {
                "spearman": safe_float(spearmanr(validation[target], prediction).statistic),
                "pearson": safe_float(pearsonr(validation[target], prediction).statistic),
                "mae": float(mean_absolute_error(validation[target], prediction)),
                "zero_baseline_mae": float(mean_absolute_error(validation[target], np.zeros_like(prediction))),
            }
        result["continuous"][feature_name] = continuous
        print(f"[probe] {split}/{feature_name}", flush=True)
    return result


def flatten_rows(results):
    rows = []
    for split_result in results:
        split = split_result["split"]
        for feature, feature_result in split_result["classification"].items():
            for label, models in feature_result["models"].items():
                for model, metrics in models.items():
                    rows.append({
                        "split": split, "feature_set": feature, "label": label, "model": model,
                        **{key: value for key, value in metrics.items() if not isinstance(value, (dict, list))},
                    })
    return rows


def decision(results):
    lookup = {item["split"]: item for item in results}
    primary = {}
    for split in SPLITS:
        options = []
        for feature in ("P1_individual", "P3_individual_social", "P5_individual_social_scene", "P6_all"):
            for model in ("logistic", "mlp32"):
                metrics = lookup[split]["classification"][feature]["models"]["union"][model]
                correlation = max(
                    lookup[split]["continuous"][feature]["gain_ade"]["spearman"],
                    lookup[split]["continuous"][feature]["gain_fde"]["spearman"],
                )
                options.append((metrics["auroc"] + 0.1 * metrics["auprc_lift_over_prevalence"], feature, model, metrics, correlation))
        _, feature, model, metrics, correlation = max(options, key=lambda row: row[0])
        primary[split] = {"feature_set": feature, "model": model, **metrics, "maximum_gain_spearman": correlation}
    cat1_allowed = all(
        primary[split]["auroc"] >= 0.70
        and primary[split]["auprc_lift_over_prevalence"] >= 1.25
        and primary[split]["maximum_gain_spearman"] >= 0.15
        for split in SPLITS
    )
    increments = {}
    for split in SPLITS:
        cls = lookup[split]["classification"]
        p1 = cls["P1_individual"]["models"]["union"]["logistic"]
        p3 = cls["P3_individual_social"]["models"]["union"]["logistic"]
        p5 = cls["P5_individual_social_scene"]["models"]["union"]["logistic"]
        increments[split] = {
            "social_delta_auroc": p3["auroc"] - p1["auroc"],
            "social_delta_auprc": p3["auprc"] - p1["auprc"],
            "scene_delta_auroc": p5["auroc"] - p3["auroc"],
            "scene_delta_auprc": p5["auprc"] - p3["auprc"],
        }
    return {
        "primary": primary,
        "incremental_information": increments,
        "gate_rule": "Both splits: union AUROC>=0.70, AUPRC lift>=1.25, and gain Spearman>=0.15",
        "cat1_training_allowed": bool(cat1_allowed),
    }


def save_cache(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        index=values["index"], scene=values["scene"], individual=values["individual"],
        social=values["social"], representation=values["representation"],
        gain_ade=values["gain_ade"], gain_fde=values["gain_fde"],
        label_ade=values["label_ade"], label_fde=values["label_fde"], label_union=values["label_union"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    tca = load_module("tnpa_tca", TCA)
    diag = load_module("tnpa_cso_diag", tca.CSO / "run_cso1_diagnostic.py")
    geom = load_module("tnpa_geometry", tca.GEOM)
    cfg, model_class = diag.prepare(device)
    output = args.output.resolve()
    if (args.max_train is not None or args.max_val is not None) and args.output == OUTPUT:
        output = (ROOT / "smoke_output").resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for split in SPLITS:
        run = tca.source_run(split, args.seed)
        arrays = np.load(run / "split_indices.npz")
        train_indices = np.asarray(arrays["train"], dtype=np.int64)
        val_indices = np.asarray(arrays["validation"], dtype=np.int64)
        if args.max_train is not None:
            train_indices = train_indices[:args.max_train]
        if args.max_val is not None:
            val_indices = val_indices[:args.max_val]
        info = json.loads((run / "split.json").read_text(encoding="utf-8"))

        model = diag.build(split, "baseline", args.seed, cfg, model_class, device)
        values = []
        feature_names = None
        for partition, indices in (("train", train_indices), ("validation", val_indices)):
            dataset = geom.IndexedDataset(Path(info["data_path"]), Path(info["num_path"]), indices, cfg)
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                pin_memory=device.type == "cuda",
            )
            print(f"[infer] {split}/{partition} samples={len(dataset)}", flush=True)
            extracted, individual_names, social_names = extract(model, loader, device, tca)
            values.append(extracted)
            feature_names = (individual_names, social_names)
            save_cache(output / "cache" / f"{split}_{partition}_seed_{args.seed}.npz", extracted)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        train, validation = values
        sets = feature_sets(train, validation, *feature_names)
        split_result = evaluate_probes(split, train, validation, sets, args.bootstrap, args.seed)
        (output / f"{split}_summary.json").write_text(
            json.dumps(split_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        all_results.append(split_result)

    verdict = decision(all_results)
    combined = {
        "main_model_training_performed": False,
        "probe_training_only": True,
        "tail_definition": "score ranks 11-20",
        "labels": {
            "ade": "ADE@10-ADE@20 >= 0.05",
            "fde": "FDE@10-FDE@20 >= 0.10",
            "union": "ADE material OR FDE material",
        },
        "leakage_policy": "All features are available from observed history, scene id, or frozen encoder at inference time",
        "results": all_results,
        "decision": verdict,
    }
    (output / "combined_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = flatten_rows(all_results)
    with (output / "classification_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"[done] CAT1 allowed={verdict['cat1_training_allowed']} output={output}", flush=True)


if __name__ == "__main__":
    main()
