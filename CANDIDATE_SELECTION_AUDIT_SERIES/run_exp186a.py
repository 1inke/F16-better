"""EXP-186A frozen five-fold candidate selection feature and rule audit."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "EXP_186A_CANDIDATE_SELECTION_FEATURE_AUDIT"
CACHE = OUT / "cache"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "THREE_BRANCH_FOUNDATION_SERIES"))
import candidate_selection_features as feature_lib  # noqa: E402
import run_exp176a_final_score_audit as base  # noqa: E402


def cache_path(fold: str, split: str, kind: str) -> Path:
    return CACHE / fold / f"{split}_{kind}.npz"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dataset_loader(fold: str, split: str, cfg, trainer, batch_size: int, max_samples: int):
    dataset = trainer.TrajectoryDataset(
        base.DATA / f"{fold}_data_{split}.npy", base.DATA / f"{fold}_num_{split}.npy", cfg
    )
    if max_samples > 0 and len(dataset) > max_samples:
        indices = np.linspace(0, len(dataset) - 1, max_samples, dtype=np.int64).tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def collect_raw(model, loader, trainer, device: torch.device) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {}
    captured = {}

    def mode_hook(_module, _inputs, output):
        captured["mode_hidden"] = output.detach()

    handle = model.decoder.mode_mlp.register_forward_hook(mode_hook)

    def add(name: str, value: torch.Tensor, dtype=np.float32):
        parts.setdefault(name, []).append(value.detach().cpu().numpy().astype(dtype, copy=False))

    repeat_error = None
    with torch.inference_mode():
        for past_raw, future_raw, num_valid, *_ in loader:
            past_raw = past_raw.float().to(device)
            future_raw = future_raw.float().to(device)
            num_valid = num_valid.int().to(device)
            past, future = trainer.normalize_to_focal_heading(past_raw, future_raw)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            raw_prediction = output[2] if isinstance(output, (tuple, list)) else output
            prediction = (raw_prediction + center.unsqueeze(-3))[:, 0]
            if repeat_error is None:
                first = prediction.clone()
                second_output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
                second_raw = second_output[2] if isinstance(second_output, (tuple, list)) else second_output
                second = (second_raw + center.unsqueeze(-3))[:, 0]
                repeat_error = float((first - second).abs().max().cpu())
            logits = model.last_mode_logits[:, 0]
            source = model.decoder.last_anchor_proposals[:, 0] + center[:, 0, None]
            corrected = model.decoder.last_corrected_anchors[:, 0] + center[:, 0, None]
            mode_hidden = captured["mode_hidden"][:, 0]
            selected = model.decoder.last_selected_indices[:, 0]
            expected = torch.arange(20, device=device)[None].expand_as(selected)
            if not torch.equal(selected, expected):
                raise RuntimeError("Candidate identity/order changed inside frozen decoder")
            distance = torch.linalg.vector_norm(prediction - future[:, 0, None], dim=-1)
            add("past", past, np.float16)
            add("num_valid", num_valid, np.int8)
            add("prediction", prediction, np.float16)
            add("source_anchors", source, np.float16)
            add("corrected_anchors", corrected, np.float16)
            add("mode_hidden", mode_hidden, np.float16)
            add("logits", logits)
            add("ade", distance.mean(-1))
            add("fde", distance[..., -1])
    handle.remove()
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    arrays["repeat_max_abs"] = np.asarray([repeat_error or 0.0], np.float32)
    if arrays["prediction"].shape[1:] != (20, 12, 2):
        raise RuntimeError(f"candidate shape {arrays['prediction'].shape}")
    if arrays["source_anchors"].shape[1:] != (20, 3, 2):
        raise RuntimeError(f"anchor shape {arrays['source_anchors'].shape}")
    if arrays["mode_hidden"].shape[1:] != (20, 64):
        raise RuntimeError(f"mode hidden shape {arrays['mode_hidden'].shape}")
    return arrays


def extract_all(folds: tuple[str, ...], device: torch.device, batch_size: int, max_samples: int, force: bool) -> None:
    base.configure(device)
    trainer = base.load_module("exp186_trainer", base.TRAINER)
    models = base.load_module("exp186_models", base.MODEL_FILE)
    adapter = base.load_module("exp186_adapter", base.ADAPTER)
    for fold in folds:
        model, cfg = base.build(fold, 1, device, trainer, models, adapter)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        for split in ("train", "test"):
            path = cache_path(fold, split, "raw")
            if path.is_file() and not force:
                print(f"[raw] reuse {fold}/{split}", flush=True)
                continue
            loader = dataset_loader(fold, split, cfg, trainer, batch_size, max_samples)
            print(f"[raw] {fold}/{split}: {len(loader.dataset)} samples", flush=True)
            arrays = collect_raw(model, loader, trainer, device)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {name: raw[name] for name in raw.files}


def train_sigmas(raw: dict[str, np.ndarray], max_pairs: int = 200000) -> tuple[float, float]:
    modes = raw["prediction"].shape[1]
    pairs_per_sample = modes * (modes - 1) // 2
    count = min(len(raw["prediction"]), max(1, max_pairs // pairs_per_sample))
    ids = np.linspace(0, len(raw["prediction"]) - 1, count, dtype=np.int64)
    prediction = torch.from_numpy(raw["prediction"][ids].astype(np.float32))
    endpoint = torch.cdist(prediction[..., -1, :], prediction[..., -1, :])
    trajectory = torch.linalg.vector_norm(prediction[:, :, None] - prediction[:, None], dim=-1).mean(-1)
    tri = torch.triu_indices(modes, modes, offset=1)
    return float(endpoint[:, tri[0], tri[1]].median()), float(trajectory[:, tri[0], tri[1]].median())


def build_bank(raw: dict[str, np.ndarray], sigma_endpoint: float, sigma_trajectory: float,
               device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    pieces = {"features": [], "R1_medoid": [], "R2_weighted_MBR": [], "R3_H3_MBR": []}
    for start in range(0, len(raw["prediction"]), batch_size):
        sl = slice(start, min(start + batch_size, len(raw["prediction"])))
        values = [torch.from_numpy(raw[name][sl].astype(np.float32)).to(device) for name in ("prediction", "logits", "past", "num_valid", "source_anchors", "corrected_anchors")]
        prediction, logits, past, num_valid, source, corrected = values
        features, rules = feature_lib.candidate_feature_bank(
            prediction, logits, past, num_valid.int(), source, corrected,
            sigma_endpoint, sigma_trajectory,
        )
        pieces["features"].append(features.cpu().numpy().astype(np.float32))
        for name, score in rules.items():
            pieces[name].append(score.cpu().numpy().astype(np.float32))
    quality = raw["ade"] + 0.25 * raw["fde"]
    return {
        **{name: np.concatenate(values) for name, values in pieces.items()},
        "logits": raw["logits"], "ade": raw["ade"], "fde": raw["fde"],
        "quality": quality, "sigma_endpoint": np.asarray([sigma_endpoint]),
        "sigma_trajectory": np.asarray([sigma_trajectory]),
    }


def build_all_banks(folds: tuple[str, ...], device: torch.device, batch_size: int, force: bool) -> None:
    for fold in folds:
        train_raw = load_npz(cache_path(fold, "train", "raw"))
        sigma_endpoint, sigma_trajectory = train_sigmas(train_raw)
        print(f"[bank] {fold} train sigmas endpoint={sigma_endpoint:.4f} trajectory={sigma_trajectory:.4f}", flush=True)
        for split, raw in (("train", train_raw), ("test", load_npz(cache_path(fold, "test", "raw")))):
            path = cache_path(fold, split, "bank")
            if path.is_file() and not force:
                print(f"[bank] reuse {fold}/{split}", flush=True)
                continue
            bank = build_bank(raw, sigma_endpoint, sigma_trajectory, device, batch_size)
            np.savez_compressed(path, **bank, feature_names=np.asarray(feature_lib.FEATURES))


def within_z(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=1, keepdims=True)) / np.maximum(values.std(axis=1, keepdims=True), 1e-6)


def per_sample_spearman(score: np.ndarray, quality: np.ndarray) -> np.ndarray:
    a = rankdata(score, axis=1, method="average")
    b = rankdata(-quality, axis=1, method="average")
    a -= a.mean(1, keepdims=True)
    b -= b.mean(1, keepdims=True)
    denominator = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    result = np.full(len(score), np.nan)
    valid = denominator > 1e-12
    result[valid] = (a[valid] * b[valid]).sum(1) / denominator[valid]
    return result


def fit_score_control(train: dict[str, np.ndarray]):
    indices = [feature_lib.FEATURES.index("native_logit"), feature_lib.FEATURES.index("native_rank")]
    x = within_z(train["features"][:, :, indices]).reshape(-1, 2)
    y = (train["quality"] - train["quality"].mean(1, keepdims=True)).reshape(-1)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=1.0).fit(scaler.transform(x), y)
    return scaler, model


def predict_score_quality(bank: dict[str, np.ndarray], calibrator) -> np.ndarray:
    indices = [feature_lib.FEATURES.index("native_logit"), feature_lib.FEATURES.index("native_rank")]
    x = within_z(bank["features"][:, :, indices]).reshape(-1, 2)
    scaler, model = calibrator
    return model.predict(scaler.transform(x)).reshape(len(bank["quality"]), 20)


def correlation_rows(fold: str, bank: dict[str, np.ndarray], residual: np.ndarray | None, scope: str) -> list[dict]:
    rows = []
    target_quality = bank["quality"] if residual is None else residual
    for index, name in enumerate(feature_lib.FEATURES):
        rho = per_sample_spearman(bank["features"][..., index], target_quality)
        rows.append({
            "scope": scope, "fold": fold, "feature": name,
            "spearman": float(np.nanmean(rho)), "valid_fraction": float(np.isfinite(rho).mean()),
            "supplementary": bool(name in feature_lib.SUPPLEMENTARY),
        })
    return rows


def aggregate_feature_gate(rows: list[dict], folds: tuple[str, ...], exclude_score: bool = False) -> tuple[list[dict], list[str]]:
    summary, passed = [], []
    for name in feature_lib.FEATURES:
        values = {row["fold"]: row["spearman"] for row in rows if row["feature"] == name}
        macro = float(np.mean([values[fold] for fold in folds]))
        sign = 1 if macro >= 0 else -1
        same = sum(np.sign(values[fold]) == sign for fold in folds)
        eligible = name not in feature_lib.SUPPLEMENTARY and (not exclude_score or name not in feature_lib.SCORE_FEATURES)
        gate = bool(eligible and abs(macro) >= 0.15 and same >= max(1, min(4, len(folds))))
        row = {"feature": name, "macro_spearman": macro, "abs_macro_spearman": abs(macro), "same_sign_folds": same, "eligible": eligible, "passed": gate}
        row.update({f"{fold}_spearman": values[fold] for fold in folds})
        summary.append(row)
        if gate:
            passed.append(name)
    return summary, passed


def selection_metrics(bank: dict[str, np.ndarray], scores: np.ndarray) -> dict[str, float | int]:
    ade, fde, quality = bank["ade"], bank["fde"], bank["quality"]
    rows = np.arange(len(scores))
    selected = scores.argmax(1)
    best_q, best_ade, best_fde = quality.argmin(1), ade.argmin(1), fde.argmin(1)
    order = np.argsort(-scores, axis=1)
    regret = quality[rows, selected] - quality[rows, best_q]
    return {
        "samples": int(len(scores)),
        "top1_ADE": float(ade[rows, selected].mean()),
        "top1_FDE": float(fde[rows, selected].mean()),
        "top1_J": float((ade[rows, selected] + 0.5 * fde[rows, selected]).mean()),
        "combined_winner_hit": float((selected == best_q).mean()),
        "best_ADE_hit": float((selected == best_ade).mean()),
        "best_FDE_hit": float((selected == best_fde).mean()),
        "top3_winner_recovery": float((order[:, :3] == best_q[:, None]).any(1).mean()),
        "top5_winner_recovery": float((order[:, :5] == best_q[:, None]).any(1).mean()),
        "mean_regret": float(regret.mean()), "median_regret": float(np.median(regret)),
        "p90_regret": float(np.percentile(regret, 90)), "p95_regret": float(np.percentile(regret, 95)),
        "p99_regret": float(np.percentile(regret, 99)),
        "oracle_minADE": float(ade.min(1).mean()), "oracle_minFDE": float(fde.min(1).mean()),
    }


def ridge_scores(train: dict[str, np.ndarray], target: dict[str, np.ndarray], feature_names: list[str]) -> np.ndarray:
    indices = [feature_lib.FEATURES.index(name) for name in feature_names]
    x_train = within_z(train["features"][:, :, indices]).reshape(-1, len(indices))
    x_target = within_z(target["features"][:, :, indices]).reshape(-1, len(indices))
    y = (train["quality"] - train["quality"].mean(1, keepdims=True)).reshape(-1)
    # Inputs already use the preregistered within-sample normalization. Avoid a
    # second full-size standardized copy, which is unnecessary and can create a
    # large transient memory peak on the complete five-fold feature banks.
    model = Ridge(alpha=1.0, solver="lsqr", copy_X=False).fit(x_train, y)
    prediction = model.predict(x_target)
    del x_train, x_target
    return -prediction.reshape(len(target["quality"]), 20)


def build_rule_scores(train: dict[str, np.ndarray], target: dict[str, np.ndarray], pass_features: list[str], train_residual_rho: dict[str, float]) -> dict[str, np.ndarray]:
    result = {
        "R0_Native": target["logits"],
        "R1_Medoid": target["R1_medoid"],
        "R2_Weighted_MBR": target["R2_weighted_MBR"],
        "R3_H3_MBR": target["R3_H3_MBR"],
    }
    if pass_features:
        magnitude = sum(abs(train_residual_rho[name]) for name in pass_features)
        score = within_z(target["logits"])
        for name in pass_features:
            index = feature_lib.FEATURES.index(name)
            weight = 0.5 * train_residual_rho[name] / max(magnitude, 1e-12)
            score = score + weight * within_z(target["features"][..., index])
        result["R4_Stable_Feature"] = score
        result["R5_Ridge_supplementary"] = ridge_scores(train, target, ["native_logit", *pass_features])
    result["Linear_All25_upper_bound"] = ridge_scores(train, target, list(feature_lib.FEATURES))
    return result


def reproduction_sanity(fold: str, test: dict[str, np.ndarray]) -> dict:
    metrics_path = base.ckpt(fold, 1, "v2").parent / "metrics.json"
    official = json.loads(metrics_path.read_text(encoding="utf-8"))["evaluation_test_best_selected"]
    current = selection_metrics(test, test["logits"])
    differences = {
        "top1_ADE": abs(current["top1_ADE"] - official["original"]["ade"]),
        "top1_FDE": abs(current["top1_FDE"] - official["original"]["fde"]),
        "minADE20": abs(current["oracle_minADE"] - official["oracle"]["min_ade_20"]),
        "minFDE20": abs(current["oracle_minFDE"] - official["oracle"]["min_fde_20"]),
    }
    return {"fold": fold, **{f"abs_diff_{key}": value for key, value in differences.items()}, "passed_1e_5": bool(max(differences.values()) < 1e-5)}


def winner_analysis(folds_data: dict, residual_pass: list[str]) -> list[dict]:
    rows = []
    for name in feature_lib.FEATURES:
        index = feature_lib.FEATURES.index(name)
        winners, others, ranks, per_sample_better = [], [], [], []
        for data in folds_data.values():
            feature = data["test"]["features"][..., index]
            quality = data["test"]["quality"]
            winner = quality.argmin(1)
            sample = np.arange(len(winner))
            winners.append(feature[sample, winner])
            mask = np.ones_like(feature, dtype=bool); mask[sample, winner] = False
            others.append(feature[mask])
            nonwinner = feature[mask].reshape(len(feature), 19)
            per_sample_better.append((feature[sample, winner, None] > nonwinner).mean(1))
            order = np.argsort(-feature, axis=1)
            ranks.append((order == winner[:, None]).argmax(1) + 1)
        winner_values, other_values, rank = np.concatenate(winners), np.concatenate(others), np.concatenate(ranks)
        better_fraction = np.concatenate(per_sample_better)
        pooled_std = math.sqrt((winner_values.var() + other_values.var()) / 2 + 1e-12)
        rows.append({
            "feature": name, "residual_signal_pass": name in residual_pass,
            "winner_mean": float(winner_values.mean()), "winner_median": float(np.median(winner_values)),
            "nonwinner_mean": float(other_values.mean()), "effect_size": float((winner_values.mean() - other_values.mean()) / pooled_std),
            "winner_better_fraction": float(better_fraction.mean()),
            "winner_mean_rank": float(rank.mean()), "winner_median_rank": float(np.median(rank)),
            "hit1": float((rank <= 1).mean()), "hit3": float((rank <= 3).mean()), "hit5": float((rank <= 5).mean()),
        })
    return rows


def macro_rule_table(per_fold: list[dict], folds: tuple[str, ...]) -> list[dict]:
    rows = []
    for rule in sorted(set(row["rule"] for row in per_fold)):
        selected = [row for row in per_fold if row["rule"] == rule]
        result = {"rule": rule, "folds": len(selected)}
        metric_names = [key for key, value in selected[0].items() if isinstance(value, (int, float)) and key not in {"samples"}]
        for key in metric_names:
            result[key] = float(np.mean([row[key] for row in selected]))
        rows.append(result)
    return rows


def diagnostics(folds_data: dict, chosen_rule: str, test_scores: dict) -> tuple[list[dict], list[dict]]:
    quartile_rows, group_rows = [], []
    for fold, data in folds_data.items():
        bank = data["test"]
        native = bank["logits"]
        chosen = test_scores[fold][chosen_rule]
        q = bank["quality"]
        sample = np.arange(len(q)); oracle = q.argmin(1); native_selected = native.argmax(1)
        native_regret = q[sample, native_selected] - q[sample, oracle]
        margin = np.sort(native, axis=1)[:, -1] - np.sort(native, axis=1)[:, -2]
        for diagnostic, values in (("native_regret", native_regret), ("native_margin", margin)):
            edges = np.quantile(values, (0, .25, .5, .75, 1))
            for quartile in range(4):
                mask = (values >= edges[quartile]) & (values <= edges[quartile + 1] if quartile == 3 else values < edges[quartile + 1])
                # Tied diagnostic values can collapse adjacent quantiles and
                # leave an interval empty (notably native regret at zero).
                # Empty bins carry no statistics, so record/skip them instead
                # of asking percentile() to operate on an empty array.
                if not mask.any():
                    quartile_rows.append({"fold": fold, "diagnostic": diagnostic, "quartile": quartile + 1, "samples": 0})
                    continue
                native_metric = selection_metrics({key: value[mask] for key, value in bank.items() if isinstance(value, np.ndarray) and len(value) == len(q)}, native[mask])
                chosen_metric = selection_metrics({key: value[mask] for key, value in bank.items() if isinstance(value, np.ndarray) and len(value) == len(q)}, chosen[mask])
                quartile_rows.append({"fold": fold, "diagnostic": diagnostic, "quartile": quartile + 1, "samples": int(mask.sum()), "native_J": native_metric["top1_J"], "selected_J": chosen_metric["top1_J"], "J_gain_pct": 100 * (1 - chosen_metric["top1_J"] / max(native_metric["top1_J"], 1e-12)), "native_regret": native_metric["mean_regret"], "selected_regret": chosen_metric["mean_regret"]})
        correct = native_selected == oracle
        for group, mask in (("native_correct", correct), ("native_wrong", ~correct)):
            if not mask.any():
                group_rows.append({"fold": fold, "group": group, "samples": 0})
                continue
            row = {"fold": fold, "group": group, "samples": int(mask.sum()), "native_margin": float(margin[mask].mean())}
            for name in feature_lib.FEATURES:
                row[name] = float(bank["features"][mask, :, feature_lib.FEATURES.index(name)].mean())
            group_rows.append(row)
    return quartile_rows, group_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", default=["all"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    folds = FOLDS if args.folds == ["all"] else tuple(args.folds)
    if unknown := sorted(set(folds) - set(FOLDS)):
        raise ValueError(unknown)
    global OUT, CACHE
    if args.smoke:
        OUT = OUT / "preflight"
        CACHE = OUT / "cache"
        if args.max_samples <= 0:
            args.max_samples = 100
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    missing = [str(base.ckpt(fold, 1, "v2")) for fold in folds if not base.ckpt(fold, 1, "v2").is_file()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    extract_all(folds, device, args.batch_size, args.max_samples, args.force_cache)
    build_all_banks(folds, device, args.feature_batch_size, args.force_cache)

    preflight = {"feature_count": len(feature_lib.FEATURES), "candidate_identity_asserted": True, "GT_usage": "quality labels and offline evaluation only", "folds": {}}
    for fold in folds:
        fold_checks = {}
        for split in ("train", "test"):
            raw = load_npz(cache_path(fold, split, "raw"))
            bank = load_npz(cache_path(fold, split, "bank"))
            fold_checks[split] = {
                "samples": int(len(raw["prediction"])), "candidate_shape": list(raw["prediction"].shape[1:]),
                "source_anchor_shape": list(raw["source_anchors"].shape[1:]), "corrected_anchor_shape": list(raw["corrected_anchors"].shape[1:]),
                "mode_hidden_shape": list(raw["mode_hidden"].shape[1:]), "repeat_max_abs": float(raw["repeat_max_abs"][0]),
                "features_finite": bool(np.isfinite(bank["features"]).all()),
                "sigma_endpoint_from_source_train": float(bank["sigma_endpoint"][0]), "sigma_trajectory_from_source_train": float(bank["sigma_trajectory"][0]),
            }
        preflight["folds"][fold] = fold_checks
    (OUT / "preflight_checks.json").write_text(json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")

    folds_data, raw_rows, residual_rows, source_residual_rows, sanity = {}, [], [], [], []
    for fold in folds:
        train, test = load_npz(cache_path(fold, "train", "bank")), load_npz(cache_path(fold, "test", "bank"))
        calibrator = fit_score_control(train)
        train_residual = train["quality"] - train["quality"].mean(1, keepdims=True) - predict_score_quality(train, calibrator)
        test_residual = test["quality"] - test["quality"].mean(1, keepdims=True) - predict_score_quality(test, calibrator)
        raw_rows.extend(correlation_rows(fold, test, None, "outer_test_raw"))
        residual_rows.extend(correlation_rows(fold, test, test_residual, "outer_test_residual"))
        source_residual_rows.extend(correlation_rows(fold, train, train_residual, "source_train_residual"))
        train_rho = {row["feature"]: row["spearman"] for row in correlation_rows(fold, train, train_residual, "source_train_residual")}
        folds_data[fold] = {"train": train, "test": test, "train_residual_rho": train_rho}
        if not args.smoke and args.max_samples <= 0:
            sanity.append(reproduction_sanity(fold, test))

    raw_summary, raw_pass = aggregate_feature_gate(raw_rows, folds, exclude_score=False)
    residual_summary, residual_pass = aggregate_feature_gate(residual_rows, folds, exclude_score=True)
    source_summary, source_pass = aggregate_feature_gate(source_residual_rows, folds, exclude_score=True)
    write_csv(OUT / "feature_spearman.csv", raw_summary)
    write_csv(OUT / "residual_spearman.csv", residual_summary)
    write_csv(OUT / "source_train_residual_spearman.csv", source_summary)
    write_csv(OUT / "candidate_identity_and_reproduction.csv", sanity)
    write_csv(OUT / "winner_feature_analysis.csv", winner_analysis(folds_data, residual_pass))
    write_csv(OUT / "feature_directions.csv", [{"feature": name, "stored_as": feature_lib.STORED_AS[name], "higher_is_better": True, "supplementary": name in feature_lib.SUPPLEMENTARY} for name in feature_lib.FEATURES])
    gc.collect()

    per_fold_train, per_fold_test, train_scores, test_scores = [], [], {}, {}
    for fold, data in folds_data.items():
        train_scores[fold] = build_rule_scores(data["train"], data["train"], source_pass, data["train_residual_rho"])
        test_scores[fold] = build_rule_scores(data["train"], data["test"], source_pass, data["train_residual_rho"])
        for rule, score in train_scores[fold].items():
            per_fold_train.append({"fold": fold, "rule": rule, **selection_metrics(data["train"], score)})
        for rule, score in test_scores[fold].items():
            per_fold_test.append({"fold": fold, "rule": rule, **selection_metrics(data["test"], score)})
    train_macro = macro_rule_table(per_fold_train, folds)
    test_macro = macro_rule_table(per_fold_test, folds)
    primary_rules = [row for row in train_macro if row["rule"] in {"R0_Native", "R1_Medoid", "R2_Weighted_MBR", "R3_H3_MBR", "R4_Stable_Feature"}]
    chosen_rule = min(primary_rules, key=lambda row: row["top1_J"])["rule"]
    test_lookup = {row["rule"]: row for row in test_macro}
    native, chosen = test_lookup["R0_Native"], test_lookup[chosen_rule]
    positive_folds = sum(
        next(row for row in per_fold_test if row["fold"] == fold and row["rule"] == chosen_rule)["top1_J"]
        < next(row for row in per_fold_test if row["fold"] == fold and row["rule"] == "R0_Native")["top1_J"]
        for fold in folds
    )
    j_gain = 1 - chosen["top1_J"] / max(native["top1_J"], 1e-12)
    regret_gain = 1 - chosen["mean_regret"] / max(native["mean_regret"], 1e-12)
    p95_change = chosen["p95_regret"] / max(native["p95_regret"], 1e-12) - 1
    gates = {"top1_combined_gain_pct": 100*j_gain, "top1_combined_gain_ge_2pct": bool(j_gain >= .02), "positive_folds": int(positive_folds), "positive_folds_ge_4_of_5": bool(positive_folds >= 4 and len(folds) == 5), "mean_regret_gain_pct": 100*regret_gain, "mean_regret_gain_ge_5pct": bool(regret_gain >= .05), "p95_regret_change_pct": 100*p95_change, "p95_regret_safe": bool(p95_change <= .02)}
    if args.smoke or len(folds) < 5:
        decision = "PREFLIGHT_OK"
    elif j_gain >= .05 and regret_gain >= .08 and positive_folds >= 4 and p95_change < 0:
        decision = "STRONG_PASS"
    elif all((gates["top1_combined_gain_ge_2pct"], gates["positive_folds_ge_4_of_5"], gates["mean_regret_gain_ge_5pct"], gates["p95_regret_safe"])):
        decision = "PASS"
    elif 0.01 <= j_gain < .02 or positive_folds == 3:
        decision = "WEAK_SIGNAL"
    else:
        decision = "STOP_HEURISTIC_SELECTION"
    quartile_rows, group_rows = diagnostics(folds_data, chosen_rule, test_scores)
    write_csv(OUT / "per_fold_selection.csv", per_fold_test)
    write_csv(OUT / "selection_rules.csv", test_macro)
    write_csv(OUT / "source_train_rule_selection.csv", train_macro)
    write_csv(OUT / "hard_sample_diagnostics.csv", quartile_rows)
    write_csv(OUT / "native_correct_wrong.csv", group_rows)
    payload = {"experiment": "EXP-186A", "model": "EXP-173 F16+H3+Joint-V2 seed1 frozen", "smoke": args.smoke, "folds": list(folds), "primary_quality": "ADE + 0.25FDE", "top1_metric": "ADE + 0.5FDE", "feature_count": 25, "raw_signal_pass_features": raw_pass, "outer_test_residual_signal_pass_features": residual_pass, "source_train_rule_features": source_pass, "source_train_selected_rule": chosen_rule, "selection_rules": test_macro, "reproduction_sanity": sanity, "gates": gates, "decision": decision}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ["# EXP-186A Candidate Selection Feature Audit", "", f"Decision: **{decision}**", "", f"- Source/train-selected rule: {chosen_rule}", f"- Raw-signal features: {raw_pass or 'none'}", f"- Outer-test residual-signal features: {residual_pass or 'none'}", f"- Source/train rule features: {source_pass or 'none'}", f"- Top1 J gain: {100*j_gain:+.2f}%", f"- Positive folds: {positive_folds}/{len(folds)}", f"- Mean regret gain: {100*regret_gain:+.2f}%", f"- P95 regret change: {100*p95_change:+.2f}%", "", "F16, H3, Joint-V2, native logits, and all K20 trajectories remained frozen."]
    (OUT / "RESULT_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"chosen_rule": chosen_rule, "source_features": source_pass, "gates": gates, "decision": decision}, ensure_ascii=False, indent=2))
    print(f"[done] {OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
