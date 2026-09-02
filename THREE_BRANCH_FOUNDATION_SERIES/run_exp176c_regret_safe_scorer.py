"""EXP-176C: scene-held-out, regret-safe bounded residual scorer.

Only the two features admitted by EXP-176B are used.  EXP-173 candidates and
native logits are collected under inference_mode and never updated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_exp176b_cross_scene_feature_audit as feature_audit  # noqa: E402


base = feature_audit.base
DEFAULT_OUT = HERE / "EXP_176C_REGRET_SAFE_BOUNDED_SCORER"
FEATURES = ("acceleration_mismatch", "mean_social_clearance")
BETA = 0.35
KL_WEIGHT = 0.10
HIDDEN = 32


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BoundedResidualScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def within_sample_standardize(features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        normalized = (features - mean) / std
        # A constant feature (for example no social neighbour) must contribute
        # no candidate ordering signal.
        return torch.where(std > 1.0001e-5, normalized, torch.zeros_like(normalized))

    def forward(self, features: torch.Tensor, original_logits: torch.Tensor) -> torch.Tensor:
        normalized = self.within_sample_standardize(features)
        residual = torch.tanh(self.network(normalized).squeeze(-1))
        sigma = original_logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        return original_logits + BETA * sigma * residual


def run_cache_path(cache_dir: Path, outer_fold: str, data_scene: str, seed: int, max_samples: int) -> Path:
    suffix = "full" if max_samples <= 0 else f"n{max_samples}"
    return cache_dir / f"outer_{outer_fold}__scene_{data_scene}__seed_{seed}__{suffix}.npz"


def collect_scene(model, cfg, helper, data_scene: str, device: torch.device, batch_size: int, max_samples: int):
    dataset = helper.TrajectoryDataset(base.DATA / f"{data_scene}_data_test.npy", base.DATA / f"{data_scene}_num_test.npy", cfg)
    if max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    parts = {name: [] for name in ("features", "logits", "quality", "ade", "fde")}
    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            past, future, num_valid = past.to(device), future.to(device), num_valid.to(device).int()
            past, future = helper.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            raw_prediction = output[2] if isinstance(output, (tuple, list)) else output
            prediction = (raw_prediction + center.unsqueeze(-3))[:, 0]
            logits = model.last_mode_logits[:, 0]
            source = model.decoder.last_anchor_proposals[:, 0] + center[:, 0, None]
            corrected = model.decoder.last_corrected_anchors[:, 0] + center[:, 0, None]
            all_features = feature_audit.candidate_features(prediction, logits, past, num_valid, source, corrected)
            features = np.stack([all_features[name] for name in FEATURES], axis=-1).astype(np.float32)
            distance = torch.linalg.vector_norm(prediction - future[:, 0, None], dim=-1)
            ade, fde = distance.mean(dim=-1), distance[..., -1]
            parts["features"].append(features)
            parts["logits"].append(logits.float().cpu().numpy())
            parts["ade"].append(ade.float().cpu().numpy())
            parts["fde"].append(fde.float().cpu().numpy())
            parts["quality"].append((ade + 0.25 * fde).float().cpu().numpy())
    return {name: np.concatenate(values) for name, values in parts.items()}


def get_scene_cache(model, cfg, helper, cache_dir: Path, outer_fold: str, data_scene: str, seed: int,
                    device: torch.device, batch_size: int, max_samples: int, force: bool):
    path = run_cache_path(cache_dir, outer_fold, data_scene, seed, max_samples)
    if path.is_file() and not force:
        with np.load(path) as raw:
            arrays = {name: raw[name] for name in ("features", "logits", "quality", "ade", "fde")}
        if arrays["features"].shape[-1] == len(FEATURES):
            print(f"[cache] reuse {path.name}", flush=True)
            return arrays
    print(f"[cache] collect outer={outer_fold} data_scene={data_scene} seed={seed}", flush=True)
    arrays = collect_scene(model, cfg, helper, data_scene, device, batch_size, max_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, feature_names=np.asarray(FEATURES))
    return arrays


def concat_arrays(arrays: list[dict]) -> dict:
    return {name: np.concatenate([item[name] for item in arrays]) for name in arrays[0]}


def decisive_margin(scene_arrays: list[dict], rng: np.random.Generator, max_pairs: int = 250000) -> float:
    samples = []
    i, j = np.triu_indices(20, k=1)
    per_scene = max(1, max_pairs // len(scene_arrays))
    for arrays in scene_arrays:
        quality = arrays["quality"]
        ids = rng.choice(len(quality), size=min(len(quality), max(1, per_scene // len(i))), replace=False)
        differences = np.abs(quality[ids][:, i] - quality[ids][:, j]).reshape(-1)
        if len(differences) > per_scene:
            differences = rng.choice(differences, size=per_scene, replace=False)
        samples.append(differences)
    return float(np.median(np.concatenate(samples)))


def tensors(arrays: dict, ids: np.ndarray, device: torch.device):
    return tuple(torch.from_numpy(arrays[name][ids]).to(device=device, dtype=torch.float32) for name in ("features", "logits", "quality"))


def pairwise_loss(model: nn.Module, features: torch.Tensor, logits: torch.Tensor, quality: torch.Tensor, margin: float):
    new_logits = model(features, logits)
    i, j = torch.triu_indices(quality.shape[1], quality.shape[1], offset=1, device=quality.device)
    quality_difference = quality[:, i] - quality[:, j]
    decisive = quality_difference.abs() > margin
    preference = torch.sign(-quality_difference)
    score_difference = new_logits[:, i] - new_logits[:, j]
    if decisive.any():
        ranking = F.softplus(-preference[decisive] * score_difference[decisive]).mean()
    else:
        ranking = new_logits.sum() * 0.0
    native_probability = torch.softmax(logits.detach(), dim=1)
    anchoring = F.kl_div(torch.log_softmax(new_logits, dim=1), native_probability, reduction="batchmean")
    return ranking + KL_WEIGHT * anchoring, ranking.detach(), anchoring.detach()


def train_one_epoch(model, optimizer, scenes: list[dict], device: torch.device, batch_size: int,
                    samples_per_scene: int, margin: float, rng: np.random.Generator):
    model.train()
    per_scene_batch = max(1, batch_size // len(scenes))
    steps = max(1, math.ceil(samples_per_scene / per_scene_batch))
    total = rank_total = anchor_total = 0.0
    for _ in range(steps):
        batches = []
        for arrays in scenes:
            ids = rng.choice(len(arrays["quality"]), size=per_scene_batch, replace=len(arrays["quality"]) < per_scene_batch)
            batches.append(tensors(arrays, ids, device))
        features = torch.cat([batch[0] for batch in batches]); logits = torch.cat([batch[1] for batch in batches]); quality = torch.cat([batch[2] for batch in batches])
        loss, ranking, anchoring = pairwise_loss(model, features, logits, quality, margin)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        total += float(loss.detach()); rank_total += float(ranking); anchor_total += float(anchoring)
    return {"loss": total / steps, "ranking_loss": rank_total / steps, "anchor_kl": anchor_total / steps}


def predict_scores(model, arrays: dict, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval(); parts = []
    with torch.inference_mode():
        for start in range(0, len(arrays["quality"]), batch_size):
            ids = np.arange(start, min(start + batch_size, len(arrays["quality"])))
            features, logits, _ = tensors(arrays, ids, device)
            parts.append(model(features, logits).cpu().numpy())
    return np.concatenate(parts)


def score_metrics(arrays: dict, scores: np.ndarray):
    rows = np.arange(len(scores)); selected = scores.argmax(axis=1); best = arrays["quality"].argmin(axis=1)
    regret = arrays["quality"][rows, selected] - arrays["quality"][rows, best]
    return {
        "top1_ADE": float(arrays["ade"][rows, selected].mean()),
        "top1_FDE": float(arrays["fde"][rows, selected].mean()),
        "top1_combined_cost": float(arrays["quality"][rows, selected].mean()),
        "mean_regret": float(regret.mean()), "p90_regret": float(np.percentile(regret, 90)),
        "p95_regret": float(np.percentile(regret, 95)), "combined_winner_hit": float((selected == best).mean()),
        "active_modes": int(np.unique(selected).size),
    }


def evaluation(model, arrays: dict, device: torch.device, batch_size: int):
    native = score_metrics(arrays, arrays["logits"])
    bounded = score_metrics(arrays, predict_scores(model, arrays, device, batch_size))
    rows = np.arange(len(arrays["ade"])); best_ade = arrays["ade"].argmin(axis=1)
    oracle = {
        "minADE": float(arrays["ade"].min(axis=1).mean()), "minFDE": float(arrays["fde"].min(axis=1).mean()),
        "same_path_FDE": float(arrays["fde"][rows, best_ade].mean()),
    }
    improvement = {
        key: float(1.0 - bounded[key] / max(native[key], 1e-12))
        for key in ("top1_ADE", "top1_FDE", "top1_combined_cost", "mean_regret")
    }
    improvement["p95_regret_relative_change"] = float(bounded["p95_regret"] / max(native["p95_regret"], 1e-12) - 1.0)
    return {"samples": len(arrays["quality"]), "oracle": oracle, "native": native, "bounded": bounded, "relative_improvement": improvement}


def cross_validate(source: dict[str, dict], device: torch.device, seed: int, epochs: int, batch_size: int,
                   samples_per_scene: int, lr: float):
    rotations, histories = {}, []
    names = list(source)
    for val_index, val_name in enumerate(names):
        train_names = [name for name in names if name != val_name]
        train_scenes = [source[name] for name in train_names]
        local_seed = seed * 1000 + 176 + val_index
        seed_all(local_seed); rng = np.random.default_rng(local_seed)
        margin = decisive_margin(train_scenes, rng)
        model = BoundedResidualScorer().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        rows = []
        for epoch in range(epochs + 1):
            if epoch > 0:
                train_row = train_one_epoch(model, optimizer, train_scenes, device, batch_size, samples_per_scene, margin, rng)
            else:
                train_row = {"loss": None, "ranking_loss": None, "anchor_kl": None}
            val_result = evaluation(model, source[val_name], device, batch_size)
            row = {"validation_scene": val_name, "epoch": epoch, "margin": margin, **train_row,
                   **{f"val_{key}": value for key, value in val_result["relative_improvement"].items()},
                   "val_bounded_combined": val_result["bounded"]["top1_combined_cost"],
                   "val_native_combined": val_result["native"]["top1_combined_cost"]}
            rows.append(row); histories.append(row)
        rotations[val_name] = rows
    epoch_rows = []
    for epoch in range(epochs + 1):
        rows = [rotations[name][epoch] for name in names]
        epoch_rows.append({
            "epoch": epoch,
            "macro_combined_improvement": float(np.mean([row["val_top1_combined_cost"] for row in rows])),
            "macro_ade_improvement": float(np.mean([row["val_top1_ADE"] for row in rows])),
            "macro_fde_improvement": float(np.mean([row["val_top1_FDE"] for row in rows])),
            "macro_regret_improvement": float(np.mean([row["val_mean_regret"] for row in rows])),
            "macro_p95_relative_change": float(np.mean([row["val_p95_regret_relative_change"] for row in rows])),
        })
    safe = [row for row in epoch_rows if row["macro_p95_relative_change"] <= 0.02]
    candidates = safe if safe else epoch_rows
    selected = max(candidates, key=lambda row: (row["macro_combined_improvement"], row["macro_regret_improvement"], -row["epoch"]))
    return selected["epoch"], histories, epoch_rows


def fit_final(source: dict[str, dict], device: torch.device, seed: int, epochs: int, batch_size: int,
              samples_per_scene: int, lr: float):
    local_seed = seed * 1000 + 17600
    seed_all(local_seed); rng = np.random.default_rng(local_seed)
    scenes = list(source.values()); margin = decisive_margin(scenes, rng)
    model = BoundedResidualScorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        row = train_one_epoch(model, optimizer, scenes, device, batch_size, samples_per_scene, margin, rng)
        history.append({"epoch": epoch, "margin": margin, **row})
    return model, history, margin


def main():
    parser = argparse.ArgumentParser(description="EXP-176C bounded residual scorer with source-scene CV")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--samples-per-scene", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); args.output_dir = args.output_dir.resolve()
    folds = base.FOLDS if args.folds == ["all"] else args.folds
    unknown = sorted(set(folds) - set(base.FOLDS))
    if unknown:
        raise ValueError(f"Unknown folds: {unknown}")
    if len(folds) < 2:
        raise ValueError("At least two outer folds are required")
    audit_path = HERE / "EXP_176B_CROSS_SCENE_SCORE_FEATURE_AUDIT" / "cross_scene_feature_audit.json"
    if not audit_path.is_file() or not json.loads(audit_path.read_text(encoding="utf-8")).get("gate_pass"):
        raise RuntimeError("EXP-176B gate has not passed")
    missing = [str(base.ckpt(fold, seed, "v2")) for fold in folds for seed in args.seeds if not base.ckpt(fold, seed, "v2").is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    trainable_parameters = sum(parameter.numel() for parameter in BoundedResidualScorer().parameters())
    if args.dry_run:
        print(json.dumps({"status": "preflight_ok", "experiment": "EXP-176C", "folds": folds, "seeds": args.seeds,
                          "features": FEATURES, "beta": BETA, "hidden": HIDDEN, "trainable_parameters": trainable_parameters,
                          "outer_runs": len(folds) * len(args.seeds)}, ensure_ascii=False, indent=2)); return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    base.configure(device)
    helper = base.load_module("exp176c_trainer", base.TRAINER)
    models = base.load_module("exp176c_models", base.MODEL_FILE)
    adapter = base.load_module("exp176c_adapter", base.ADAPTER)
    cache_dir = args.output_dir / "cache"; fold_rows, cv_rows, final_histories = [], [], []
    for seed in args.seeds:
        for outer_fold in folds:
            print(f"[outer] fold={outer_fold} seed={seed}", flush=True)
            model, cfg = base.build(outer_fold, seed, device, helper, models, adapter)
            source = {}
            for scene in base.FOLDS:
                if scene == outer_fold:
                    continue
                source[scene] = get_scene_cache(model, cfg, helper, cache_dir, outer_fold, scene, seed, device,
                                                args.inference_batch_size, args.max_samples, args.force_cache)
            selected_epoch, history, epoch_summary = cross_validate(source, device, seed + base.FOLDS.index(outer_fold) * 100,
                                                                    args.epochs, args.batch_size, args.samples_per_scene,
                                                                    args.learning_rate)
            for row in history:
                cv_rows.append({"outer_fold": outer_fold, "seed": seed, **row})
            (args.output_dir / "cv").mkdir(parents=True, exist_ok=True)
            (args.output_dir / "cv" / f"{outer_fold}_seed_{seed}.json").write_text(
                json.dumps({"selected_epoch": selected_epoch, "epoch_summary": epoch_summary}, indent=2), encoding="utf-8")
            scorer, final_history, margin = fit_final(source, device, seed + base.FOLDS.index(outer_fold) * 100,
                                                       selected_epoch, args.batch_size, args.samples_per_scene, args.learning_rate)
            for row in final_history:
                final_histories.append({"outer_fold": outer_fold, "seed": seed, **row})
            # The held-out target is accessed only after source-scene CV and final fitting.
            target = get_scene_cache(model, cfg, helper, cache_dir, outer_fold, outer_fold, seed, device,
                                     args.inference_batch_size, args.max_samples, args.force_cache)
            result = evaluation(scorer, target, device, args.batch_size)
            combined_improved = result["relative_improvement"]["top1_combined_cost"] > 0.0
            fold_rows.append({"fold": outer_fold, "seed": seed, "selected_epoch": selected_epoch, "pair_margin": margin,
                              "combined_improved": combined_improved, **{f"oracle_{k}": v for k, v in result["oracle"].items()},
                              **{f"native_{k}": v for k, v in result["native"].items()},
                              **{f"bounded_{k}": v for k, v in result["bounded"].items()},
                              **{f"improvement_{k}": v for k, v in result["relative_improvement"].items()}})
            checkpoint_dir = args.output_dir / "scorers"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": scorer.state_dict(), "features": FEATURES, "beta": BETA, "hidden": HIDDEN,
                        "outer_fold": outer_fold, "seed": seed, "selected_epoch": selected_epoch},
                       checkpoint_dir / f"{outer_fold}_seed_{seed}.pth")
            del model, scorer, source, target
            if device.type == "cuda": torch.cuda.empty_cache()
    metric_pairs = [("top1_ADE", "native_top1_ADE", "bounded_top1_ADE"), ("top1_FDE", "native_top1_FDE", "bounded_top1_FDE"),
                    ("top1_combined_cost", "native_top1_combined_cost", "bounded_top1_combined_cost"),
                    ("mean_regret", "native_mean_regret", "bounded_mean_regret"), ("p95_regret", "native_p95_regret", "bounded_p95_regret")]
    macro = {}
    for name, native_key, bounded_key in metric_pairs:
        native_mean = float(np.mean([row[native_key] for row in fold_rows])); bounded_mean = float(np.mean([row[bounded_key] for row in fold_rows]))
        macro[f"native_{name}"] = native_mean; macro[f"bounded_{name}"] = bounded_mean
        macro[f"improvement_{name}"] = float(1.0 - bounded_mean / max(native_mean, 1e-12))
    fold_means = {fold: float(np.mean([row["improvement_top1_combined_cost"] for row in fold_rows if row["fold"] == fold])) for fold in folds}
    improved_folds = sum(value > 0.0 for value in fold_means.values())
    gates = {
        "top1_ADE_ge_5pct": macro["improvement_top1_ADE"] >= 0.05,
        "top1_FDE_ge_5pct": macro["improvement_top1_FDE"] >= 0.05,
        "mean_regret_ge_10pct": macro["improvement_mean_regret"] >= 0.10,
        "p95_regret_not_worse_2pct": macro["bounded_p95_regret"] <= macro["native_p95_regret"] * 1.02,
        "combined_improves_ge_4_of_5_folds": improved_folds >= 4 and len(folds) >= 5,
    }
    payload = {"experiment": "EXP-176C", "candidate_generator": "frozen EXP173 F16 Joint-V2", "features": FEATURES,
               "scorer": {"architecture": "2->32->1", "trainable_parameters": trainable_parameters, "beta": BETA,
                          "pairwise_margin": "source-train q-difference P50", "anchor_kl_weight": KL_WEIGHT},
               "validation": "four-way source-scene-held-out CV selects epoch; target accessed only afterward",
               "fold_seed_results": fold_rows, "macro": macro, "fold_combined_improvement": fold_means,
               "improved_fold_count": improved_folds, "gates": gates, "gate_pass": all(gates.values()),
               "recommendation": "Keep EXP-176C" if all(gates.values()) else "Reject EXP-176C under the predefined safety gate"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, rows in (("per_fold_seed.csv", fold_rows), ("cv_history.csv", cv_rows), ("final_training_history.csv", final_histories)):
        if rows:
            with (args.output_dir / filename).open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"macro": macro, "gates": gates, "gate_pass": payload["gate_pass"], "recommendation": payload["recommendation"]}, ensure_ascii=False, indent=2))
    print(f"[complete] {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
