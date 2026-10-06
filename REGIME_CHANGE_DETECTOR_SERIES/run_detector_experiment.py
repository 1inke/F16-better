"""Observation-only regime-change feasibility and frozen-baseline failure diagnosis."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_recall_curve,
    roc_auc_score, roc_curve,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASTRA = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = ASTRA / "datasets" / "eth_ucy" / "processed_data_diverse" / "eth_data_train.npy"
NUM = ASTRA / "datasets" / "eth_ucy" / "processed_data_diverse" / "eth_num_train.npy"
CONFIG = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
C00_DIR = (
    ROOT / "QUALITY_AWARE_F16_SERIES_V2" / "results" / "p0_protocol"
    / "C0_EXP173_EXACT" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"
)
C00 = C00_DIR / "best_selected.pth"
H3 = (
    ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
    / "results" / "eth" / "h3" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1" / "best_selected.pth"
)
SPEC_PATH = HERE / "experiment_config.json"
PREFLIGHT = HERE / "preflight.json"
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
VARIANTS = ("d0_motion_only", "d1_motion_social")


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup(device: torch.device):
    os.chdir(ASTRA)
    sys.path[:0] = [str(ASTRA), str(ASTRA / "scripts"), str(ROOT / "FRONT_ENCODER_SERIES")]
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20

    cfg = load_locked_q20(CONFIG)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    front = load_module("regime_front", ROOT / "FRONT_ENCODER_SERIES" / "front_encoder_models.py")
    trainer = load_module("regime_trainer", ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py")
    base = load_module("regime_base", ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py")
    return cfg, front, trainer, base


def exact_splits(spec: dict) -> Dict[str, np.ndarray]:
    with np.load(C00_DIR / "split_indices.npz", allow_pickle=False) as payload:
        official_train = np.asarray(payload["train"], dtype=np.int64)
        validation = np.asarray(payload["validation"], dtype=np.int64)
    raw = np.load(DATA, mmap_mode="r")
    scene_ids = np.asarray(raw[:, 0, 0, 0], dtype=np.int64)
    dev_parts = []
    for scene in np.unique(scene_ids):
        candidates = official_train[scene_ids[official_train] == scene]
        count = max(1, min(len(candidates) - 1, int(round(len(candidates) * 0.05))))
        dev_parts.append(candidates[-count:])
    detector_dev = np.sort(np.concatenate(dev_parts))
    detector_train = np.setdiff1d(official_train, detector_dev, assume_unique=True)
    if np.intersect1d(detector_train, detector_dev).size:
        raise RuntimeError("Detector train/dev overlap")
    if np.intersect1d(official_train, validation).size:
        raise RuntimeError("Official train/validation overlap")
    if not np.array_equal(np.sort(np.concatenate((detector_train, detector_dev))), official_train):
        raise RuntimeError("Detector train/dev do not reconstruct the locked C00 training split")
    return {"train": detector_train, "dev": detector_dev, "validation": validation}


def future_change_score(past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    observed = past[:, 0, -1] - past[:, 0, -2]
    future_mean = (future[:, 0, -1] - past[:, 0, -1]) / float(future.shape[-2])
    observed_speed = torch.linalg.vector_norm(observed, dim=-1)
    future_speed = torch.linalg.vector_norm(future_mean, dim=-1)
    product = observed_speed * future_speed
    cosine = (observed * future_mean).sum(dim=-1) / product.clamp_min(1.0e-6)
    cosine = torch.where(product > 1.0e-6, cosine.clamp(-1.0, 1.0), torch.ones_like(cosine))
    turn = ((1.0 - cosine) / 2.0).clamp(0.0, 1.0)
    speed = (
        torch.abs(torch.log((future_speed + 0.05) / (observed_speed + 0.05))) / math.log(2.0)
    ).clamp(0.0, 1.0)
    return torch.maximum(turn, speed)


def group_from_score(score: np.ndarray, spec: dict) -> np.ndarray:
    group = np.full(len(score), 1, dtype=np.int64)
    group[score < spec["stable_threshold"]] = 0
    group[score >= spec["change_threshold"]] = 2
    return group


def motion_features(past: torch.Tensor) -> torch.Tensor:
    """Causal focal-history features; `past` is already observation-only heading normalized."""
    focal = past[:, 0]
    velocity = focal[:, 1:] - focal[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    acc_norm = torch.linalg.vector_norm(acceleration, dim=-1)
    v0, v1 = velocity[:, :-1], velocity[:, 1:]
    denom = (torch.linalg.vector_norm(v0, dim=-1) * torch.linalg.vector_norm(v1, dim=-1)).clamp_min(1e-6)
    turn_cos = (v0 * v1).sum(dim=-1) / denom
    turn_sin = (v0[..., 0] * v1[..., 1] - v0[..., 1] * v1[..., 0]) / denom
    moving = denom > 1e-6
    turn_cos = torch.where(moving, turn_cos.clamp(-1, 1), torch.ones_like(turn_cos))
    turn_sin = torch.where(moving, turn_sin.clamp(-1, 1), torch.zeros_like(turn_sin))
    summary = torch.stack((
        speed[:, -1], speed.mean(1), speed.std(1, unbiased=False),
        acc_norm.mean(1), acc_norm.max(1).values,
        torch.linalg.vector_norm(focal[:, -1] - focal[:, 0], dim=-1),
        speed.sum(1), turn_cos.mean(1), turn_sin.abs().mean(1),
    ), dim=-1)
    return torch.cat((
        focal.flatten(1), velocity.flatten(1), acceleration.flatten(1),
        speed, acc_norm, turn_cos, turn_sin, summary,
    ), dim=-1)


class DetectorMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, observed_features: torch.Tensor) -> torch.Tensor:
        return self.network(observed_features).squeeze(-1)


def build_baseline(cfg, front, base, device: torch.device):
    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(H3)
    os.environ["EXP173_DECODER_INIT_SEED"] = "1"
    model = front.FrontEncoderASTRA(copy.deepcopy(cfg), "f16_temporal_social_cross")
    base.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
    model.load_state_dict(torch.load(C00, map_location="cpu", weights_only=True), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()


def extract_features(
    name: str, ids: np.ndarray, dataset, model, trainer, device: torch.device,
    batch_size: int, spec: dict, collect_baseline: bool,
) -> dict:
    loader = DataLoader(Subset(dataset, ids.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    parts: Dict[str, list] = {key: [] for key in ("motion", "social", "change_score", "change_group")}
    if collect_baseline:
        for key in ("ade", "fde", "logits"):
            parts[key] = []
    seen = 0
    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past_local, future_local = trainer.normalize_to_focal_heading(past, future)
            motion = motion_features(past_local)
            center = past_local.mean(dim=-2, keepdim=True)
            prediction = model(
                past_local - center, None, None, mode="testing", num_valid=num_valid,
                graph_loc=past_local,
            )[2] + center.unsqueeze(-3)
            social = model.agent_aware_trans_encoder.last_agent_features[:, 0]
            score = future_change_score(past_local, future_local)
            parts["motion"].append(motion.cpu().numpy().astype(np.float32))
            parts["social"].append(social.cpu().numpy().astype(np.float32))
            parts["change_score"].append(score.cpu().numpy().astype(np.float32))
            parts["change_group"].append(group_from_score(score.cpu().numpy(), spec))
            if collect_baseline:
                candidates = prediction[:, 0]
                errors = torch.linalg.vector_norm(candidates - future_local[:, 0, None], dim=-1)
                parts["ade"].append(errors.mean(-1).cpu().numpy().astype(np.float32))
                parts["fde"].append(errors[..., -1].cpu().numpy().astype(np.float32))
                parts["logits"].append(model.last_mode_logits[:, 0].cpu().numpy().astype(np.float32))
            seen += len(past)
    if seen != len(ids):
        raise RuntimeError(f"Feature count mismatch for {name}: {seen} != {len(ids)}")
    result = {key: np.concatenate(value) for key, value in parts.items()}
    result["sample_id"] = ids.copy()
    np.savez_compressed(RESULTS / f"features_{name}.npz", **result)
    return result


def standardize(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(0, keepdims=True)
    std = train.std(0, keepdims=True)
    std = np.where(std < 1.0e-6, 1.0, std)
    transformed = [((values - mean) / std).astype(np.float32) for values in (train,) + others]
    return transformed, mean.astype(np.float32), std.astype(np.float32)


def binary_arrays(features: np.ndarray, groups: np.ndarray):
    mask = groups != 1
    return features[mask], (groups[mask] == 2).astype(np.float32)


def train_detector(variant: str, train_data: dict, dev_data: dict, spec: dict, device: torch.device):
    train_x = train_data["motion"] if variant == "d0_motion_only" else np.concatenate(
        (train_data["motion"], train_data["social"]), axis=1
    )
    dev_x = dev_data["motion"] if variant == "d0_motion_only" else np.concatenate(
        (dev_data["motion"], dev_data["social"]), axis=1
    )
    (train_x, dev_x), mean, std = standardize(train_x, dev_x)
    train_x, train_y = binary_arrays(train_x, train_data["change_group"])
    dev_x, dev_y = binary_arrays(dev_x, dev_data["change_group"])
    model = DetectorMLP(train_x.shape[1], spec["dropout"]).to(device)
    negatives = float((train_y == 0).sum())
    positives = float((train_y == 1).sum())
    pos_weight = torch.tensor(negatives / max(positives, 1.0), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"]
    )
    generator = torch.Generator().manual_seed(spec["seed"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=spec["batch_size"], shuffle=True, generator=generator, num_workers=0,
    )
    dev_tensor = torch.from_numpy(dev_x).to(device)
    history, best_auc, best_epoch, best_state = [], -float("inf"), -1, None
    for epoch in range(spec["epochs"]):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            dev_p = torch.sigmoid(model(dev_tensor)).cpu().numpy()
        dev_auc = float(roc_auc_score(dev_y, dev_p))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "dev_auroc": dev_auc})
        if dev_auc > best_auc:
            best_auc, best_epoch = dev_auc, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"No detector checkpoint for {variant}")
    model.load_state_dict(best_state)
    variant_dir = RESULTS / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, variant_dir / "best_dev_auroc.pth")
    np.savez_compressed(variant_dir / "scaler.npz", mean=mean, std=std)
    write_csv(variant_dir / "history.csv", history)
    return model.eval(), mean, std, {"best_dev_auroc": best_auc, "best_epoch": best_epoch,
                                     "input_dim": int(train_x.shape[1]),
                                     "train_binary_n": int(len(train_y)),
                                     "train_positive_prevalence": float(train_y.mean())}


def predict(model, values: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    values = ((values - mean) / std).astype(np.float32)
    loader = DataLoader(torch.from_numpy(values), batch_size=1024, shuffle=False)
    output = []
    with torch.inference_mode():
        for batch in loader:
            output.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def ece_score(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if mask.any():
            value += mask.mean() * abs(probability[mask].mean() - labels[mask].mean())
    return float(value)


def bootstrap_detector(labels: np.ndarray, probability: np.ndarray, repetitions: int, seed: int) -> dict:
    negative, positive = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in ("auroc", "pr_auc", "brier")}
    for _ in range(repetitions):
        indices = np.concatenate((
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ))
        y, p = labels[indices], probability[indices]
        samples["auroc"].append(roc_auc_score(y, p))
        samples["pr_auc"].append(average_precision_score(y, p))
        samples["brier"].append(brier_score_loss(y, p))
    result = {}
    for name, values in samples.items():
        result[f"{name}_ci_lower"], result[f"{name}_ci_upper"] = map(
            float, np.quantile(values, (0.025, 0.975))
        )
    return result


def bootstrap_correlation(x: np.ndarray, y: np.ndarray, repetitions: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pearson_samples, spearman_samples = [], []
    for _ in range(repetitions):
        index = rng.integers(0, len(x), len(x))
        bx, by = x[index], y[index]
        pearson_samples.append(np.corrcoef(bx, by)[0, 1])
        spearman_samples.append(np.corrcoef(rankdata(bx), rankdata(by))[0, 1])
    p_ci = np.quantile(pearson_samples, (0.025, 0.975))
    s_ci = np.quantile(spearman_samples, (0.025, 0.975))
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "pearson_ci_lower": float(p_ci[0]), "pearson_ci_upper": float(p_ci[1]),
        "spearman": float(spearmanr(x, y).statistic),
        "spearman_ci_lower": float(s_ci[0]), "spearman_ci_upper": float(s_ci[1]),
    }


def mean_ci(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repetitions):
        means.append(values[rng.integers(0, len(values), len(values))].mean())
    return tuple(map(float, np.quantile(means, (0.025, 0.975))))


def detector_metrics(probability: np.ndarray, validation: dict, spec: dict, seed: int) -> tuple[dict, list]:
    groups = validation["change_group"]
    binary = groups != 1
    labels = (groups[binary] == 2).astype(np.int64)
    p_binary = probability[binary]
    metrics = {
        "binary_n": int(binary.sum()), "positive_prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, p_binary)),
        "pr_auc": float(average_precision_score(labels, p_binary)),
        "brier": float(brier_score_loss(labels, p_binary)),
        "ece": ece_score(labels, p_binary),
    }
    metrics.update(bootstrap_detector(labels, p_binary, spec["bootstrap_repetitions"], seed))
    metrics.update(bootstrap_correlation(
        probability, validation["change_score"], spec["bootstrap_repetitions"], seed + 1
    ))
    rows = []
    names = ("stable", "transition", "change")
    for group_id, name in enumerate(names):
        values = probability[groups == group_id]
        lower, upper = mean_ci(values, spec["bootstrap_repetitions"], seed + 10 + group_id)
        rows.append({"group": name, "n": len(values), "mean_probability": float(values.mean()),
                     "median_probability": float(np.median(values)),
                     "ci_lower": lower, "ci_upper": upper})
    metrics["monotonic_group_probability"] = bool(
        rows[0]["mean_probability"] < rows[1]["mean_probability"] < rows[2]["mean_probability"]
    )
    metrics["gate_a"] = bool(
        metrics["auroc_ci_lower"] > 0.5
        and metrics["pr_auc"] > metrics["positive_prevalence"]
        and metrics["monotonic_group_probability"]
        and metrics["spearman"] > 0.0 and metrics["spearman_ci_lower"] > 0.0
    )
    return metrics, rows


def baseline_diagnostics(validation: dict) -> dict:
    ade, fde, logits = validation["ade"], validation["fde"], validation["logits"]
    rows = np.arange(len(ade))
    selected = logits.argmax(1)
    oracle = ade.argmin(1)
    order = np.argsort(-logits, axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[rows[:, None], order] = np.arange(logits.shape[1])[None]
    return {
        "oracle_ade": ade[rows, oracle].astype(np.float64),
        "selected_ade": ade[rows, selected].astype(np.float64),
        "ade_regret": (ade[rows, selected] - ade[rows, oracle]).astype(np.float64),
        "oracle_fde": fde[rows, oracle].astype(np.float64),
        "selected_fde": fde[rows, selected].astype(np.float64),
        "fde_regret": (fde[rows, selected] - fde[rows, oracle]).astype(np.float64),
        "gt_best_candidate_idx": oracle.astype(np.int64),
        "gt_best_candidate_rank": (rank[rows, oracle] + 1).astype(np.float64),
        "selected_candidate_idx": selected.astype(np.int64),
    }


def failure_analysis(probability: np.ndarray, validation: dict, spec: dict) -> tuple[dict, list, dict]:
    values = baseline_diagnostics(validation)
    correlations = {}
    for offset, metric in enumerate(("ade_regret", "fde_regret", "gt_best_candidate_rank")):
        correlations[metric] = bootstrap_correlation(
            probability, values[metric], spec["bootstrap_repetitions"], spec["bootstrap_seed"] + 100 + offset
        )
    order = np.argsort(probability, kind="stable")
    bin_index = np.empty(len(probability), dtype=np.int64)
    for index, members in enumerate(np.array_split(order, 5), start=1):
        bin_index[members] = index
    bin_rows = []
    for index in range(1, 6):
        mask = bin_index == index
        bin_rows.append({
            "bin": f"Q{index}", "n": int(mask.sum()),
            "mean_probability": float(probability[mask].mean()),
            **{metric: float(array[mask].mean()) for metric, array in values.items()
               if metric not in ("gt_best_candidate_idx", "selected_candidate_idx")},
        })
    positive = [metric for metric, row in correlations.items()
                if row["spearman"] > 0.0 and row["spearman_ci_lower"] > 0.0]
    high_worse = (
        bin_rows[-1]["ade_regret"] > bin_rows[0]["ade_regret"]
        or bin_rows[-1]["fde_regret"] > bin_rows[0]["fde_regret"]
        or bin_rows[-1]["gt_best_candidate_rank"] > bin_rows[0]["gt_best_candidate_rank"]
    )
    result = {"correlations": correlations, "positive_core_correlations": positive,
              "q5_worse_than_q1": bool(high_worse), "gate_b": bool(positive and high_worse)}
    return result, bin_rows, values | {"probability_bin": bin_index}


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(probabilities: Dict[str, np.ndarray], validation: dict, best: str,
                 bin_rows: list, baseline: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    groups = validation["change_group"]
    binary = groups != 1
    labels = (groups[binary] == 2).astype(int)
    plt.figure(figsize=(6, 5))
    for name, p in probabilities.items():
        fpr, tpr, _ = roc_curve(labels, p[binary]); plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "01_roc_curve.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 5))
    for name, p in probabilities.items():
        precision, recall, _ = precision_recall_curve(labels, p[binary]); plt.plot(recall, precision, label=name)
    plt.axhline(labels.mean(), color="k", linestyle="--"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "02_precision_recall_curve.png", dpi=180); plt.close()
    p = probabilities[best]
    plt.figure(figsize=(6, 5))
    edges = np.linspace(0, 1, 11)
    centers, observed = [], []
    for index in range(10):
        mask = (p[binary] >= edges[index]) & (p[binary] < edges[index + 1] if index < 9 else p[binary] <= 1)
        if mask.any(): centers.append(float(p[binary][mask].mean())); observed.append(float(labels[mask].mean()))
    plt.plot(centers, observed, "o-"); plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.tight_layout()
    plt.savefig(FIGURES / "03_reliability_diagram.png", dpi=180); plt.close()
    # Matplotlib 3.8 uses ``labels``; ``tick_labels`` was introduced later.
    plt.figure(figsize=(7, 5)); plt.boxplot([p[groups == i] for i in range(3)], labels=["stable", "transition", "change"], showfliers=False); plt.ylabel("p(change)"); plt.tight_layout()
    plt.savefig(FIGURES / "04_group_probability_boxplot.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 5)); plt.scatter(validation["change_score"], p, s=8, alpha=.25); plt.xlabel("Future change score (diagnostic only)"); plt.ylabel("p(change)"); plt.tight_layout()
    plt.savefig(FIGURES / "05_probability_vs_change_score.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 5)); plt.scatter(p, baseline["ade_regret"], s=8, alpha=.25); plt.xlabel("p(change)"); plt.ylabel("ADE regret"); plt.tight_layout()
    plt.savefig(FIGURES / "06_probability_vs_ade_regret.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 5)); plt.plot(range(1, 6), [row["ade_regret"] for row in bin_rows], "o-"); plt.xlabel("Probability quintile"); plt.ylabel("Mean ADE regret"); plt.tight_layout()
    plt.savefig(FIGURES / "07_probability_bin_vs_regret.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 5));
    high = p >= np.quantile(p, .8)
    bins = np.arange(1, 22) - .5
    plt.hist(baseline["gt_best_candidate_rank"][groups == 2], bins=bins, alpha=.55, label="true change")
    plt.hist(baseline["gt_best_candidate_rank"][high], bins=bins, alpha=.55, label="top p(change) quintile")
    plt.xlabel("GT-nearest candidate rank (1=best)"); plt.ylabel("Count"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "08_high_change_oracle_rank.png", dpi=180); plt.close()


def preflight(spec: dict) -> dict:
    required = [DATA, NUM, CONFIG, C00, H3, C00_DIR / "split_indices.npz", SPEC_PATH, Path(__file__)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    splits = exact_splits(spec)
    raw = np.load(DATA, mmap_mode="r")
    coords = torch.from_numpy(np.array(raw[:, 0, :, -2:], dtype=np.float32, copy=True))
    past, future = coords[:, :8], coords[:, 8:20]
    score = future_change_score(past[:, None], future[:, None]).numpy()
    group = group_from_score(score, spec)
    counts = {name: {label: int((group[ids] == value).sum()) for value, label in enumerate(("stable", "transition", "change"))}
              for name, ids in splits.items()}
    if any(min(values.values()) < 20 for values in counts.values()):
        raise RuntimeError(f"A split has too few regime samples: {counts}")
    manifest = {
        "status": "PREFLIGHT_PASS_NO_TRAINING", "protocol": spec["protocol"],
        "split_counts": {name: int(len(ids)) for name, ids in splits.items()},
        "group_counts": counts,
        "validation_id_sha256": hashlib.sha256(splits["validation"].tobytes()).hexdigest(),
        "sources": {str(path): sha256(path) for path in required},
        "observation_only_inputs": True, "future_used_only_for_training_labels_and_offline_evaluation": True,
        "test_paths_present": False, "stage_c_automatic": False,
    }
    if PREFLIGHT.is_file():
        old = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError("Preflight/source drift detected; review before replacing the lock")
    else:
        PREFLIGHT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def run(device: torch.device, spec: dict) -> None:
    if (RESULTS / "verdict.json").is_file():
        print(f"[skip complete] {RESULTS / 'verdict.json'}")
        return
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    seed_everything(spec["seed"])
    splits = exact_splits(spec)
    np.savez_compressed(RESULTS / "split_indices.npz", **splits)
    cfg, front, trainer, base = setup(device)
    dataset = trainer.TrajectoryDataset(DATA, NUM, cfg)
    baseline_model = build_baseline(cfg, front, base, device)
    extracted = {}
    for name in ("train", "dev", "validation"):
        print(f"[features] {name}: {len(splits[name])}", flush=True)
        extracted[name] = extract_features(
            name, splits[name], dataset, baseline_model, trainer, device,
            spec["feature_batch_size"], spec, collect_baseline=(name == "validation"),
        )
    del baseline_model
    if device.type == "cuda": torch.cuda.empty_cache()

    detector_rows, group_rows, probabilities, training = [], [], {}, {}
    for index, variant in enumerate(VARIANTS):
        print(f"[train detector] {variant}", flush=True)
        seed_everything(spec["seed"])
        model, mean, std, train_info = train_detector(variant, extracted["train"], extracted["dev"], spec, device)
        val_x = extracted["validation"]["motion"] if variant == "d0_motion_only" else np.concatenate(
            (extracted["validation"]["motion"], extracted["validation"]["social"]), axis=1
        )
        probability = predict(model, val_x, mean, std, device)
        probabilities[variant] = probability
        metrics, groups = detector_metrics(
            probability, extracted["validation"], spec, spec["bootstrap_seed"] + index * 1000
        )
        training[variant] = train_info
        detector_rows.append({"model": variant, **train_info, **metrics})
        group_rows.extend({"model": variant, **row} for row in groups)
        np.save(RESULTS / variant / "validation_probability.npy", probability)

    best = max(VARIANTS, key=lambda name: training[name]["best_dev_auroc"])
    best_metrics = next(row for row in detector_rows if row["model"] == best)
    failure, bin_rows, baseline = failure_analysis(probabilities[best], extracted["validation"], spec)
    gate_a = bool(best_metrics["gate_a"])
    gate_b = bool(failure["gate_b"])
    verdict = (
        "STOP_CHANGE_DETECTOR" if not gate_a else
        "STOP_CHANGE_AWARE_SCORING" if not gate_b else
        "H1_H2_PASS_READY_FOR_MINIMAL_INTERVENTION_REVIEW"
    )
    write_csv(RESULTS / "detector_metrics.csv", detector_rows)
    write_csv(RESULTS / "group_probability.csv", group_rows)
    write_csv(RESULTS / "probability_bin_regret.csv", bin_rows)
    sample_rows = []
    for index, sample_id in enumerate(extracted["validation"]["sample_id"]):
        sample_rows.append({
            "sample_id": int(sample_id), "split": "validation",
            "change_label": int(extracted["validation"]["change_group"][index] == 2),
            "change_group": ("stable", "transition", "change")[extracted["validation"]["change_group"][index]],
            "future_change_score": float(extracted["validation"]["change_score"][index]),
            "p_change": float(probabilities[best][index]),
            **{key: (int(value[index]) if key.endswith("idx") else float(value[index]))
               for key, value in baseline.items()},
        })
    write_csv(RESULTS / "sample_level_diagnostics.csv", sample_rows)
    make_figures(probabilities, extracted["validation"], best, bin_rows, baseline)
    payload = {
        "status": verdict, "best_detector_selected_on_internal_dev_only": best,
        "gate_a": gate_a, "gate_b": gate_b, "detector_metrics": detector_rows,
        "failure_diagnosis": failure, "training": training,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_samples": int(len(splits["validation"])), "test_accessed": False,
        "stage_c_started": False,
        "bootstrap_scope": "sample uncertainty conditional on one detector seed",
    }
    (RESULTS / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "verdict.json").write_text(json.dumps({
        "verdict": verdict, "best_detector": best, "gate_a": gate_a, "gate_b": gate_b,
        "stage_c_started": False, "test_accessed": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Observation-Only Regime Change Detector 验证报告", "", f"状态：`{verdict}`", "",
        f"D-best 由内部 detector-dev AUROC 选择：`{best}`；正式 validation 未参与模型或变体选择。", "",
        "## Detector performance", "",
        "| Model | Dev AUROC | Val AUROC | 95% CI | PR-AUC | Prevalence | Brier | ECE | Spearman | 95% CI | Gate A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in detector_rows:
        lines.append(
            f"| {row['model']} | {row['best_dev_auroc']:.4f} | {row['auroc']:.4f} | "
            f"[{row['auroc_ci_lower']:.4f}, {row['auroc_ci_upper']:.4f}] | {row['pr_auc']:.4f} | "
            f"{row['positive_prevalence']:.4f} | {row['brier']:.4f} | {row['ece']:.4f} | "
            f"{row['spearman']:.4f} | [{row['spearman_ci_lower']:.4f}, {row['spearman_ci_upper']:.4f}] | "
            f"{'PASS' if row['gate_a'] else 'FAIL'} |"
        )
    lines += ["", "## D-best probability groups", "", "| Group | N | Mean p | Median p | 95% CI |",
              "|---|---:|---:|---:|---:|"]
    for row in group_rows:
        if row["model"] == best:
            lines.append(f"| {row['group']} | {row['n']} | {row['mean_probability']:.4f} | {row['median_probability']:.4f} | [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] |")
    lines += ["", "## Failure diagnosis", "",
              f"Gate B: `{'PASS' if gate_b else 'FAIL'}`", ""]
    for metric, row in failure["correlations"].items():
        lines.append(f"- p(change) vs {metric}: Spearman={row['spearman']:.4f}, 95% CI [{row['spearman_ci_lower']:.4f}, {row['spearman_ci_upper']:.4f}].")
    lines += ["", "## Boundaries", "",
              "- 输入仅包含 observed history 与冻结 F16 observation-only social representation。",
              "- future 只用于训练标签和离线评估；没有 test 访问。",
              "- 当前 baseline 没有 scene/map encoding，因此 D2 没有伪造执行。",
              "- Stage C 不自动启动；即使 H1/H2 通过，也需要人工审阅后另立实验。",
              "- bootstrap 只度量固定 seed 下样本不确定性，不代表 detector 训练种子不确定性。"]
    (RESULTS / "REGIME_CHANGE_DETECTOR_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "best_detector": best, "elapsed_seconds": payload["elapsed_seconds"]}, ensure_ascii=False))


def status() -> None:
    targets = {
        "preflight": PREFLIGHT,
        "features_train": RESULTS / "features_train.npz",
        "d0": RESULTS / "d0_motion_only" / "best_dev_auroc.pth",
        "d1": RESULTS / "d1_motion_social" / "best_dev_auroc.pth",
        "analysis": RESULTS / "verdict.json",
        "report": RESULTS / "REGIME_CHANGE_DETECTOR_REPORT.md",
    }
    for name, path in targets.items():
        print(f"[{'ready' if path.is_file() else 'pending'}] {name}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "run", "status"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    spec = load_spec()
    if args.stage == "status":
        status(); return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    if args.stage == "preflight":
        preflight(spec)
    else:
        preflight(spec)
        run(device, spec)


if __name__ == "__main__":
    main()
