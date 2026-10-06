#!/usr/bin/env python3
"""Observation-only directional regime and explicit-candidate ETH pilot."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG_PATH = HERE / "config.json"
FORMAL = ROOT / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES"
FORMAL_RUNNER = FORMAL / "run_formal_regime_selection.py"
TRAIN_CACHE = FORMAL / "results" / "cache" / "train_dev" / "eth_train.npz"
DEV_CACHE = FORMAL / "results" / "cache" / "train_dev" / "eth_dev.npz"
RAW_TRAIN = Path(
    r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy"
    r"\processed_data_diverse\eth_data_train.npy"
)
FORMAL_CONFIG = FORMAL / "formal_config.json"
FORMAL_MODEL_DIR = FORMAL / "results" / "models" / "eth" / "seed_1"
RESULTS = HERE / "results"
MODELS = RESULTS / "models"
ANALYSIS = RESULTS / "analysis"
PREFLIGHT = HERE / "preflight.json"
LOCK = RESULTS / "model_lock.json"

CLASS_NAMES = ("continue", "decelerate", "stop", "accelerate", "left", "right", "reverse")
CONTINUE, DECELERATE, STOP, ACCELERATE, LEFT, RIGHT, REVERSE = range(7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "run", "status"), default="status")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if tuple(cfg["intent_classes"]) != CLASS_NAMES:
        raise RuntimeError("Intent-class contract changed")
    if cfg["test_access"] != "forbidden" or cfg["fold"] != "eth" or cfg["seed"] != 1:
        raise RuntimeError("Pilot scope contract changed")
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rotate_to_focal_heading(values: np.ndarray, past: np.ndarray) -> np.ndarray:
    displacement = past[:, -1] - past[:, -4]
    speed = np.linalg.norm(displacement, axis=-1)
    heading = np.zeros_like(displacement)
    heading[:, 0] = 1.0
    moving = speed > 0.03
    heading[moving] = displacement[moving] / speed[moving, None]
    cosine, sine = heading[:, 0], heading[:, 1]
    relative = values - past[:, -1:, :]
    x = relative[..., 0] * cosine[:, None] + relative[..., 1] * sine[:, None]
    y = -relative[..., 0] * sine[:, None] + relative[..., 1] * cosine[:, None]
    return np.stack((x, y), axis=-1).astype(np.float32)


def raw_focal_trajectories(sample_ids: np.ndarray, cached_motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(RAW_TRAIN, mmap_mode="r")
    coordinates = np.asarray(raw[sample_ids, 0, :, -2:], dtype=np.float32)
    past_world, future_world = coordinates[:, :8], coordinates[:, 8:20]
    past_local = rotate_to_focal_heading(past_world, past_world)
    future_local = rotate_to_focal_heading(future_world, past_world)
    error = float(np.max(np.abs(past_local.reshape(len(past_local), -1) - cached_motion[:, :16])))
    if error > 5e-5:
        raise RuntimeError(f"Raw/cache focal-history alignment failed: {error:.6g}")
    return past_local, future_local


def classify_velocity(observed: np.ndarray, future_mean: np.ndarray, cfg: dict) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float32)
    future_mean = np.asarray(future_mean, dtype=np.float32)
    observed_speed = np.linalg.norm(observed, axis=-1)
    future_speed = np.linalg.norm(future_mean, axis=-1)
    ratio = (future_speed + 0.01) / (observed_speed + 0.01)
    product = observed_speed * future_speed
    cosine = np.ones_like(product)
    sine = np.zeros_like(product)
    moving = product > 1e-6
    cosine[moving] = np.sum(observed[moving] * future_mean[moving], axis=-1) / product[moving]
    sine[moving] = (
        observed[moving, 0] * future_mean[moving, 1]
        - observed[moving, 1] * future_mean[moving, 0]
    ) / product[moving]
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arctan2(sine, cosine))

    labels = np.full(observed_speed.shape, CONTINUE, dtype=np.int64)
    labels[ratio <= cfg["decelerate_speed_ratio"]] = DECELERATE
    labels[ratio >= cfg["accelerate_speed_ratio"]] = ACCELERATE
    labels[angle >= cfg["turn_angle_degrees"]] = LEFT
    labels[angle <= -cfg["turn_angle_degrees"]] = RIGHT
    labels[cosine <= cfg["reverse_cosine_threshold"]] = REVERSE
    stopped = (future_speed <= cfg["stop_absolute_speed"]) | (ratio <= cfg["stop_speed_ratio"])
    labels[stopped] = STOP
    return labels


def maneuver_labels(past: np.ndarray, future: np.ndarray, cfg: dict) -> np.ndarray:
    observed = past[:, -1] - past[:, -2]
    future_mean = future[:, -1] / float(future.shape[1])
    return classify_velocity(observed, future_mean, cfg)


def _integrated_turn(base_velocity: np.ndarray, speed_factors: np.ndarray, degrees: float) -> np.ndarray:
    batch, steps = len(base_velocity), len(speed_factors)
    angles = np.deg2rad(np.linspace(degrees / steps, degrees, steps, dtype=np.float32))
    cosine, sine = np.cos(angles), np.sin(angles)
    vx, vy = base_velocity[:, 0:1], base_velocity[:, 1:2]
    rotated_x = vx * cosine[None] - vy * sine[None]
    rotated_y = vx * sine[None] + vy * cosine[None]
    velocity = np.stack((rotated_x, rotated_y), axis=-1)
    velocity = velocity * speed_factors[None, :, None]
    return np.cumsum(velocity, axis=1).astype(np.float32)


def explicit_candidates(past: np.ndarray) -> np.ndarray:
    """Eight deterministic observation-only proposals; no future argument exists."""
    last_velocity = past[:, -1] - past[:, -2]
    smooth_velocity = (past[:, -1] - past[:, -4]) / 3.0
    last_speed = np.linalg.norm(last_velocity, axis=-1)
    base = np.where((last_speed > 0.03)[:, None], last_velocity, smooth_velocity)
    base_speed = np.linalg.norm(base, axis=-1)
    fallback = np.zeros_like(base)
    fallback[:, 0] = 0.05
    base = np.where((base_speed > 0.015)[:, None], base, fallback)
    steps = 12
    ones = np.ones(steps, dtype=np.float32)
    decel = np.linspace(0.92, 0.0, steps, dtype=np.float32)
    accel = np.linspace(1.05, 1.50, steps, dtype=np.float32)
    proposals = [
        np.zeros((len(past), steps, 2), dtype=np.float32),
        _integrated_turn(base, decel, 0.0),
        _integrated_turn(base, accel, 0.0),
        _integrated_turn(base, ones, 35.0),
        _integrated_turn(base, ones, -35.0),
        _integrated_turn(base, ones, 75.0),
        _integrated_turn(base, ones, -75.0),
        _integrated_turn(base, ones, 180.0),
    ]
    return np.stack(proposals, axis=1)


def candidate_features(
    past: np.ndarray,
    relative: np.ndarray,
    logits: np.ndarray,
    intent_labels: np.ndarray,
    explicit_flag: np.ndarray,
) -> np.ndarray:
    history_velocity = past[:, -1] - past[:, -2]
    origin = np.zeros_like(relative[:, :, :1])
    velocity = np.diff(np.concatenate((origin, relative), axis=2), axis=2)
    early = velocity[:, :, :3].mean(axis=2)
    hist_speed = np.linalg.norm(history_velocity, axis=-1)[:, None]
    early_speed = np.linalg.norm(early, axis=-1)
    product = hist_speed * early_speed
    cosine = np.ones_like(product)
    valid = product > 1e-6
    dot = np.sum(early * history_velocity[:, None], axis=-1)
    cosine[valid] = np.clip(dot[valid] / product[valid], -1.0, 1.0)
    direction_mismatch = (1.0 - cosine) / 2.0
    speed_mismatch = np.abs(np.log((early_speed + 0.05) / (hist_speed + 0.05)))
    velocity_mismatch = np.linalg.norm(early - history_velocity[:, None], axis=-1)
    acceleration = velocity[:, :, 1:4] - velocity[:, :, :3]
    acceleration_mismatch = np.linalg.norm(acceleration, axis=-1).mean(-1)
    steps = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 4, 1)
    cv = steps * history_velocity[:, None, None]
    cv_deviation = np.linalg.norm(relative[:, :, :4] - cv, axis=-1).mean(-1)
    v0, v1 = velocity[:, :, :3], velocity[:, :, 1:4]
    denominator = np.linalg.norm(v0, axis=-1) * np.linalg.norm(v1, axis=-1)
    turn_cosine = np.ones_like(denominator)
    valid_turn = denominator > 1e-6
    turn_dot = np.sum(v0 * v1, axis=-1)
    turn_cosine[valid_turn] = np.clip(turn_dot[valid_turn] / denominator[valid_turn], -1.0, 1.0)
    curvature = (1.0 - turn_cosine).mean(-1)
    mismatch = np.stack((
        velocity_mismatch, direction_mismatch, speed_mismatch,
        acceleration_mismatch, cv_deviation, curvature,
    ), axis=-1)
    centered_logits = logits - logits.mean(axis=-1, keepdims=True)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float32)[intent_labels]
    features = np.concatenate((
        relative.reshape(*relative.shape[:2], -1), mismatch,
        centered_logits[..., None], one_hot, explicit_flag[..., None],
    ), axis=-1)
    return features.astype(np.float32)


def candidate_intent_labels(past: np.ndarray, trajectories: np.ndarray, cfg: dict) -> np.ndarray:
    observed = np.repeat((past[:, -1] - past[:, -2])[:, None], trajectories.shape[1], axis=1)
    future_mean = trajectories[:, :, -1] / float(trajectories.shape[2])
    return classify_velocity(observed.reshape(-1, 2), future_mean.reshape(-1, 2), cfg).reshape(
        trajectories.shape[:2]
    )


def build_candidate_set(
    cache: dict[str, np.ndarray], past: np.ndarray, future: np.ndarray, cfg: dict, augmented: bool,
) -> dict[str, np.ndarray]:
    trajectories = cache["candidate"][..., :24].reshape(-1, 20, 12, 2).astype(np.float32)
    logits = cache["logits"].astype(np.float32)
    explicit_flag = np.zeros(logits.shape, dtype=np.float32)
    if augmented:
        proposals = explicit_candidates(past)
        row_std = logits.std(axis=1, keepdims=True)
        prior = np.median(logits, axis=1, keepdims=True) - cfg["explicit_prior_std_offset"] * row_std
        proposal_logits = np.repeat(prior.astype(np.float32), proposals.shape[1], axis=1)
        trajectories = np.concatenate((trajectories, proposals), axis=1)
        logits = np.concatenate((logits, proposal_logits), axis=1)
        explicit_flag = np.concatenate((
            explicit_flag, np.ones((len(past), proposals.shape[1]), dtype=np.float32)
        ), axis=1)
    errors = np.linalg.norm(trajectories - future[:, None], axis=-1)
    ade, fde = errors.mean(-1).astype(np.float32), errors[..., -1].astype(np.float32)
    if not augmented:
        if float(np.max(np.abs(ade - cache["ade"]))) > 2e-5 or float(np.max(np.abs(fde - cache["fde"]))) > 2e-5:
            raise RuntimeError("Base-candidate error audit failed")
    labels = candidate_intent_labels(past, trajectories, cfg)
    features = candidate_features(past, trajectories, logits, labels, explicit_flag)
    return {
        "trajectory": trajectories, "logits": logits, "ade": ade, "fde": fde,
        "features": features, "intent_label": labels, "explicit_flag": explicit_flag,
    }


class IntentClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, len(CLASS_NAMES)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class DirectionalHead(nn.Module):
    def __init__(
        self, observation_dim: int, candidate_dim: int, hidden: int, dropout: float,
        attenuate: bool, beta: float, floor: float,
    ) -> None:
        super().__init__()
        self.attenuate = attenuate
        self.beta = float(beta)
        self.floor = float(floor)
        self.network = nn.Sequential(
            nn.Linear(observation_dim + candidate_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.last_residual: torch.Tensor | None = None

    def forward(
        self, observation: torch.Tensor, candidate: torch.Tensor,
        native_logits: torch.Tensor, intent_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        repeated = observation[:, None].expand(-1, candidate.shape[1], -1)
        residual = self.network(torch.cat((repeated, candidate), dim=-1)).squeeze(-1)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        self.last_residual = residual
        centered = native_logits - native_logits.mean(dim=-1, keepdim=True)
        if self.attenuate:
            entropy = -(
                intent_probability.clamp_min(1e-8) * intent_probability.clamp_min(1e-8).log()
            ).sum(-1) / math.log(intent_probability.shape[-1])
            risk = 1.0 - intent_probability[:, CONTINUE]
            scale = 1.0 - self.beta * risk * (0.5 + 0.5 * entropy)
            scale = scale.clamp(self.floor, 1.0)
        else:
            scale = torch.ones(len(observation), device=observation.device, dtype=observation.dtype)
        return scale[:, None] * centered + residual, scale


def macro_f1(labels: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for class_id in range(len(CLASS_NAMES)):
        tp = np.sum((labels == class_id) & (prediction == class_id))
        fp = np.sum((labels != class_id) & (prediction == class_id))
        fn = np.sum((labels == class_id) & (prediction != class_id))
        scores.append((2.0 * tp) / max(2.0 * tp + fp + fn, 1.0))
    return float(np.mean(scores))


def normalized_observation(cache: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raw = np.concatenate((cache["motion"], cache["social"]), axis=1).astype(np.float32)
    return ((raw - mean) / std).astype(np.float32)


def predict_intent(model: IntentClassifier, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval(); output = []
    with torch.inference_mode():
        for start in range(0, len(values), 2048):
            output.append(torch.softmax(model(torch.from_numpy(values[start:start + 2048]).to(device)), -1).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def train_intent(
    cache: dict[str, np.ndarray], labels: np.ndarray, fit: np.ndarray, calibration: np.ndarray,
    cfg: dict, device: torch.device,
) -> tuple[IntentClassifier, dict]:
    destination = MODELS / "intent_classifier.pth"
    raw = np.concatenate((cache["motion"], cache["social"]), axis=1).astype(np.float32)
    if destination.is_file():
        saved = torch.load(destination, map_location="cpu", weights_only=False)
        model = IntentClassifier(saved["input_dim"], cfg["intent_dropout"])
        model.load_state_dict(saved["state_dict"]); model.to(device).eval()
        return model, saved
    mean = raw[fit].mean(0, keepdims=True).astype(np.float32)
    std = raw[fit].std(0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    values = ((raw - mean) / std).astype(np.float32)
    counts = np.bincount(labels[fit], minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    model = IntentClassifier(values.shape[1], cfg["intent_dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["intent_learning_rate"],
                                  weight_decay=cfg["intent_weight_decay"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values[fit]), torch.from_numpy(labels[fit])),
        batch_size=cfg["intent_batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]), num_workers=0,
    )
    best_key, best_state, best_epoch, history = (-math.inf, math.inf), None, -1, []
    class_weight = torch.from_numpy(weights).to(device)
    for epoch in range(1, cfg["intent_epochs"] + 1):
        model.train(); losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, weight=class_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probability = predict_intent(model, values[calibration], device)
        prediction = probability.argmax(1)
        f1 = macro_f1(labels[calibration], prediction)
        ce = float(-np.log(probability[np.arange(len(calibration)), labels[calibration]].clip(1e-8)).mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "cal_macro_f1": f1, "cal_ce": ce})
        key = (f1, -ce)
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    saved = {
        "state_dict": best_state, "mean": mean, "std": std, "input_dim": values.shape[1],
        "best_epoch": best_epoch, "best_cal_macro_f1": best_key[0], "class_counts_fit": counts,
        "formal_dev_accessed": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, destination); write_csv(MODELS / "intent_history.csv", history)
    model.load_state_dict(best_state); model.to(device).eval()
    return model, saved


def scale_candidates(features: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[fit].mean((0, 1), keepdims=True).astype(np.float32)
    std = features[fit].std((0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((features - mean) / std).astype(np.float32), mean, std


def selection_arrays(logits: np.ndarray, candidate_set: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    selected = logits.argmax(1); row = np.arange(len(logits)); oracle = candidate_set["ade"].argmin(1)
    return {
        "selected_index": selected.astype(np.int64),
        "selected_ade": candidate_set["ade"][row, selected].astype(np.float64),
        "selected_fde": candidate_set["fde"][row, selected].astype(np.float64),
        "oracle_index": oracle.astype(np.int64),
        "oracle_ade": candidate_set["ade"][row, oracle].astype(np.float64),
        "oracle_fde": candidate_set["fde"][row, oracle].astype(np.float64),
    }


def predict_head(
    model: DirectionalHead, observation: np.ndarray, candidate: np.ndarray,
    native: np.ndarray, probability: np.ndarray, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); logits, scales = [], []
    with torch.inference_mode():
        for start in range(0, len(observation), 1024):
            output, scale = model(
                torch.from_numpy(observation[start:start + 1024]).to(device),
                torch.from_numpy(candidate[start:start + 1024]).to(device),
                torch.from_numpy(native[start:start + 1024]).to(device),
                torch.from_numpy(probability[start:start + 1024]).to(device),
            )
            logits.append(output.cpu().numpy()); scales.append(scale.cpu().numpy())
    return np.concatenate(logits), np.concatenate(scales)


def train_head(
    name: str, attenuate: bool, cache: dict[str, np.ndarray], candidate_set: dict[str, np.ndarray],
    observation: np.ndarray, probability: np.ndarray, fit: np.ndarray, calibration: np.ndarray,
    cfg: dict, device: torch.device,
) -> tuple[DirectionalHead, dict]:
    destination = MODELS / f"{name}.pth"
    if destination.is_file():
        saved = torch.load(destination, map_location="cpu", weights_only=False)
        model = DirectionalHead(
            saved["observation_dim"], saved["candidate_dim"], cfg["reranker_hidden_dim"],
            cfg["reranker_dropout"], saved["attenuate"], cfg["native_attenuation_beta"],
            cfg["native_attenuation_floor"],
        )
        model.load_state_dict(saved["state_dict"]); model.to(device).eval()
        return model, saved
    scaled, candidate_mean, candidate_std = scale_candidates(candidate_set["features"], fit)
    quality = candidate_set["ade"] + cfg["quality_fde_weight"] * candidate_set["fde"]
    target = torch.softmax(torch.from_numpy(-quality / cfg["soft_target_temperature"]), -1).numpy().astype(np.float32)
    model = DirectionalHead(
        observation.shape[1], scaled.shape[2], cfg["reranker_hidden_dim"], cfg["reranker_dropout"],
        attenuate, cfg["native_attenuation_beta"], cfg["native_attenuation_floor"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["reranker_learning_rate"],
                                  weight_decay=cfg["reranker_weight_decay"])
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(observation[fit]), torch.from_numpy(scaled[fit]),
            torch.from_numpy(candidate_set["logits"][fit]), torch.from_numpy(probability[fit]),
            torch.from_numpy(target[fit]),
        ),
        batch_size=cfg["reranker_batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]), num_workers=0,
    )
    best_key, best_state, best_epoch, history = (math.inf, math.inf), None, -1, []
    for epoch in range(1, cfg["reranker_epochs"] + 1):
        model.train(); losses = []
        for obs, cand, native, prob, target_prob in loader:
            obs, cand, native = obs.to(device), cand.to(device), native.to(device)
            prob, target_prob = prob.to(device), target_prob.to(device)
            output, _ = model(obs, cand, native, prob)
            loss = -(target_prob * F.log_softmax(output, -1)).sum(-1).mean()
            loss = loss + cfg["residual_l2_weight"] * model.last_residual.square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        cal_logits, cal_scale = predict_head(
            model, observation[calibration], scaled[calibration],
            candidate_set["logits"][calibration], probability[calibration], device,
        )
        arrays = selection_arrays(cal_logits, {key: value[calibration] for key, value in candidate_set.items()})
        ade, fde = float(arrays["selected_ade"].mean()), float(arrays["selected_fde"].mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "cal_selected_ade": ade, "cal_selected_fde": fde,
                        "cal_native_scale": float(cal_scale.mean())})
        key = (ade, fde)
        if key < best_key:
            best_key, best_epoch = key, epoch
            best_state = {param: value.detach().cpu().clone() for param, value in model.state_dict().items()}
    saved = {
        "state_dict": best_state, "candidate_mean": candidate_mean, "candidate_std": candidate_std,
        "observation_dim": observation.shape[1], "candidate_dim": scaled.shape[2],
        "attenuate": attenuate, "best_epoch": best_epoch,
        "best_cal_selected_ade": best_key[0], "best_cal_selected_fde": best_key[1],
        "formal_dev_accessed": False, "candidate_count": candidate_set["logits"].shape[1],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, destination); write_csv(MODELS / f"{name}_history.csv", history)
    model.load_state_dict(best_state); model.to(device).eval()
    return model, saved


def transform_candidate(features: np.ndarray, saved: dict) -> np.ndarray:
    return ((features - saved["candidate_mean"]) / saved["candidate_std"]).astype(np.float32)


def metric_rows(name: str, arrays: dict[str, np.ndarray], groups: np.ndarray, scale: np.ndarray | None = None) -> list[dict]:
    rows = []
    for group_name, mask in (
        ("overall", np.ones(len(groups), dtype=bool)),
        ("stable", groups == 0), ("transition", groups == 1), ("change", groups == 2),
    ):
        if not np.any(mask):
            continue
        rows.append({
            "arm": name, "group": group_name, "n": int(mask.sum()),
            "selected_ade": float(arrays["selected_ade"][mask].mean()),
            "selected_fde": float(arrays["selected_fde"][mask].mean()),
            "minade": float(arrays["oracle_ade"][mask].mean()),
            "minfde_at_minade": float(arrays["oracle_fde"][mask].mean()),
            "ade_regret": float((arrays["selected_ade"][mask] - arrays["oracle_ade"][mask]).mean()),
            "native_scale": float(scale[mask].mean()) if scale is not None else 1.0,
        })
    return rows


def paired_bootstrap_delta(
    treatment: np.ndarray, reference: np.ndarray, repetitions: int, seed: int,
) -> tuple[float, float, float]:
    difference = np.asarray(treatment, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 250):
        count = min(250, repetitions - start)
        indices = rng.integers(0, len(difference), size=(count, len(difference)))
        values[start:start + count] = difference[indices].mean(axis=1)
    low, high = np.quantile(values, (0.025, 0.975))
    return float(difference.mean()), float(low), float(high)


def scalar_d1_reference(dev: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    runner = load_module("directional_pilot_formal_reference", FORMAL_RUNNER)
    formal_cfg = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    detector = FORMAL_MODEL_DIR / "d1_social.pth"
    head_path = FORMAL_MODEL_DIR / "d1_social_gate_best.pth"
    probability = runner.detector_probability(dev, detector, device)
    head, saved = runner._load_head(head_path, formal_cfg, device)
    return runner._predict_head(head, saved, dev, probability, device)


def evaluate_and_report(
    cfg: dict, dev: dict[str, np.ndarray], dev_base: dict[str, np.ndarray],
    dev_aug: dict[str, np.ndarray], intent_model: IntentClassifier, intent_saved: dict,
    heads: dict[str, tuple[DirectionalHead, dict]], device: torch.device,
) -> dict:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    obs_normalized = normalized_observation(dev, intent_saved["mean"], intent_saved["std"])
    probability = predict_intent(intent_model, obs_normalized, device)
    observation = np.concatenate((obs_normalized, probability), axis=1).astype(np.float32)
    groups = dev["change_group"]
    rows, per_sample = [], {}

    native_arrays = selection_arrays(dev_base["logits"], dev_base)
    rows += metric_rows("native_base20", native_arrays, groups)
    per_sample["native_base20"] = native_arrays

    d1_logits = scalar_d1_reference(dev, device)
    d1_arrays = selection_arrays(d1_logits, dev_base)
    rows += metric_rows("scalar_d1_reference_base20", d1_arrays, groups)
    per_sample["scalar_d1_reference_base20"] = d1_arrays

    for name, candidate_set in (
        ("directional_base20", dev_base),
        ("directional_attenuated_base20", dev_base),
        ("directional_attenuated_aug28", dev_aug),
    ):
        model, saved = heads[name]
        candidate = transform_candidate(candidate_set["features"], saved)
        logits, scale = predict_head(model, observation, candidate, candidate_set["logits"], probability, device)
        arrays = selection_arrays(logits, candidate_set)
        rows += metric_rows(name, arrays, groups, scale)
        per_sample[name] = arrays
        np.savez_compressed(
            ANALYSIS / f"{name}_per_sample.npz", sample_id=dev["sample_id"],
            intent_probability=probability, native_scale=scale, logits=logits, **arrays,
        )

    write_csv(ANALYSIS / "metrics.csv", rows)
    overall = {row["arm"]: row for row in rows if row["group"] == "overall"}
    native = overall["native_base20"]
    directional = overall["directional_base20"]
    attenuated = overall["directional_attenuated_base20"]
    augmented = overall["directional_attenuated_aug28"]
    bootstrap_specs = [
        ("directional_vs_native_ade", per_sample["directional_base20"]["selected_ade"], native_arrays["selected_ade"]),
        ("directional_vs_native_fde", per_sample["directional_base20"]["selected_fde"], native_arrays["selected_fde"]),
        ("attenuated_vs_native_ade", per_sample["directional_attenuated_base20"]["selected_ade"], native_arrays["selected_ade"]),
        ("attenuated_vs_native_fde", per_sample["directional_attenuated_base20"]["selected_fde"], native_arrays["selected_fde"]),
        ("augmented_vs_attenuated_ade", per_sample["directional_attenuated_aug28"]["selected_ade"], per_sample["directional_attenuated_base20"]["selected_ade"]),
        ("augmented_vs_attenuated_fde", per_sample["directional_attenuated_aug28"]["selected_fde"], per_sample["directional_attenuated_base20"]["selected_fde"]),
        ("augmented_vs_base_oracle_ade", per_sample["directional_attenuated_aug28"]["oracle_ade"], per_sample["directional_attenuated_base20"]["oracle_ade"]),
    ]
    bootstrap_rows = []
    for offset, (name, treatment, reference) in enumerate(bootstrap_specs):
        delta, low, high = paired_bootstrap_delta(
            treatment, reference, cfg["bootstrap_repetitions"], cfg["bootstrap_seed"] + offset,
        )
        bootstrap_rows.append({"comparison": name, "delta": delta, "ci95_low": low, "ci95_high": high})
    write_csv(ANALYSIS / "paired_bootstrap.csv", bootstrap_rows)
    augmented_arrays = per_sample["directional_attenuated_aug28"]
    explicit_diagnostics = {
        "explicit_oracle_n": int((augmented_arrays["oracle_index"] >= 20).sum()),
        "explicit_oracle_fraction": float((augmented_arrays["oracle_index"] >= 20).mean()),
        "explicit_selected_n": int((augmented_arrays["selected_index"] >= 20).sum()),
        "explicit_selected_fraction": float((augmented_arrays["selected_index"] >= 20).mean()),
        "oracle_ade_gain": float(per_sample["directional_attenuated_base20"]["oracle_ade"].mean() - augmented_arrays["oracle_ade"].mean()),
        "oracle_ade_gain_fraction": float(1.0 - augmented_arrays["oracle_ade"].mean() / per_sample["directional_attenuated_base20"]["oracle_ade"].mean()),
    }
    gates = {
        "directional_selection_gain": (
            directional["selected_ade"] <= native["selected_ade"] * (1.0 - cfg["minimum_selection_gain_fraction"])
        ),
        "attenuation_beats_plain_directional": attenuated["selected_ade"] < directional["selected_ade"],
        "attenuated_fde_guard": (
            attenuated["selected_fde"] <= native["selected_fde"] * (1.0 + cfg["maximum_fde_degradation_fraction"])
        ),
        "augmented_oracle_gain": (
            augmented["minade"] <= native["minade"] * (1.0 - cfg["minimum_augmented_oracle_gain_fraction"])
        ),
        "augmented_selection_gain": augmented["selected_ade"] < attenuated["selected_ade"],
    }
    if gates["directional_selection_gain"] and gates["attenuation_beats_plain_directional"] and gates["attenuated_fde_guard"]:
        selection_verdict = "DIRECTIONAL_SELECTION_SIGNAL"
    else:
        selection_verdict = "NO_DIRECTIONAL_SELECTION_SIGNAL"
    generator_verdict = "EXPLICIT_CANDIDATE_COVERAGE_SIGNAL" if gates["augmented_oracle_gain"] else "NO_EXPLICIT_CANDIDATE_COVERAGE_SIGNAL"
    verdict = (
        "ADVANCE_COMBINED_DIRECTIONAL_DESIGN"
        if all(gates.values()) else f"PILOT_REVIEW_REQUIRED__{selection_verdict}__{generator_verdict}"
    )
    intent_prediction = probability.argmax(1)
    dev_past, dev_future = raw_focal_trajectories(dev["sample_id"], dev["motion"])
    dev_labels = maneuver_labels(dev_past, dev_future, cfg)
    intent_metrics = {
        "accuracy": float((intent_prediction == dev_labels).mean()),
        "macro_f1": macro_f1(dev_labels, intent_prediction),
        "class_counts": np.bincount(dev_labels, minlength=len(CLASS_NAMES)).tolist(),
        "confusion": np.asarray([
            [int(np.sum((dev_labels == i) & (intent_prediction == j))) for j in range(len(CLASS_NAMES))]
            for i in range(len(CLASS_NAMES))
        ]).tolist(),
    }
    summary = {
        "protocol": cfg["protocol"], "fold": "eth", "seed": 1,
        "formal_dev_n": len(dev_labels), "intent_dev": intent_metrics,
        "overall": overall, "paired_bootstrap": bootstrap_rows,
        "explicit_candidate_diagnostics": explicit_diagnostics,
        "gates": gates, "verdict": verdict,
        "scope_warning": "Single-fold single-seed formal-dev pilot; no held-out test was accessed.",
    }
    (ANALYSIS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Directional regime + explicit candidate pilot result", "",
        f"- Verdict: `{verdict}`", "- Fold/seed: ETH / 1", "- Held-out test access: none",
        f"- Intent dev accuracy / macro-F1: {intent_metrics['accuracy']:.4f} / {intent_metrics['macro_f1']:.4f}", "",
        "| Arm | Selected ADE | Selected FDE | minADE | ADE regret | Native scale |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("native_base20", "scalar_d1_reference_base20", "directional_base20",
                 "directional_attenuated_base20", "directional_attenuated_aug28"):
        row = overall[name]
        lines.append(
            f"| {name} | {row['selected_ade']:.6f} | {row['selected_fde']:.6f} | "
            f"{row['minade']:.6f} | {row['ade_regret']:.6f} | {row['native_scale']:.4f} |"
        )
    lines += ["", "## Gates", ""] + [f"- {name}: {value}" for name, value in gates.items()]
    lines += [
        "", "## Explicit-candidate diagnostic", "",
        f"- Explicit proposal is oracle for {explicit_diagnostics['explicit_oracle_n']}/{len(dev_labels)} samples "
        f"({explicit_diagnostics['explicit_oracle_fraction']:.2%}).",
        f"- The selector chooses an explicit proposal for {explicit_diagnostics['explicit_selected_n']}/{len(dev_labels)} samples.",
        f"- Oracle ADE gain: {explicit_diagnostics['oracle_ade_gain']:.6f} "
        f"({explicit_diagnostics['oracle_ade_gain_fraction']:.2%}).", "",
        "## Paired bootstrap deltas (treatment - reference)", "",
    ]
    for row in bootstrap_rows:
        lines.append(
            f"- {row['comparison']}: {row['delta']:+.6f}, 95% CI "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]"
        )
    lines += ["", "This pilot cannot authorize test evaluation or seed/fold expansion without review.", ""]
    (ANALYSIS / "PILOT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def internal_split(length: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    calibration_n = max(1, int(round(length * fraction)))
    fit = np.arange(0, length - calibration_n, dtype=np.int64)
    calibration = np.arange(length - calibration_n, length, dtype=np.int64)
    if np.intersect1d(fit, calibration).size or len(fit) + len(calibration) != length:
        raise RuntimeError("Internal split failure")
    return fit, calibration


def preflight(cfg: dict, device: torch.device) -> None:
    required = [CONFIG_PATH, FORMAL_RUNNER, TRAIN_CACHE, DEV_CACHE, RAW_TRAIN, FORMAL_CONFIG,
                FORMAL_MODEL_DIR / "d1_social.pth", FORMAL_MODEL_DIR / "d1_social_gate_best.pth", Path(__file__)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    train = load_npz(TRAIN_CACHE)
    past, future = raw_focal_trajectories(train["sample_id"], train["motion"])
    fit, calibration = internal_split(len(train["sample_id"]), cfg["internal_calibration_fraction"])
    labels = maneuver_labels(past, future, cfg)
    fit_counts = np.bincount(labels[fit], minlength=len(CLASS_NAMES))
    cal_counts = np.bincount(labels[calibration], minlength=len(CLASS_NAMES))
    if np.any(fit_counts < 20) or np.any(cal_counts < 5):
        raise RuntimeError(f"Insufficient intent class support: fit={fit_counts}, cal={cal_counts}")
    proposals = explicit_candidates(past[:8])
    if proposals.shape != (8, 8, 12, 2) or not np.isfinite(proposals).all():
        raise RuntimeError("Explicit-candidate smoke failed")
    report = {
        "status": "PREFLIGHT_PASS_NO_TRAINING", "device": str(device),
        "cuda_available": torch.cuda.is_available(), "train_n": len(train["sample_id"]),
        "fit_n": len(fit), "calibration_n": len(calibration),
        "fit_class_counts": dict(zip(CLASS_NAMES, fit_counts.tolist())),
        "calibration_class_counts": dict(zip(CLASS_NAMES, cal_counts.tolist())),
        "test_access": False, "formal_dev_used_for_selection": False,
        "source_hashes": {str(path.resolve()): sha256(path) for path in required if path.suffix != ".npy"},
    }
    PREFLIGHT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def run(cfg: dict, device: torch.device) -> None:
    seed_everything(cfg["seed"])
    if not PREFLIGHT.is_file():
        raise RuntimeError("Run preflight first")
    train = load_npz(TRAIN_CACHE)
    train_past, train_future = raw_focal_trajectories(train["sample_id"], train["motion"])
    fit, calibration = internal_split(len(train["sample_id"]), cfg["internal_calibration_fraction"])
    labels = maneuver_labels(train_past, train_future, cfg)
    print(f"[intent] fit={len(fit)} calibration={len(calibration)}", flush=True)
    intent_model, intent_saved = train_intent(train, labels, fit, calibration, cfg, device)
    obs_normalized = normalized_observation(train, intent_saved["mean"], intent_saved["std"])
    probability = predict_intent(intent_model, obs_normalized, device)
    observation = np.concatenate((obs_normalized, probability), axis=1).astype(np.float32)

    print("[candidate cache] base20", flush=True)
    train_base = build_candidate_set(train, train_past, train_future, cfg, augmented=False)
    print("[candidate cache] augmented28", flush=True)
    train_aug = build_candidate_set(train, train_past, train_future, cfg, augmented=True)
    heads = {}
    for name, attenuate, candidate_set in (
        ("directional_base20", False, train_base),
        ("directional_attenuated_base20", True, train_base),
        ("directional_attenuated_aug28", True, train_aug),
    ):
        print(f"[reranker] {name}", flush=True)
        heads[name] = train_head(
            name, attenuate, train, candidate_set, observation, probability,
            fit, calibration, cfg, device,
        )
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({
        "locked": True, "formal_dev_accessed_before_lock": False, "test_accessed": False,
        "models": ["intent_classifier", *heads], "locked_at_unix": time.time(),
    }, indent=2), encoding="utf-8")

    print("[formal dev] checkpoints locked; loading diagnostic dev once", flush=True)
    dev = load_npz(DEV_CACHE)
    dev_past, dev_future = raw_focal_trajectories(dev["sample_id"], dev["motion"])
    dev_base = build_candidate_set(dev, dev_past, dev_future, cfg, augmented=False)
    dev_aug = build_candidate_set(dev, dev_past, dev_future, cfg, augmented=True)
    summary = evaluate_and_report(cfg, dev, dev_base, dev_aug, intent_model, intent_saved, heads, device)
    print(json.dumps(summary, indent=2), flush=True)


def status() -> None:
    paths = [PREFLIGHT, MODELS / "intent_classifier.pth", MODELS / "directional_base20.pth",
             MODELS / "directional_attenuated_base20.pth", MODELS / "directional_attenuated_aug28.pth",
             LOCK, ANALYSIS / "summary.json", ANALYSIS / "PILOT_REPORT.md"]
    for path in paths:
        print(f"{'READY' if path.is_file() else 'PENDING':7s} {path}")


def main() -> None:
    args = parse_args(); cfg = load_config(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.action == "preflight":
        preflight(cfg, device)
    elif args.action == "run":
        run(cfg, device)
    else:
        status()


if __name__ == "__main__":
    main()
