"""Five-fold formal test of observation-only regime-adaptive candidate selection."""
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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASTRA = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
DATA_ROOT = ASTRA / "datasets" / "eth_ucy" / "processed_data_diverse"
CONFIG = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
BASE_ROOT = (
    ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"
    / "EXP_156B_FOLD_LOCAL_CURRENT_DECODER" / "results"
)
REGIME_RUNNER = ROOT / "REGIME_CHANGE_DETECTOR_SERIES" / "run_detector_experiment.py"
FRONT_MODELS = ROOT / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
FRONT_TRAINER = ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
V2_ADAPTER = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
SPEC_PATH = HERE / "formal_config.json"
PREFLIGHT = HERE / "preflight.json"
RESULTS = HERE / "results"
TRAIN_CACHE = RESULTS / "cache" / "train_dev"
TEST_CACHE = RESULTS / "cache" / "test"
MODELS = RESULTS / "models"
EVALUATION = RESULTS / "evaluation"
ANALYSIS = RESULTS / "analysis"


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
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


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fold_data(fold: str, split: str) -> tuple[Path, Path]:
    return (
        DATA_ROOT / f"{fold}_data_{split}.npy",
        DATA_ROOT / f"{fold}_num_{split}.npy",
    )


def fold_run(fold: str, stage: str = "v2") -> Path:
    return BASE_ROOT / fold / stage / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"


def model_dir(fold: str, seed: int) -> Path:
    return MODELS / fold / f"seed_{seed}"


def cache_path(fold: str, split: str) -> Path:
    root = TEST_CACHE if split == "test" else TRAIN_CACHE
    return root / f"{fold}_{split}.npz"


def setup_backend(device: torch.device):
    os.chdir(ASTRA)
    for path in (ASTRA, ASTRA / "scripts", ROOT / "FRONT_ENCODER_SERIES"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
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
    regime = load_module("formal_regime_base", REGIME_RUNNER)
    front = load_module("formal_front", FRONT_MODELS)
    trainer = load_module("formal_trainer", FRONT_TRAINER)
    adapter = load_module("formal_v2_adapter", V2_ADAPTER)
    return cfg, regime, front, trainer, adapter


def build_baseline(fold: str, cfg, front, adapter, device: torch.device):
    h3 = fold_run(fold, "h3") / "best_selected.pth"
    v2 = fold_run(fold, "v2") / "best_selected.pth"
    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3)
    os.environ["EXP173_DECODER_INIT_SEED"] = "1"
    model = front.FrontEncoderASTRA(copy.deepcopy(cfg), "f16_temporal_social_cross")
    # The fold-local result metadata uses the short V2 experiment label, while
    # the adapter registry exposes the same architecture under this full name.
    adapter.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
    model.load_state_dict(torch.load(v2, map_location="cpu", weights_only=True), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()


def candidate_features(past: torch.Tensor, relative: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Observation-derived history/candidate features; no future argument exists."""
    focal = past[:, 0]
    history_velocity = focal[:, -1] - focal[:, -2]
    origin = torch.zeros_like(relative[:, :, :1])
    velocity = torch.diff(torch.cat((origin, relative), dim=2), dim=2)
    early = velocity[:, :, :3].mean(dim=2)
    hist_speed = torch.linalg.vector_norm(history_velocity, dim=-1)[:, None]
    early_speed = torch.linalg.vector_norm(early, dim=-1)
    product = hist_speed * early_speed
    cosine = (early * history_velocity[:, None]).sum(-1) / product.clamp_min(1e-6)
    cosine = torch.where(product > 1e-6, cosine.clamp(-1, 1), torch.ones_like(cosine))
    direction_mismatch = (1.0 - cosine) / 2.0
    speed_mismatch = torch.abs(torch.log((early_speed + 0.05) / (hist_speed + 0.05)))
    velocity_mismatch = torch.linalg.vector_norm(early - history_velocity[:, None], dim=-1)
    acceleration = velocity[:, :, 1:4] - velocity[:, :, :3]
    acceleration_mismatch = torch.linalg.vector_norm(acceleration, dim=-1).mean(-1)
    steps = torch.arange(1, 5, device=past.device, dtype=past.dtype).view(1, 1, 4, 1)
    cv = steps * history_velocity[:, None, None]
    cv_deviation = torch.linalg.vector_norm(relative[:, :, :4] - cv, dim=-1).mean(-1)
    v0, v1 = velocity[:, :, :3], velocity[:, :, 1:4]
    denom = (torch.linalg.vector_norm(v0, dim=-1) * torch.linalg.vector_norm(v1, dim=-1)).clamp_min(1e-6)
    turn = 1.0 - ((v0 * v1).sum(-1) / denom).clamp(-1, 1)
    curvature = turn.mean(-1)
    mismatch = torch.stack((
        velocity_mismatch, direction_mismatch, speed_mismatch,
        acceleration_mismatch, cv_deviation, curvature,
    ), dim=-1)
    centered_logits = logits - logits.mean(dim=-1, keepdim=True)
    return torch.cat((relative.flatten(2), mismatch, centered_logits[..., None]), dim=-1)


class HistoryCorrectionHead(nn.Module):
    def __init__(self, motion_dim: int, candidate_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(motion_dim + candidate_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.last_residual: torch.Tensor | None = None

    def forward(
        self, motion: torch.Tensor, candidate: torch.Tensor,
        native_logits: torch.Tensor, gate: torch.Tensor,
    ) -> torch.Tensor:
        repeated = motion[:, None].expand(-1, candidate.shape[1], -1)
        residual = self.network(torch.cat((repeated, candidate), dim=-1)).squeeze(-1)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        self.last_residual = residual
        return native_logits + gate[:, None] * residual


def _split_ids(fold: str) -> Dict[str, np.ndarray]:
    with np.load(fold_run(fold) / "split_indices.npz", allow_pickle=False) as payload:
        return {
            "train": np.asarray(payload["train"], dtype=np.int64),
            "dev": np.asarray(payload["validation"], dtype=np.int64),
        }


def _extract_cache(
    fold: str, split: str, ids: np.ndarray, dataset, model, trainer, regime,
    device: torch.device, batch_size: int, spec: dict,
) -> dict:
    loader = DataLoader(Subset(dataset, ids.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    keys = ("motion", "social", "candidate", "logits", "ade", "fde", "change_score", "change_group")
    parts: Dict[str, list] = {key: [] for key in keys}
    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            past, future = past.to(device), future.to(device)
            num_valid = num_valid.to(device).int()
            past_local, future_local = trainer.normalize_to_focal_heading(past, future)
            center = past_local.mean(dim=-2, keepdim=True)
            prediction = model(
                past_local - center, None, None, mode="testing", num_valid=num_valid,
                graph_loc=past_local,
            )[2] + center.unsqueeze(-3)
            logits = model.last_mode_logits[:, 0]
            candidates = prediction[:, 0]
            relative = candidates - past_local[:, 0, -1, None, None]
            errors = torch.linalg.vector_norm(candidates - future_local[:, 0, None], dim=-1)
            score = regime.future_change_score(past_local, future_local)
            parts["motion"].append(regime.motion_features(past_local).cpu().numpy().astype(np.float32))
            parts["social"].append(
                model.agent_aware_trans_encoder.last_agent_features[:, 0].cpu().numpy().astype(np.float32)
            )
            parts["candidate"].append(candidate_features(past_local, relative, logits).cpu().numpy().astype(np.float32))
            parts["logits"].append(logits.cpu().numpy().astype(np.float32))
            parts["ade"].append(errors.mean(-1).cpu().numpy().astype(np.float32))
            parts["fde"].append(errors[..., -1].cpu().numpy().astype(np.float32))
            score_np = score.cpu().numpy().astype(np.float32)
            parts["change_score"].append(score_np)
            parts["change_group"].append(regime.group_from_score(score_np, spec))
    result = {key: np.concatenate(value) for key, value in parts.items()}
    result["sample_id"] = ids.astype(np.int64, copy=True)
    if len(result["sample_id"]) != len(ids):
        raise RuntimeError(f"{fold}/{split} cache count mismatch")
    destination = cache_path(fold, split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **result)
    return result


def load_cache(fold: str, split: str) -> dict:
    path = cache_path(fold, split)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def prepare_train(spec: dict, device: torch.device, force: bool = False) -> None:
    preflight(spec)
    cfg, regime, front, trainer, adapter = setup_backend(device)
    for fold in spec["folds"]:
        paths = [cache_path(fold, "train"), cache_path(fold, "dev")]
        if all(path.is_file() for path in paths) and not force:
            print(f"[skip train/dev cache] {fold}", flush=True)
            continue
        data_path, num_path = fold_data(fold, "train")
        dataset = trainer.TrajectoryDataset(data_path, num_path, cfg)
        model = build_baseline(fold, cfg, front, adapter, device)
        for split, ids in _split_ids(fold).items():
            if cache_path(fold, split).is_file() and not force:
                continue
            print(f"[extract {fold}/{split}] {len(ids)}", flush=True)
            _extract_cache(fold, split, ids, dataset, model, trainer, regime, device,
                           spec["feature_batch_size"], spec)
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()


def standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    std = train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((values - mean) / std).astype(np.float32), mean, std


def _binary(features: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = groups != 1
    return features[keep], (groups[keep] == 2).astype(np.float32)


def _temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    temperatures = np.exp(np.linspace(-2.0, 2.0, 161))
    losses = []
    for temp in temperatures:
        scaled = logits / temp
        losses.append(float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled)))
    return float(temperatures[int(np.argmin(losses))])


def train_detector(
    fold: str, seed: int, variant: str, train: dict, dev: dict,
    spec: dict, device: torch.device,
) -> Path:
    destination = model_dir(fold, seed) / f"{variant}.pth"
    if destination.is_file():
        return destination
    train_x = train["motion"] if variant == "d0_motion" else np.concatenate((train["motion"], train["social"]), 1)
    dev_x = dev["motion"] if variant == "d0_motion" else np.concatenate((dev["motion"], dev["social"]), 1)
    _, mean, std = standardize(train_x, train_x)
    train_x = ((train_x - mean) / std).astype(np.float32)
    dev_x = ((dev_x - mean) / std).astype(np.float32)
    train_x, train_y = _binary(train_x, train["change_group"])
    dev_x, dev_y = _binary(dev_x, dev["change_group"])
    seed_everything(seed)
    regime = load_module(f"detector_arch_{fold}_{seed}_{variant}", REGIME_RUNNER)
    model = regime.DetectorMLP(train_x.shape[1], spec["detector_dropout"]).to(device)
    pos_weight = float((train_y == 0).sum()) / max(float((train_y == 1).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["detector_learning_rate"],
                                  weight_decay=spec["detector_weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                        batch_size=spec["detector_batch_size"], shuffle=True,
                        generator=generator, num_workers=0)
    dev_tensor = torch.from_numpy(dev_x).to(device)
    best_auc, best_epoch, best_state, history = -math.inf, -1, None, []
    for epoch in range(1, spec["detector_epochs"] + 1):
        model.train(); losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            dev_logits = model(dev_tensor).cpu().numpy()
        auc = float(roc_auc_score(dev_y, dev_logits))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_auroc": auc})
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"No detector state for {fold}/{seed}/{variant}")
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        dev_logits = model(dev_tensor).cpu().numpy()
    temperature = _temperature(dev_logits, dev_y)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std, "temperature": temperature,
                "input_dim": int(train_x.shape[1]), "variant": variant, "seed": seed,
                "best_epoch": best_epoch, "best_dev_auroc": best_auc}, destination)
    write_csv(destination.with_suffix(".history.csv"), history)
    return destination


def detector_probability(cache: dict, checkpoint: Path, device: torch.device) -> np.ndarray:
    # These locally-created checkpoints include NumPy scaler arrays in addition
    # to tensors, so they require the ordinary trusted-local payload loader.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    x = cache["motion"] if saved["variant"] == "d0_motion" else np.concatenate((cache["motion"], cache["social"]), 1)
    x = ((x - saved["mean"]) / saved["std"]).astype(np.float32)
    regime = load_module(f"detector_predict_{hash(str(checkpoint)) & 0xfffffff}", REGIME_RUNNER)
    model = regime.DetectorMLP(saved["input_dim"], load_spec()["detector_dropout"])
    model.load_state_dict(saved["state_dict"]); model.to(device).eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), 2048):
            logits = model(torch.from_numpy(x[start:start + 2048]).to(device)) / saved["temperature"]
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def _scalers(train: dict) -> dict:
    motion_mean = train["motion"].mean(0, keepdims=True).astype(np.float32)
    motion_std = train["motion"].std(0, keepdims=True).astype(np.float32)
    motion_std = np.where(motion_std < 1e-6, 1.0, motion_std).astype(np.float32)
    candidate_mean = train["candidate"].mean((0, 1), keepdims=True).astype(np.float32)
    candidate_std = train["candidate"].std((0, 1), keepdims=True).astype(np.float32)
    candidate_std = np.where(candidate_std < 1e-6, 1.0, candidate_std).astype(np.float32)
    return {"motion_mean": motion_mean, "motion_std": motion_std,
            "candidate_mean": candidate_mean, "candidate_std": candidate_std}


def _normalize_cache(cache: dict, scalers: dict) -> tuple[np.ndarray, np.ndarray]:
    motion = ((cache["motion"] - scalers["motion_mean"]) / scalers["motion_std"]).astype(np.float32)
    candidate = ((cache["candidate"] - scalers["candidate_mean"]) / scalers["candidate_std"]).astype(np.float32)
    return motion, candidate


def _permutation(length: int, fold: str, seed: int, split: str) -> np.ndarray:
    token = hashlib.sha256(f"{fold}:{seed}:{split}:shuffled-d1".encode()).digest()
    return np.random.default_rng(int.from_bytes(token[:8], "little")).permutation(length)


def arm_gate(
    arm: str, p0: np.ndarray, p1: np.ndarray, fold: str, seed: int,
    split: str, constant: float,
) -> np.ndarray:
    if arm == "constant_gate":
        return np.full(len(p1), constant, dtype=np.float32)
    if arm == "shuffled_d1":
        return p1[_permutation(len(p1), fold, seed, split)]
    if arm == "d0_motion_gate":
        return p0
    if arm == "d1_social_gate":
        return p1
    raise ValueError(arm)


def _selection_arrays(logits: np.ndarray, ade: np.ndarray, fde: np.ndarray) -> dict:
    row = np.arange(len(logits)); selected = logits.argmax(1); oracle = ade.argmin(1)
    return {
        "selected_index": selected.astype(np.int64),
        "selected_ade": ade[row, selected].astype(np.float64),
        "selected_fde": fde[row, selected].astype(np.float64),
        "oracle_ade": ade[row, oracle].astype(np.float64),
        "oracle_fde": fde[row, oracle].astype(np.float64),
    }


def _dev_metrics(model, motion, candidate, native, gate, cache, device) -> dict:
    model.eval(); output = []
    with torch.inference_mode():
        for start in range(0, len(motion), 1024):
            output.append(model(
                torch.from_numpy(motion[start:start + 1024]).to(device),
                torch.from_numpy(candidate[start:start + 1024]).to(device),
                torch.from_numpy(native[start:start + 1024]).to(device),
                torch.from_numpy(gate[start:start + 1024]).to(device),
            ).cpu().numpy())
    arrays = _selection_arrays(np.concatenate(output), cache["ade"], cache["fde"])
    return {"selected_ade": float(arrays["selected_ade"].mean()),
            "selected_fde": float(arrays["selected_fde"].mean())}


def train_reranker(
    fold: str, seed: int, arm: str, train: dict, dev: dict,
    p0_train: np.ndarray, p1_train: np.ndarray, p0_dev: np.ndarray, p1_dev: np.ndarray,
    spec: dict, device: torch.device,
) -> None:
    directory = model_dir(fold, seed)
    best_path, final_path = directory / f"{arm}_best.pth", directory / f"{arm}_final.pth"
    if best_path.is_file() and final_path.is_file():
        return
    seed_everything(seed)
    scalers = _scalers(train)
    train_motion, train_candidate = _normalize_cache(train, scalers)
    dev_motion, dev_candidate = _normalize_cache(dev, scalers)
    constant = float(p1_train.mean())
    train_gate = arm_gate(arm, p0_train, p1_train, fold, seed, "train", constant)
    dev_gate = arm_gate(arm, p0_dev, p1_dev, fold, seed, "dev", constant)
    model = HistoryCorrectionHead(train_motion.shape[1], train_candidate.shape[2],
                                  spec["reranker_hidden_dim"], spec["reranker_dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["reranker_learning_rate"],
                                  weight_decay=spec["reranker_weight_decay"])
    quality = train["ade"] + spec["quality_fde_weight"] * train["fde"]
    target = torch.softmax(torch.from_numpy(-quality / spec["soft_target_temperature"]), dim=-1).numpy().astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(train_motion), torch.from_numpy(train_candidate),
        torch.from_numpy(train["logits"]), torch.from_numpy(train_gate), torch.from_numpy(target),
    )
    loader = DataLoader(dataset, batch_size=spec["reranker_batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    best_key, best_state, best_epoch, history = (math.inf, math.inf), None, -1, []
    for epoch in range(1, spec["reranker_epochs"] + 1):
        model.train(); losses = []
        for motion, candidate, native, gate, target_prob in loader:
            motion, candidate, native = motion.to(device), candidate.to(device), native.to(device)
            gate, target_prob = gate.to(device), target_prob.to(device)
            output = model(motion, candidate, native, gate)
            loss = -(target_prob * F.log_softmax(output, dim=-1)).sum(-1).mean()
            loss = loss + spec["residual_l2_weight"] * model.last_residual.square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = _dev_metrics(model, dev_motion, dev_candidate, dev["logits"], dev_gate, dev, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        key = (metrics["selected_ade"], metrics["selected_fde"])
        if key < best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"No reranker state for {fold}/{seed}/{arm}")
    payload = {"state_dict": best_state, **scalers, "motion_dim": train_motion.shape[1],
               "candidate_dim": train_candidate.shape[2], "arm": arm, "fold": fold,
               "seed": seed, "constant_gate": constant, "best_epoch": best_epoch,
               "best_dev_selected_ade": best_key[0], "best_dev_selected_fde": best_key[1]}
    directory.mkdir(parents=True, exist_ok=True); torch.save(payload, best_path)
    payload["state_dict"] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    payload["final_epoch"] = spec["reranker_epochs"]
    torch.save(payload, final_path)
    write_csv(directory / f"{arm}_history.csv", history)


def _training_complete(fold: str, seed: int, spec: dict) -> bool:
    directory = model_dir(fold, seed)
    required = [directory / "d0_motion.pth", directory / "d1_social.pth"]
    for arm in spec["arms"]:
        required += [directory / f"{arm}_best.pth", directory / f"{arm}_final.pth"]
    return all(path.is_file() for path in required)


def assert_all_training_locked(spec: dict) -> None:
    missing = []
    for fold in spec["folds"]:
        for seed in spec["head_seeds"]:
            if not _training_complete(fold, seed, spec):
                missing.append(f"{fold}/seed_{seed}")
    if missing:
        raise RuntimeError("Test access blocked until every training run is locked: " + ", ".join(missing))


def train_all(spec: dict, device: torch.device) -> None:
    for fold in spec["folds"]:
        train, dev = load_cache(fold, "train"), load_cache(fold, "dev")
        for seed in spec["head_seeds"]:
            if _training_complete(fold, seed, spec):
                print(f"[skip locked] {fold}/seed_{seed}", flush=True); continue
            print(f"[detectors] {fold}/seed_{seed}", flush=True)
            d0 = train_detector(fold, seed, "d0_motion", train, dev, spec, device)
            d1 = train_detector(fold, seed, "d1_social", train, dev, spec, device)
            p0_train, p1_train = detector_probability(train, d0, device), detector_probability(train, d1, device)
            p0_dev, p1_dev = detector_probability(dev, d0, device), detector_probability(dev, d1, device)
            for arm in spec["arms"]:
                print(f"[reranker] {fold}/seed_{seed}/{arm}", flush=True)
                train_reranker(fold, seed, arm, train, dev, p0_train, p1_train, p0_dev, p1_dev,
                               spec, device)
            (model_dir(fold, seed) / "complete.json").write_text(json.dumps({
                "fold": fold, "seed": seed, "arms": spec["arms"], "test_accessed": False,
                "generator_seed": 1, "completed_at_unix": time.time(),
            }, indent=2), encoding="utf-8")
    assert_all_training_locked(spec)


def prepare_test(spec: dict, device: torch.device, force: bool = False) -> None:
    assert_all_training_locked(spec)
    cfg, regime, front, trainer, adapter = setup_backend(device)
    for fold in spec["folds"]:
        destination = cache_path(fold, "test")
        if destination.is_file() and not force:
            print(f"[skip test cache] {fold}", flush=True); continue
        data_path, num_path = fold_data(fold, "test")
        dataset = trainer.TrajectoryDataset(data_path, num_path, cfg)
        ids = np.arange(len(dataset), dtype=np.int64)
        model = build_baseline(fold, cfg, front, adapter, device)
        print(f"[extract held-out test] {fold}: {len(ids)}", flush=True)
        _extract_cache(fold, "test", ids, dataset, model, trainer, regime, device,
                       spec["feature_batch_size"], spec)
        del model, dataset
        if device.type == "cuda": torch.cuda.empty_cache()


def _load_head(path: Path, spec: dict, device: torch.device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = HistoryCorrectionHead(saved["motion_dim"], saved["candidate_dim"],
                                  spec["reranker_hidden_dim"], spec["reranker_dropout"])
    model.load_state_dict(saved["state_dict"]); model.to(device).eval()
    return model, saved


def _predict_head(model, saved: dict, cache: dict, gate: np.ndarray, device: torch.device) -> np.ndarray:
    motion, candidate = _normalize_cache(cache, saved)
    output = []
    with torch.inference_mode():
        for start in range(0, len(motion), 1024):
            output.append(model(
                torch.from_numpy(motion[start:start + 1024]).to(device),
                torch.from_numpy(candidate[start:start + 1024]).to(device),
                torch.from_numpy(cache["logits"][start:start + 1024]).to(device),
                torch.from_numpy(gate[start:start + 1024]).to(device),
            ).cpu().numpy())
    return np.concatenate(output)


def _metric_row(fold: str, seed: int, arm: str, checkpoint: str, cache: dict,
                arrays: dict, p1: np.ndarray) -> dict:
    group = cache["change_group"]
    threshold = np.quantile(p1, 0.8)
    high, low, stable = p1 >= threshold, p1 <= np.quantile(p1, 0.2), group == 0
    return {
        "fold": fold, "seed": seed, "arm": arm, "checkpoint": checkpoint, "n": len(group),
        "selected_ade": float(arrays["selected_ade"].mean()),
        "selected_fde": float(arrays["selected_fde"].mean()),
        "minade20": float(arrays["oracle_ade"].mean()),
        "minfde20": float(arrays["oracle_fde"].mean()),
        "ade_regret": float((arrays["selected_ade"] - arrays["oracle_ade"]).mean()),
        "stable_n": int(stable.sum()), "stable_selected_ade": float(arrays["selected_ade"][stable].mean()),
        "high_p_n": int(high.sum()), "high_p_selected_ade": float(arrays["selected_ade"][high].mean()),
        "low_p_n": int(low.sum()), "low_p_selected_ade": float(arrays["selected_ade"][low].mean()),
    }


def evaluate(spec: dict, device: torch.device) -> None:
    assert_all_training_locked(spec)
    for fold in spec["folds"]:
        if not cache_path(fold, "test").is_file():
            raise RuntimeError(f"Missing held-out test cache for {fold}")
    rows, detector_rows = [], []
    for fold in spec["folds"]:
        cache = load_cache(fold, "test")
        for seed in spec["head_seeds"]:
            directory = model_dir(fold, seed)
            p0 = detector_probability(cache, directory / "d0_motion.pth", device)
            p1 = detector_probability(cache, directory / "d1_social.pth", device)
            binary = cache["change_group"] != 1
            labels = (cache["change_group"][binary] == 2).astype(np.int64)
            detector_rows += [
                {"fold": fold, "seed": seed, "detector": "d0_motion", "n": int(binary.sum()),
                 "prevalence": float(labels.mean()), "auroc": float(roc_auc_score(labels, p0[binary])),
                 "pr_auc": float(average_precision_score(labels, p0[binary]))},
                {"fold": fold, "seed": seed, "detector": "d1_social", "n": int(binary.sum()),
                 "prevalence": float(labels.mean()), "auroc": float(roc_auc_score(labels, p1[binary])),
                 "pr_auc": float(average_precision_score(labels, p1[binary]))},
            ]
            native_arrays = _selection_arrays(cache["logits"], cache["ade"], cache["fde"])
            rows.append(_metric_row(fold, seed, "native", "fixed", cache, native_arrays, p1))
            output_dir = EVALUATION / "per_sample" / fold / f"seed_{seed}"
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_dir / "native.npz", **native_arrays, p_d1=p1,
                                change_group=cache["change_group"], sample_id=cache["sample_id"])
            for arm in spec["arms"]:
                for checkpoint in ("best", "final"):
                    model, saved = _load_head(directory / f"{arm}_{checkpoint}.pth", spec, device)
                    gate = arm_gate(arm, p0, p1, fold, seed, "test", saved["constant_gate"])
                    logits = _predict_head(model, saved, cache, gate, device)
                    arrays = _selection_arrays(logits, cache["ade"], cache["fde"])
                    rows.append(_metric_row(fold, seed, arm, checkpoint, cache, arrays, p1))
                    np.savez_compressed(output_dir / f"{arm}_{checkpoint}.npz", **arrays, p_d1=p1,
                                        gate=gate, change_group=cache["change_group"], sample_id=cache["sample_id"])
    EVALUATION.mkdir(parents=True, exist_ok=True)
    write_csv(EVALUATION / "fold_seed_metrics.csv", rows)
    write_csv(EVALUATION / "detector_test_metrics.csv", detector_rows)
    (EVALUATION / "complete.json").write_text(json.dumps({
        "status": "HELD_OUT_TEST_EVALUATION_COMPLETE", "test_accessed": True,
        "checkpoint_selection_used_test": False, "rows": len(rows),
    }, indent=2), encoding="utf-8")


def hierarchical_macro_bootstrap(
    delta_by_fold: Dict[str, np.ndarray], repetitions: int, seed: int,
) -> tuple[float, float, float]:
    point = float(np.mean([np.mean(values) for values in delta_by_fold.values()]))
    rng = np.random.default_rng(seed); samples = np.empty(repetitions, dtype=np.float64)
    items = list(delta_by_fold.items())
    for repetition in range(repetitions):
        fold_means = []
        for _, values in items:
            index = rng.integers(0, len(values), len(values))
            fold_means.append(float(values[index].mean()))
        samples[repetition] = np.mean(fold_means)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return point, float(lower), float(upper)


def _load_sample(fold: str, seed: int, name: str) -> dict:
    path = EVALUATION / "per_sample" / fold / f"seed_{seed}" / f"{name}.npz"
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _average_seed_delta(spec: dict, left: str, right: str, metric: str,
                        mask_name: str | None = None) -> Dict[str, np.ndarray]:
    result = {}
    for fold in spec["folds"]:
        seed_deltas, seed_probabilities = [], []
        reference_group = None
        for seed in spec["head_seeds"]:
            lhs = _load_sample(fold, seed, left)
            rhs = _load_sample(fold, seed, right)
            if not np.array_equal(lhs["sample_id"], rhs["sample_id"]):
                raise RuntimeError(f"Per-sample alignment failed: {fold}/{seed}/{left}/{right}")
            if reference_group is None:
                reference_group = lhs["change_group"]
            elif not np.array_equal(reference_group, lhs["change_group"]):
                raise RuntimeError(f"Change-group alignment failed: {fold}/{seed}")
            seed_deltas.append(lhs[metric] - rhs[metric])
            seed_probabilities.append(lhs["p_d1"])
        # Define subgroup membership once from the seed-averaged detector
        # probability. Per-seed quantiles can contain different numbers of tied
        # samples, which must never be stacked as though they were aligned.
        averaged_delta = np.mean(np.stack(seed_deltas), axis=0)
        averaged_probability = np.mean(np.stack(seed_probabilities), axis=0)
        if mask_name == "stable":
            averaged_delta = averaged_delta[reference_group == 0]
        elif mask_name == "high":
            averaged_delta = averaged_delta[averaged_probability >= np.quantile(averaged_probability, 0.8)]
        elif mask_name == "low":
            averaged_delta = averaged_delta[averaged_probability <= np.quantile(averaged_probability, 0.2)]
        result[fold] = averaged_delta
    return result


def analyze(spec: dict) -> None:
    if not (EVALUATION / "complete.json").is_file():
        raise RuntimeError("Held-out evaluation is incomplete")
    comparisons = [
        (arm, f"{arm}_best", "native") for arm in spec["arms"]
    ] + [
        ("d1_vs_constant", "d1_social_gate_best", "constant_gate_best"),
        ("d1_vs_shuffled", "d1_social_gate_best", "shuffled_d1_best"),
        ("d1_vs_d0", "d1_social_gate_best", "d0_motion_gate_best"),
    ]
    rows = []
    for offset, (name, left, right) in enumerate(comparisons):
        for metric in ("selected_ade", "selected_fde"):
            delta = _average_seed_delta(spec, left, right, metric)
            point, lower, upper = hierarchical_macro_bootstrap(
                delta, spec["bootstrap_repetitions"], spec["bootstrap_seed"] + offset * 20 + (metric == "selected_fde")
            )
            rows.append({"comparison": name, "metric": metric, "macro_delta": point,
                         "ci_lower": lower, "ci_upper": upper})
    for mask in ("stable", "high", "low"):
        delta = _average_seed_delta(spec, "d1_social_gate_best", "native", "selected_ade", mask)
        point, lower, upper = hierarchical_macro_bootstrap(
            delta, spec["bootstrap_repetitions"], spec["bootstrap_seed"] + 500 + len(rows)
        )
        rows.append({"comparison": f"d1_vs_native_{mask}", "metric": "selected_ade",
                     "macro_delta": point, "ci_lower": lower, "ci_upper": upper})
    primary = next(row for row in rows if row["comparison"] == "d1_social_gate" and row["metric"] == "selected_ade")
    primary_fde = next(row for row in rows if row["comparison"] == "d1_social_gate" and row["metric"] == "selected_fde")
    controls = [next(row for row in rows if row["comparison"] == name and row["metric"] == "selected_ade")
                for name in ("d1_vs_constant", "d1_vs_shuffled")]
    stable = next(row for row in rows if row["comparison"] == "d1_vs_native_stable")
    high = next(row for row in rows if row["comparison"] == "d1_vs_native_high")
    low = next(row for row in rows if row["comparison"] == "d1_vs_native_low")
    native_macro = np.mean([
        _load_sample(fold, spec["head_seeds"][0], "native")["selected_ade"].mean()
        for fold in spec["folds"]
    ])
    native_fde_macro = np.mean([
        _load_sample(fold, spec["head_seeds"][0], "native")["selected_fde"].mean()
        for fold in spec["folds"]
    ])
    gain = -primary["macro_delta"] / native_macro
    gates = {
        "primary_gain_at_least_1pct": bool(gain >= spec["minimum_primary_gain_fraction"] and primary["ci_upper"] < 0),
        "beats_constant_and_shuffled": bool(all(row["ci_upper"] < 0 for row in controls)),
        "fde_not_materially_worse": bool(
            primary_fde["macro_delta"]
            <= spec["maximum_fde_degradation_fraction"] * native_fde_macro
        ),
        "stable_not_materially_harmed": bool(stable["macro_delta"] <= spec["maximum_stable_ade_degradation_fraction"] * native_macro),
        "effect_concentrated_high_p": bool(high["macro_delta"] < low["macro_delta"]),
    }
    verdict = "ADVANCE_REGIME_ADAPTIVE_SELECTION" if all(gates.values()) else "STOP_OR_REDESIGN_REGIME_ADAPTIVE_SELECTION"
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    write_csv(ANALYSIS / "paired_macro_comparisons.csv", rows)
    summary = {"verdict": verdict, "gates": gates, "primary_gain_fraction": float(gain),
               "native_macro_selected_ade": float(native_macro),
               "native_macro_selected_fde": float(native_fde_macro), "comparisons": rows,
               "test_accessed": True, "test_used_for_selection": False,
               "seed_scope": spec["generator_seed_scope"]}
    (ANALYSIS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Formal Regime-Adaptive Selection Report", "", f"Verdict: `{verdict}`", "",
             "五折 Macro；三个 seed 仅属于 detector/reranker，生成器固定为每折 seed-1。", "",
             "| Comparison | Metric | Macro delta | 95% CI |", "|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['macro_delta']:.6f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] |")
    lines += ["", "## Gates", ""] + [f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in gates.items()]
    lines += ["", "Test was evaluated once after every train/dev-selected checkpoint was locked; test labels were not used for selection."]
    (ANALYSIS / "FORMAL_REGIME_SELECTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "primary_gain_fraction": gain, "gates": gates}, ensure_ascii=False, indent=2))


def preflight(spec: dict) -> dict:
    required = [PYTHON, CONFIG, REGIME_RUNNER, FRONT_MODELS, FRONT_TRAINER, V2_ADAPTER, SPEC_PATH, Path(__file__)]
    for fold in spec["folds"]:
        required += [*fold_data(fold, "train"), *fold_data(fold, "test")]
        required += [fold_run(fold, "h3") / "best_selected.pth",
                     fold_run(fold, "v2") / "best_selected.pth",
                     fold_run(fold, "v2") / "split_indices.npz",
                     fold_run(fold, "v2") / "metrics.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required sources:\n" + "\n".join(missing))
    if len(set(spec["arms"])) != 4 or spec["primary_arm"] != "d1_social_gate":
        raise RuntimeError("Formal arm contract changed")
    folds = {}
    for fold in spec["folds"]:
        ids = _split_ids(fold)
        if np.intersect1d(ids["train"], ids["dev"]).size:
            raise RuntimeError(f"Train/dev overlap for {fold}")
        metrics = json.loads((fold_run(fold) / "metrics.json").read_text(encoding="utf-8"))
        if metrics.get("evaluation_test_best_selected") is not None:
            raise RuntimeError(f"Source V2 unexpectedly contains test evaluation for {fold}")
        folds[fold] = {"train": len(ids["train"]), "dev": len(ids["dev"]),
                       "train_id_sha256": hashlib.sha256(ids["train"].tobytes()).hexdigest(),
                       "dev_id_sha256": hashlib.sha256(ids["dev"].tobytes()).hexdigest()}
    manifest = {
        "status": "PREFLIGHT_PASS_NO_TRAINING_NO_TEST_LOAD", "protocol": spec["protocol"],
        "folds": folds, "arms": spec["arms"], "head_seeds": spec["head_seeds"],
        "source_hashes": {str(path): sha256(path) for path in required if path.suffix not in {".npy"}},
        "large_array_locks": {str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                              for path in required if path.suffix == ".npy"},
        "test_access_authorized": True, "test_load_deferred_until_all_training_locked": True,
        "preflight_started_training": False,
    }
    if PREFLIGHT.is_file():
        old = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError("Preflight/source drift detected; inspect before replacing the lock")
    else:
        PREFLIGHT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def smoke(spec: dict, device: torch.device) -> None:
    preflight(spec)
    cfg, regime, front, trainer, adapter = setup_backend(device)
    fold = spec["folds"][0]; data, num = fold_data(fold, "train")
    dataset = trainer.TrajectoryDataset(data, num, cfg); model = build_baseline(fold, cfg, front, adapter, device)
    loader = DataLoader(Subset(dataset, _split_ids(fold)["train"][:8].tolist()), batch_size=8, shuffle=False)
    past, future, num_valid, *_ = next(iter(loader)); past, future = past.to(device), future.to(device)
    num_valid = num_valid.to(device).int(); past_local, future_local = trainer.normalize_to_focal_heading(past, future)
    center = past_local.mean(-2, keepdim=True)
    prediction = model(past_local - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past_local)[2] + center.unsqueeze(-3)
    logits = model.last_mode_logits[:, 0]; relative = prediction[:, 0] - past_local[:, 0, -1, None, None]
    motion = regime.motion_features(past_local); candidate = candidate_features(past_local, relative, logits)
    head = HistoryCorrectionHead(motion.shape[1], candidate.shape[-1], spec["reranker_hidden_dim"], 0.0).to(device)
    output = head(motion, candidate, logits, torch.rand(len(past), device=device))
    errors = torch.linalg.vector_norm(prediction[:, 0] - future_local[:, 0, None], dim=-1)
    target = torch.softmax(-(errors.mean(-1) + .5 * errors[..., -1]) / spec["soft_target_temperature"], -1)
    loss = -(target * F.log_softmax(output, -1)).sum(-1).mean(); loss.backward()
    if output.shape != logits.shape or not torch.isfinite(loss) or not any(p.grad is not None for p in head.parameters()):
        raise RuntimeError("Real-data smoke failed")
    print(json.dumps({"status": "REAL_GPU_SMOKE_PASS", "motion_shape": list(motion.shape),
                      "candidate_shape": list(candidate.shape), "logit_shape": list(logits.shape),
                      "loss": float(loss.detach().cpu())}, indent=2))


def status(spec: dict) -> None:
    print(f"[{'ready' if PREFLIGHT.is_file() else 'pending'}] preflight: {PREFLIGHT}")
    for fold in spec["folds"]:
        train_ready = all(cache_path(fold, split).is_file() for split in ("train", "dev"))
        locked = all(_training_complete(fold, seed, spec) for seed in spec["head_seeds"])
        print(f"[{'ready' if train_ready else 'pending'}] {fold} train/dev cache")
        print(f"[{'ready' if locked else 'pending'}] {fold} all model locks")
        print(f"[{'ready' if cache_path(fold, 'test').is_file() else 'pending'}] {fold} test cache")
    print(f"[{'ready' if (EVALUATION / 'complete.json').is_file() else 'pending'}] held-out evaluation")
    print(f"[{'ready' if (ANALYSIS / 'summary.json').is_file() else 'pending'}] formal analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "preparetrain", "train", "preparetest",
                                              "evaluate", "analyze", "all", "status", "smoke"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); spec = load_spec()
    if args.stage == "status": status(spec); return
    if args.stage == "analyze": analyze(spec); return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    if args.stage == "preflight": preflight(spec)
    elif args.stage == "smoke": smoke(spec, device)
    elif args.stage == "preparetrain": prepare_train(spec, device, args.force)
    elif args.stage == "train": train_all(spec, device)
    elif args.stage == "preparetest": prepare_test(spec, device, args.force)
    elif args.stage == "evaluate": evaluate(spec, device)
    else:
        preflight(spec)
        prepare_train(spec, device, args.force)
        train_all(spec, device)
        prepare_test(spec, device, args.force)
        evaluate(spec, device)
        analyze(spec)


if __name__ == "__main__":
    main()
