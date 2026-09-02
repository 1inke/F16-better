"""EXP-183A-R2-V: spatial DINO, style suppression, mode-conditioned audit."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import cv2
import numpy as np
import timm
import torch
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from exp183ar2_probe_models import TinyCandidateQualityProbe, parameter_count


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "EXP_183A_R2_SPATIAL_DINO_MODE_SCENE"
CACHE = OUT / "cache"
MODEL_CACHE = CACHE / "model_features"
SCENE_CACHE = CACHE / "mode_scene"
PREVIEWS = OUT / "previews"
DATA = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
ASSETS = ROOT / "SCENE_QUERY_CURRENT_DECODER_SDD_SERIES" / "processed_scene_assets"
S0 = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175S0_FINAL_JOINT_V2_SDD_BASELINE"
BASE = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175A_SCENE_UTILITY_AUDIT" / "run_exp175a_scene_utility_audit.py"
SDD_ROOT = Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset")
PIXEL_SCALE = 20.0
SPLITS = ("train", "validation", "test")
SCENE_VARIANTS = ("Q1_spatial_raw", "Q2_crop_centered", "Q3_video_centered")
ALL_VARIANTS = ("Native", "Q0_model_only") + SCENE_VARIANTS


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_loader(split: str, cfg, trainer, batch_size: int, max_samples: int):
    if split in {"train", "validation"}:
        prefix = "sdd_standard_train_val"
        sidecar = ASSETS / "sdd_train_val_scene_ids.npy"
        scene_ids = np.load(sidecar, mmap_mode="r")
        keep = np.flatnonzero(scene_ids < 31) if split == "train" else np.flatnonzero((scene_ids >= 31) & (scene_ids <= 42))
    else:
        prefix = "sdd_standard_test"
        sidecar = ASSETS / "sdd_test_scene_ids.npy"
        scene_ids = np.load(sidecar, mmap_mode="r")
        keep = np.arange(len(scene_ids))
    dataset = trainer.TrajectoryDataset(
        DATA / f"{prefix}_data.npy", DATA / f"{prefix}_num.npy", cfg,
        sample_scene_path=sidecar,
    )
    if max_samples > 0 and len(keep) > max_samples:
        keep = keep[np.linspace(0, len(keep) - 1, max_samples, dtype=np.int64)]
    return DataLoader(Subset(dataset, keep.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)


def model_cache_path(split: str) -> Path:
    return MODEL_CACHE / f"{split}.npz"


def extract_model_features(device: torch.device, batch_size: int, max_samples: int, force: bool) -> None:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    if not force and all(model_cache_path(split).is_file() for split in SPLITS):
        print("[model] reuse complete frozen-model cache", flush=True)
        return
    base = load_module("exp183ar2_base", BASE)
    model, cfg, trainer = base.build_model(1, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def mode_hook(_module, _inputs, output):
        captured["mode_hidden"] = output.detach()

    handle = model.decoder.mode_mlp.register_forward_hook(mode_hook)
    for split in SPLITS:
        parts: dict[str, list[np.ndarray]] = {}

        def add(name: str, value: torch.Tensor, dtype=np.float32):
            parts.setdefault(name, []).append(value.detach().cpu().numpy().astype(dtype, copy=False))

        loader = split_loader(split, cfg, trainer, batch_size, max_samples)
        print(f"[model] {split}: {len(loader.dataset)} samples", flush=True)
        with torch.inference_mode():
            for past_abs, future_abs, num_valid, scene_id, *_ in loader:
                past_abs = past_abs.float().to(device)
                future_abs = future_abs.float().to(device)
                num_valid = num_valid.int().to(device)
                scene_id = scene_id.long().to(device)
                past, future = trainer.normalize_to_focal_heading(past_abs, future_abs)
                center = past.mean(dim=-2, keepdim=True)
                output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
                raw_prediction = output[2] if isinstance(output, (tuple, list)) else output
                prediction = (raw_prediction + center.unsqueeze(-3))[:, 0]
                mode_hidden = captured.get("mode_hidden")
                if mode_hidden is None or mode_hidden.shape[2] != 20:
                    raise RuntimeError("Failed to capture 20 frozen H3 mode embeddings")
                mode_hidden = mode_hidden[:, 0]
                logits = model.last_mode_logits[:, 0]
                source = model.decoder.last_anchor_proposals[:, 0] + center[:, 0, None]
                corrected = model.decoder.last_corrected_anchors[:, 0] + center[:, 0, None]
                last_local = past[:, 0, -1][:, None, None, :]
                anchor_delta = source - last_local
                correction_norm = torch.linalg.vector_norm(corrected - source, dim=-1)
                distance = torch.linalg.vector_norm(prediction - future[:, 0, None], dim=-1)
                ade_px = distance.mean(-1) * PIXEL_SCALE
                fde_px = distance[..., -1] * PIXEL_SCALE
                quality_px = ade_px + 0.25 * fde_px
                lag = min(3, past_abs.shape[-2] - 1)
                heading = F.normalize(past_abs[:, 0, -1] - past_abs[:, 0, -1-lag], dim=-1, eps=1e-6)
                add("mode_hidden", mode_hidden, np.float16)
                add("anchors", anchor_delta, np.float32)
                add("logits", logits)
                add("correction_norm", correction_norm)
                add("final_candidate", prediction, np.float16)
                add("ade", ade_px)
                add("fde", fde_px)
                add("quality", quality_px)
                add("scene_id", scene_id, np.int16)
                add("last_observation", past_abs[:, 0, -1])
                add("heading", heading)
                add("past_absolute", past_abs[:, 0], np.float32)
        arrays = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
        assert arrays["mode_hidden"].shape[1:] == (20, 64)
        assert arrays["anchors"].shape[1:] == (20, 3, 2)
        np.savez_compressed(model_cache_path(split), **arrays)
    handle.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_model_split(split: str) -> dict[str, np.ndarray]:
    with np.load(model_cache_path(split), allow_pickle=False) as raw:
        return {name: raw[name] for name in raw.files}


def image_bank() -> tuple[dict[int, np.ndarray], dict[int, str]]:
    manifest = json.loads((S0 / "video_ids" / "video_id_manifest.json").read_text(encoding="utf-8"))
    id_to_name = {int(value): key for key, value in manifest["mapping"].items()}
    images = {}
    for scene_id, name in id_to_name.items():
        path = SDD_ROOT / "annotations" / name / "reference.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        images[scene_id] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return images, id_to_name


def heading_crop(image: np.ndarray, center_model: np.ndarray, heading: np.ndarray, size: int = 224) -> np.ndarray:
    center = np.asarray(center_model, np.float32) * PIXEL_SCALE
    heading = np.asarray(heading, np.float32)
    right = np.asarray((-heading[1], heading[0]), np.float32)
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2
    xx, yy = np.meshgrid(axis, axis)
    map_x = center[0] + xx * right[0] - yy * heading[0]
    map_y = center[1] + xx * right[1] - yy * heading[1]
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class SpatialDINO:
    def __init__(self, device: torch.device, pretrained: bool = True):
        self.device = device
        self.model = timm.create_model(
            "vit_small_patch14_dinov2", pretrained=pretrained, num_classes=0, img_size=224
        ).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
        self.last_total_tokens = None

    def grids(self, crops: list[np.ndarray]) -> torch.Tensor:
        tensor = torch.from_numpy(np.stack(crops)).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda", dtype=torch.float16):
            output = self.model.forward_features((tensor - self.mean) / self.std)
        if isinstance(output, dict):
            output = output.get("x_norm_patchtokens", output.get("x_prenorm"))
        if output is None or output.ndim != 3:
            raise RuntimeError(f"Unexpected DINO forward_features output: {type(output)}")
        if output.shape[1] < 256:
            raise RuntimeError(f"DINO patch count too small: {tuple(output.shape)}")
        self.last_total_tokens = int(output.shape[1])
        patches = output[:, -256:]
        if patches.shape[1] != 256:
            raise RuntimeError(f"Expected exactly 256 spatial patches, got {patches.shape[1]}")
        return patches.float().reshape(len(crops), 16, 16, -1)


def normalized_grids(raw: torch.Tensor, prototype: torch.Tensor | None, kind: str) -> torch.Tensor:
    if kind == "Q1_spatial_raw":
        return raw
    if kind == "Q2_crop_centered":
        centered = raw - raw.mean(dim=(1, 2), keepdim=True)
    elif kind == "Q3_video_centered":
        if prototype is None:
            raise RuntimeError("video prototype required")
        centered = raw - prototype[:, None, None]
    else:
        raise ValueError(kind)
    return F.layer_norm(centered, (centered.shape[-1],))


def query_grid(grid: torch.Tensor, anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear 3x3 Gaussian query of [B,16,16,D] at local H3 anchors."""
    crop_x = 111.5 + anchors[..., 1] * PIXEL_SCALE
    crop_y = 111.5 - anchors[..., 0] * PIXEL_SCALE
    in_bounds = (crop_x >= 0) & (crop_x <= 223) & (crop_y >= 0) & (crop_y <= 223)
    patch_x = (crop_x + 0.5) / 14.0 - 0.5
    patch_y = (crop_y + 0.5) / 14.0 - 0.5
    offsets = torch.tensor(
        [(dx, dy) for dy in (-1.0, 0.0, 1.0) for dx in (-1.0, 0.0, 1.0)],
        device=grid.device, dtype=grid.dtype,
    )
    weights = torch.exp(-0.5 * offsets.square().sum(-1))
    weights = weights / weights.sum()
    x = patch_x[..., None] + offsets[:, 0]
    y = patch_y[..., None] + offsets[:, 1]
    sample_grid = torch.stack((2 * x / 15 - 1, 2 * y / 15 - 1), dim=-1)
    shape = sample_grid.shape
    sample_grid = sample_grid.reshape(shape[0], -1, 1, 2)
    values = F.grid_sample(
        grid.permute(0, 3, 1, 2), sample_grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )[..., 0].transpose(1, 2)
    values = values.reshape(*shape[:-1], grid.shape[-1])
    pooled = (values * weights.view(1, 1, 1, 9, 1)).sum(-2)
    pooled = pooled * in_bounds[..., None].to(pooled.dtype)
    return pooled, in_bounds.to(pooled.dtype)


def preprocess_crops(crops: list[np.ndarray], device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.stack(crops)).to(device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return tensor


def build_video_prototypes(dino: SpatialDINO, images: dict[int, np.ndarray], batch_size: int, force: bool) -> np.ndarray:
    path = CACHE / "video_prototypes.npy"
    if path.is_file() and not force:
        return np.load(path)
    print("[DINO] building deterministic 4x4 reference-image prototypes", flush=True)
    prototypes = np.zeros((max(images) + 1, 384), np.float32)
    for scene_id in sorted(images):
        image = images[scene_id]
        ys = np.linspace(0, image.shape[0], 5, dtype=int)
        xs = np.linspace(0, image.shape[1], 5, dtype=int)
        tiles = [cv2.resize(image[ys[y]:ys[y+1], xs[x]:xs[x+1]], (224, 224), interpolation=cv2.INTER_AREA) for y in range(4) for x in range(4)]
        tokens = []
        for start in range(0, 16, batch_size):
            tokens.append(dino.grids(tiles[start:start+batch_size]).mean(dim=(0, 1, 2)).cpu().numpy())
        prototypes[scene_id] = np.mean(tokens, axis=0)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, prototypes)
    return prototypes


def crops_for(data: dict[str, np.ndarray], indices: np.ndarray, images: dict[int, np.ndarray]) -> list[np.ndarray]:
    return [
        heading_crop(images[int(data["scene_id"][i])], data["last_observation"][i], data["heading"][i])
        for i in indices
    ]


def fit_spatial_pcas(dino: SpatialDINO, images: dict[int, np.ndarray], prototypes: np.ndarray,
                     batch_size: int, pca_tokens: int, force: bool) -> dict[str, list[PCA]]:
    path = CACHE / "spatial_pca.npz"
    if path.is_file() and not force:
        raw = np.load(path)
        result = {}
        for variant in SCENE_VARIANTS:
            result[variant] = []
            for horizon in range(3):
                pca = PCA(n_components=32)
                pca.mean_ = raw[f"{variant}_h{horizon}_mean"]
                pca.components_ = raw[f"{variant}_h{horizon}_components"]
                pca.explained_variance_ = raw[f"{variant}_h{horizon}_variance"]
                pca.explained_variance_ratio_ = raw[f"{variant}_h{horizon}_ratio"]
                pca.n_components_ = 32
                pca.n_features_in_ = 384
                pca.noise_variance_ = 0.0
                result[variant].append(pca)
        return result
    train = load_model_split("train")
    sample_count = min(len(train["anchors"]), max(1, math.ceil(pca_tokens / 60)))
    indices = np.linspace(0, len(train["anchors"]) - 1, sample_count, dtype=np.int64)
    result: dict[str, list[PCA]] = {}
    saved = {}
    for variant in SCENE_VARIANTS:
        print(f"[PCA] collecting train-only queries for {variant}: {sample_count} samples", flush=True)
        horizon_parts = [[], [], []]
        for start in range(0, sample_count, batch_size):
            idx = indices[start:start+batch_size]
            grid = dino.grids(crops_for(train, idx, images))
            proto = torch.from_numpy(prototypes[train["scene_id"][idx].astype(int)]).to(dino.device)
            queried, _ = query_grid(normalized_grids(grid, proto, variant), torch.from_numpy(train["anchors"][idx]).to(dino.device))
            queried = queried.cpu().numpy().astype(np.float16)
            for horizon in range(3):
                horizon_parts[horizon].append(queried[:, :, horizon].reshape(-1, queried.shape[-1]))
        result[variant] = []
        for horizon in range(3):
            values = np.concatenate(horizon_parts[horizon]).astype(np.float32)
            pca = PCA(n_components=32, svd_solver="randomized", random_state=1832 + horizon)
            pca.fit(values)
            result[variant].append(pca)
            saved[f"{variant}_h{horizon}_mean"] = pca.mean_
            saved[f"{variant}_h{horizon}_components"] = pca.components_
            saved[f"{variant}_h{horizon}_variance"] = pca.explained_variance_
            saved[f"{variant}_h{horizon}_ratio"] = pca.explained_variance_ratio_
    np.savez(path, **saved)
    return result


def save_alignment_previews(data: dict[str, np.ndarray], images: dict[int, np.ndarray], id_to_name: dict[int, str], count: int = 20) -> None:
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in np.linspace(0, len(data["anchors"]) - 1, min(count, len(data["anchors"])), dtype=int):
        scene_id = int(data["scene_id"][index])
        crop = heading_crop(images[scene_id], data["last_observation"][index], data["heading"][index])
        canvas = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        for coordinate in range(0, 225, 14):
            cv2.line(canvas, (coordinate, 0), (coordinate, 223), (100, 100, 100), 1)
            cv2.line(canvas, (0, coordinate), (223, coordinate), (100, 100, 100), 1)
        anchors = data["anchors"][index]
        px = np.stack((111.5 + anchors[..., 1] * PIXEL_SCALE, 111.5 - anchors[..., 0] * PIXEL_SCALE), axis=-1)
        heading = data["heading"][index]
        right = np.asarray((-heading[1], heading[0]), np.float32)
        history_delta = (data["past_absolute"][index] - data["last_observation"][index]) * PIXEL_SCALE
        history_forward = history_delta @ heading
        history_lateral = history_delta @ right
        history_px = np.rint(np.stack((111.5 + history_lateral, 111.5 - history_forward), axis=-1)).astype(np.int32)
        cv2.polylines(canvas, [history_px], False, (255, 0, 255), 2)
        for point in history_px:
            cv2.circle(canvas, tuple(point), 2, (255, 0, 255), -1)
        for mode in range(20):
            point = tuple(np.rint(px[mode, -1]).astype(int))
            cv2.circle(canvas, point, 2, (0, 255, 255), -1)
        for color, mode in zip(((0, 0, 255), (0, 255, 0), (255, 0, 0)), (0, 7, 13)):
            points = np.rint(np.concatenate((np.array([[[111.5, 111.5]]]), px[None, mode]), axis=1)[0]).astype(np.int32)
            cv2.polylines(canvas, [points], False, color, 2)
            for point in points[1:]:
                cv2.circle(canvas, tuple(point), 14, color, 1)
        cv2.arrowedLine(canvas, (112, 112), (112, 70), (255, 255, 255), 2)
        cv2.imwrite(str(PREVIEWS / f"sample_{index:05d}.jpg"), canvas)
        for mode in range(20):
            for horizon in range(3):
                x, y = px[mode, horizon]
                rows.append({"sample": int(index), "video": id_to_name[scene_id], "mode": mode, "horizon": (4, 8, 12)[horizon], "crop_x": float(x), "crop_y": float(y), "patch_x": float((x + 0.5) / 14 - 0.5), "patch_y": float((y + 0.5) / 14 - 0.5), "in_bounds": bool(0 <= x <= 223 and 0 <= y <= 223)})
    write_csv(OUT / "spatial_alignment.csv", rows)


def build_scene_cache(dino: SpatialDINO, images: dict[int, np.ndarray], prototypes: np.ndarray,
                      pcas: dict[str, list[PCA]], batch_size: int, force: bool) -> None:
    SCENE_CACHE.mkdir(parents=True, exist_ok=True)
    expected = [SCENE_CACHE / f"{split}.npz" for split in SPLITS]
    if not force and all(path.is_file() for path in expected):
        print("[scene] reuse complete mode-query cache", flush=True)
        return
    for split in SPLITS:
        data = load_model_split(split)
        parts = {variant: [] for variant in SCENE_VARIANTS}
        mask_parts = []
        pooled_parts = []
        print(f"[scene] {split}: {len(data['anchors'])} heading-aligned crops", flush=True)
        for start in range(0, len(data["anchors"]), batch_size):
            idx = np.arange(start, min(start + batch_size, len(data["anchors"])))
            grid = dino.grids(crops_for(data, idx, images))
            pooled_parts.append(grid.mean(dim=(1, 2)).cpu().numpy().astype(np.float16))
            proto = torch.from_numpy(prototypes[data["scene_id"][idx].astype(int)]).to(dino.device)
            anchors = torch.from_numpy(data["anchors"][idx]).to(dino.device)
            common_mask = None
            for variant in SCENE_VARIANTS:
                queried, in_bounds = query_grid(normalized_grids(grid, proto, variant), anchors)
                values = queried.cpu().numpy().astype(np.float32)
                transformed = [pcas[variant][h].transform(values[:, :, h].reshape(-1, 384)).reshape(len(idx), 20, 32) for h in range(3)]
                parts[variant].append(np.concatenate(transformed, axis=-1).astype(np.float16))
                common_mask = in_bounds.cpu().numpy().astype(np.float16)
            mask_parts.append(common_mask)
        np.savez_compressed(
            SCENE_CACHE / f"{split}.npz",
            **{variant: np.concatenate(parts[variant]) for variant in SCENE_VARIANTS},
            in_bounds=np.concatenate(mask_parts), pooled_dino=np.concatenate(pooled_parts),
        )


def write_preflight_checks(dino: SpatialDINO, first_grid: torch.Tensor, pcas: dict[str, list[PCA]]) -> None:
    centered = first_grid - first_grid.mean(dim=(1, 2), keepdim=True)
    split_bounds = {}
    for split in SPLITS:
        with np.load(SCENE_CACHE / f"{split}.npz", allow_pickle=False) as raw:
            mask = raw["in_bounds"].astype(np.float32)
        split_bounds[split] = {
            f"t{horizon}_in_bounds_ratio": float(mask[:, :, index].mean())
            for index, horizon in enumerate((4, 8, 12))
        }
    checks = {
        "dino_total_tokens": int(dino.last_total_tokens or 0),
        "dino_patch_tokens_after_prefix_removal": 256,
        "dino_grid": [16, 16],
        "prefix_tokens_removed": int((dino.last_total_tokens or 256) - 256),
        "crop_centered_pre_layernorm_max_abs_spatial_mean": float(centered.mean(dim=(1, 2)).abs().max().cpu()),
        "heading_crop": "up_is_focal_forward",
        "anchor_source": "frozen_H3_source_anchors_not_GT",
        "query": "bilinear_plus_3x3_gaussian_sigma_1_patch",
        "video_prototype": "reference.jpg deterministic 4x4 tiles; no trajectory labels",
        "pca": "fit on train-video queries only; 32D per horizon",
        "pca_dimensions": {variant: [int(pca.n_components_) for pca in values] for variant, values in pcas.items()},
        "probe_input_dimension": 170,
        "q0_and_scene_input_dimensions_equal": True,
        "future_usage": "ADE/FDE quality label only",
        "in_bounds": split_bounds,
    }
    (OUT / "preflight_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")


def load_all(split: str) -> dict[str, np.ndarray]:
    data = load_model_split(split)
    with np.load(SCENE_CACHE / f"{split}.npz", allow_pickle=False) as scene:
        data.update({name: scene[name] for name in scene.files})
    data["model_feature"] = np.concatenate((data["mode_hidden"].astype(np.float32), data["anchors"].reshape(len(data["anchors"]), 20, 6), data["logits"][..., None]), axis=-1)
    data["centered_quality"] = data["quality"] - data["quality"].mean(axis=1, keepdims=True)
    return data


def average_spearman(scores: np.ndarray, quality: np.ndarray) -> float:
    score_rank = rankdata(scores, axis=1, method="average")
    quality_rank = rankdata(-quality, axis=1, method="average")
    score_rank -= score_rank.mean(1, keepdims=True)
    quality_rank -= quality_rank.mean(1, keepdims=True)
    denominator = np.sqrt((score_rank ** 2).sum(1) * (quality_rank ** 2).sum(1))
    valid = denominator > 1e-12
    rho = np.full(len(scores), np.nan)
    rho[valid] = (score_rank[valid] * quality_rank[valid]).sum(1) / denominator[valid]
    return float(np.nanmean(rho))


def score_metrics(data: dict[str, np.ndarray], scores: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    if mask is None:
        mask = np.ones(len(scores), dtype=bool)
    scores = scores[mask]
    quality = data["quality"][mask]
    rows = np.arange(len(scores))
    oracle = quality.argmin(1)
    selected = scores.argmax(1)
    order = np.argsort(-scores, axis=1)
    regret = quality[rows, selected] - quality[rows, oracle]
    return {
        "samples": int(len(scores)),
        "spearman": average_spearman(scores, quality),
        "top1_hit": float((selected == oracle).mean()),
        "top3_recall": float((order[:, :3] == oracle[:, None]).any(1).mean()),
        "top5_recall": float((order[:, :5] == oracle[:, None]).any(1).mean()),
        "mean_regret_px": float(regret.mean()),
        "p95_regret_px": float(np.percentile(regret, 95)),
    }


def video_macro(data: dict[str, np.ndarray], scores: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    for scene_id in sorted(np.unique(data["scene_id"]).tolist()):
        mask = data["scene_id"] == scene_id
        metric = score_metrics(data, scores, mask)
        rows.append({"scene_id": int(scene_id), **metric})
    return float(np.mean([row["spearman"] for row in rows])), rows


def ridge_input(data: dict[str, np.ndarray], variant: str) -> np.ndarray:
    if variant == "Q0_model_only":
        scene = np.zeros((len(data["quality"]), 20, 96), np.float32)
    else:
        scene = data[variant].astype(np.float32)
    return np.concatenate((data["model_feature"], scene, data["in_bounds"].astype(np.float32)), axis=-1)


def run_ridge(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray]):
    scores = {"Native": {"validation": val["logits"], "test": test["logits"]}}
    models = {}
    for variant in ALL_VARIANTS[1:]:
        print(f"[linear] {variant}", flush=True)
        x_train = ridge_input(train, variant).reshape(-1, 170)
        y_train = train["centered_quality"].reshape(-1)
        scaler = StandardScaler().fit(x_train)
        ridge = Ridge(alpha=1.0).fit(scaler.transform(x_train), y_train)
        models[variant] = (scaler, ridge)
        scores[variant] = {}
        for split, data in (("validation", val), ("test", test)):
            prediction = ridge.predict(scaler.transform(ridge_input(data, variant).reshape(-1, 170))).reshape(len(data["quality"]), 20)
            scores[variant][split] = -prediction
    return models, scores


def train_tiny(variant: str, seed: int, train: dict[str, np.ndarray], val: dict[str, np.ndarray],
               device: torch.device, batch_size: int, epochs: int, patience: int):
    seed_all(seed)
    model = TinyCandidateQualityProbe().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_state, best_loss, best_epoch, stale = None, math.inf, 0, 0

    def tensors(data, ids):
        model_feature = torch.from_numpy(data["model_feature"][ids]).float().to(device)
        scene_values = np.zeros((len(ids), 20, 96), np.float32) if variant == "Q0_model_only" else data[variant][ids].astype(np.float32)
        scene_feature = torch.from_numpy(scene_values).to(device)
        bounds = torch.from_numpy(data["in_bounds"][ids].astype(np.float32)).to(device)
        target = torch.from_numpy(data["centered_quality"][ids]).float().to(device)
        return model_feature, scene_feature, bounds, target

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train["quality"]))
        model.train()
        for start in range(0, len(order), batch_size):
            batch = tensors(train, order[start:start+batch_size])
            loss = F.smooth_l1_loss(model(*batch[:3]), batch[3])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        losses = []
        with torch.inference_mode():
            for start in range(0, len(val["quality"]), batch_size):
                batch = tensors(val, np.arange(start, min(start + batch_size, len(val["quality"]))))
                losses.append(float(F.smooth_l1_loss(model(*batch[:3]), batch[3])))
        val_loss = float(np.mean(losses))
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"variant": variant, "seed": seed, "params": parameter_count(model), "best_epoch": best_epoch, "val_loss": best_loss}


def tiny_predict(model: nn.Module, variant: str, data: dict[str, np.ndarray], device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["quality"]), batch_size):
            sl = slice(start, min(start + batch_size, len(data["quality"])))
            model_feature = torch.from_numpy(data["model_feature"][sl]).float().to(device)
            scene_values = np.zeros((len(model_feature), 20, 96), np.float32) if variant == "Q0_model_only" else data[variant][sl].astype(np.float32)
            scene_feature = torch.from_numpy(scene_values).to(device)
            bounds = torch.from_numpy(data["in_bounds"][sl].astype(np.float32)).to(device)
            outputs.append((-model(model_feature, scene_feature, bounds)).cpu().numpy())
    return np.concatenate(outputs)


def video_id_accuracy(features: np.ndarray, labels: np.ndarray, seed: int = 1832) -> float:
    rng = np.random.default_rng(seed)
    train_ids, test_ids = [], []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        ids = rng.permutation(ids)
        cut = max(1, int(round(0.7 * len(ids))))
        train_ids.extend(ids[:cut])
        test_ids.extend(ids[cut:] if cut < len(ids) else ids[-1:])
    train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
    scaler = StandardScaler().fit(features[train_ids])
    x_train, x_test = scaler.transform(features[train_ids]), scaler.transform(features[test_ids])
    classes = np.unique(labels)
    centroids = np.stack([x_train[labels[train_ids] == label].mean(0) for label in classes])
    distance = ((x_test[:, None] - centroids[None]) ** 2).sum(-1)
    prediction = classes[distance.argmin(1)]
    return float((prediction == labels[test_ids]).mean())


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_scores(stage: str, scores: dict, val: dict, test: dict, id_to_name: dict[int, str]):
    summary, per_video = [], []
    for variant, split_scores in scores.items():
        val_macro, _ = video_macro(val, split_scores["validation"])
        test_macro, video_rows = video_macro(test, split_scores["test"])
        metric = score_metrics(test, split_scores["test"])
        row = {"stage": stage, "variant": variant, "validation_video_macro_spearman": val_macro, "test_video_macro_spearman": test_macro, **metric}
        summary.append(row)
        for video_row in video_rows:
            per_video.append({"stage": stage, "variant": variant, "video": id_to_name[video_row["scene_id"]], **video_row})
    return summary, per_video


def analyze(linear_scores: dict, tiny_scores: dict | None, train: dict, val: dict, test: dict,
            id_to_name: dict[int, str], tiny_runs: list[dict], smoke: bool) -> None:
    linear_summary, linear_video = summarize_scores("linear", linear_scores, val, test, id_to_name)
    all_summary, all_video = list(linear_summary), list(linear_video)
    if tiny_scores:
        tiny_summary, tiny_video = summarize_scores("tiny_mlp_seed_mean", tiny_scores, val, test, id_to_name)
        all_summary.extend(tiny_summary)
        all_video.extend(tiny_video)
    write_csv(OUT / "candidate_quality_probe.csv", all_summary)
    write_csv(OUT / "per_video.csv", all_video)
    write_csv(OUT / "tiny_probe_runs.csv", tiny_runs)

    video_id_rows = []
    labels = train["scene_id"].astype(int)
    video_id_rows.append({"variant": "V0_pooled_DINO", "train_holdout_video_id_accuracy": video_id_accuracy(train["pooled_dino"].astype(np.float32), labels)})
    for variant in SCENE_VARIANTS:
        feature = train[variant].astype(np.float32).mean(axis=1)
        video_id_rows.append({"variant": variant, "train_holdout_video_id_accuracy": video_id_accuracy(feature, labels)})
    write_csv(OUT / "video_id_probe.csv", video_id_rows)

    stage = "tiny_mlp_seed_mean" if tiny_scores and "Q0_model_only" in tiny_scores and any(v in tiny_scores for v in ("Q2_crop_centered", "Q3_video_centered")) else "linear"
    table = [row for row in all_summary if row["stage"] == stage]
    lookup = {row["variant"]: row for row in table}
    q0 = lookup["Q0_model_only"]
    candidate_names = [name for name in ("Q2_crop_centered", "Q3_video_centered") if name in lookup]
    chosen = max(candidate_names, key=lambda name: lookup[name]["validation_video_macro_spearman"])
    best = lookup[chosen]
    stage_video = [row for row in all_video if row["stage"] == stage]
    q0_video = {row["scene_id"]: row for row in stage_video if row["variant"] == "Q0_model_only"}
    best_video = {row["scene_id"]: row for row in stage_video if row["variant"] == chosen}
    positive = sum(best_video[key]["spearman"] > q0_video[key]["spearman"] for key in q0_video)
    rho_gain = best["test_video_macro_spearman"] - q0["test_video_macro_spearman"]
    top5_gain = best["top5_recall"] - q0["top5_recall"]
    regret_gain = 1 - best["mean_regret_px"] / max(q0["mean_regret_px"], 1e-12)
    p95_safe = best["p95_regret_px"] <= 1.02 * q0["p95_regret_px"]
    gates = {
        "spearman_gain": float(rho_gain), "spearman_gain_ge_0p05": bool(rho_gain >= 0.05),
        "top5_gain_pp": float(100 * top5_gain), "top5_gain_ge_5pp": bool(top5_gain >= 0.05),
        "positive_videos": int(positive), "positive_videos_ge_12_of_17": bool(positive >= 12),
        "mean_regret_gain_pct": float(100 * regret_gain), "mean_regret_gain_ge_5pct": bool(regret_gain >= 0.05),
        "p95_regret_change_pct": float(100 * (best["p95_regret_px"] / max(q0["p95_regret_px"], 1e-12) - 1)), "p95_regret_safe": bool(p95_safe),
    }
    if rho_gain >= 0.08 and top5_gain >= 0.08 and regret_gain >= 0.08 and positive >= 14 and p95_safe:
        decision = "STRONG_PASS"
    elif rho_gain >= 0.05 and top5_gain >= 0.05 and regret_gain >= 0.05 and positive >= 12 and p95_safe:
        decision = "PASS"
    elif rho_gain < 0.02 or positive < 10:
        decision = "STOP_SCENE_VISUAL_BRANCH"
    else:
        decision = "WEAK_SIGNAL"
    payload = {
        "experiment": "EXP-183A-R2-V", "dataset": "SDD", "smoke": smoke,
        "trajectory_model": "EXP-175S0 seed1 frozen", "scene_backbone": "DINOv2-small frozen offline",
        "selection_stage": stage, "validation_selected_scene_variant": chosen,
        "samples": {split: int(len(data["quality"])) for split, data in (("train", train), ("validation", val), ("test", test))},
        "variants": table, "video_id_diagnostic": video_id_rows, "gates": gates, "decision": decision,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# EXP-183A-R2-V Spatial DINO Mode-Conditioned Scene Audit", "",
        f"Decision: **{decision}**", "",
        f"- Selection stage: {stage}", f"- Validation-selected camera treatment: {chosen}",
        f"- Video-macro Spearman gain over Q0: {rho_gain:+.4f}", f"- Top5 winner recall gain: {100*top5_gain:+.2f} pp",
        f"- Positive test videos: {positive}/17", f"- Mean regret gain: {100*regret_gain:+.2f}%",
        f"- P95 regret change: {gates['p95_regret_change_pct']:+.2f}%", "",
        "The frozen trajectory generator, H3 anchors, Joint-V2 refinement, and native score were not modified.",
    ]
    (OUT / "RESULT_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dino-batch-size", type=int, default=32)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--pca-tokens", type=int, default=200000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--untrained-dino", action="store_true", help="Smoke plumbing only")
    args = parser.parse_args()
    global OUT, CACHE, MODEL_CACHE, SCENE_CACHE, PREVIEWS
    if args.smoke:
        OUT = OUT / "preflight"
        CACHE = OUT / "cache"
        MODEL_CACHE = CACHE / "model_features"
        SCENE_CACHE = CACHE / "mode_scene"
        PREVIEWS = OUT / "previews"
        if args.max_samples <= 0:
            args.max_samples = 32
        args.pca_tokens = min(args.pca_tokens, args.max_samples * 60)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    extract_model_features(device, args.batch_size, args.max_samples, args.force_cache)
    train_model = load_model_split("train")
    images, id_to_name = image_bank()
    save_alignment_previews(train_model, images, id_to_name, 20)
    dino = SpatialDINO(device, pretrained=not args.untrained_dino)
    # A real forward pass enforces the 16x16 patch-grid preflight.
    first_crop = heading_crop(images[int(train_model["scene_id"][0])], train_model["last_observation"][0], train_model["heading"][0])
    first_grid = dino.grids([first_crop])
    assert first_grid.shape[1:3] == (16, 16)
    prototypes = build_video_prototypes(dino, images, args.dino_batch_size, args.force_cache)
    pcas = fit_spatial_pcas(dino, images, prototypes, args.dino_batch_size, args.pca_tokens, args.force_cache)
    build_scene_cache(dino, images, prototypes, pcas, args.dino_batch_size, args.force_cache)
    write_preflight_checks(dino, first_grid, pcas)
    del dino
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train, val, test = (load_all(split) for split in SPLITS)
    _, linear_scores = run_ridge(train, val, test)
    linear_summary, linear_video = summarize_scores("linear", linear_scores, val, test, id_to_name)
    linear_lookup = {row["variant"]: row for row in linear_summary}
    q0_rho = linear_lookup["Q0_model_only"]["validation_video_macro_spearman"]
    _, q0_validation_rows = video_macro(val, linear_scores["Q0_model_only"]["validation"])
    q0_validation = {row["scene_id"]: row for row in q0_validation_rows}
    stable = []
    for variant in SCENE_VARIANTS:
        _, variant_validation_rows = video_macro(val, linear_scores[variant]["validation"])
        variant_validation = {row["scene_id"]: row for row in variant_validation_rows}
        positive = sum(variant_validation[key]["spearman"] > q0_validation[key]["spearman"] for key in q0_validation)
        if linear_lookup[variant]["validation_video_macro_spearman"] > q0_rho and positive >= 7:
            stable.append(variant)
    tiny_scores, tiny_runs = None, []
    if stable:
        variants = ["Q0_model_only"] + stable
        tiny_seed_scores = {variant: {"validation": [], "test": []} for variant in variants}
        for variant in variants:
            for seed in args.probe_seeds:
                print(f"[tiny] {variant} seed={seed}", flush=True)
                model, row = train_tiny(variant, seed, train, val, device, args.probe_batch_size, args.epochs, args.patience)
                tiny_runs.append(row)
                for split, data in (("validation", val), ("test", test)):
                    tiny_seed_scores[variant][split].append(tiny_predict(model, variant, data, device, args.probe_batch_size))
        tiny_scores = {variant: {split: np.mean(values, axis=0) for split, values in split_values.items()} for variant, split_values in tiny_seed_scores.items()}
        tiny_scores["Native"] = {"validation": val["logits"], "test": test["logits"]}
    else:
        print("[tiny] skipped: no spatial linear variant showed stable positive cross-video signal", flush=True)
    analyze(linear_scores, tiny_scores, train, val, test, id_to_name, tiny_runs, args.smoke)
    print(f"[done] {OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
