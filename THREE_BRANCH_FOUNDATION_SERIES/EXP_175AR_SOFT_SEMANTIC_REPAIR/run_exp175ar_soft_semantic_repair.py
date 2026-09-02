from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
OLD_ASSETS = ROOT / "SCENE_QUERY_CURRENT_DECODER_SDD_SERIES" / "processed_scene_assets"
AUDIT = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175A_SCENE_UTILITY_AUDIT" / "run_exp175a_scene_utility_audit.py"
DEFAULT_DATASET = Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset")
DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
CACHE = ROOT / "SCENE_QUERY_SERIES" / ".cache" / "huggingface"

WALKABLE = {"floor", "road", "grass", "sidewalk", "earth", "field", "path", "stairs", "stairway"}
OBSTACLE = {"wall", "building", "car", "fence", "bus", "truck", "streetlight", "pole", "bicycle"}
VEGETATION = {"tree", "grass", "plant", "flower"}
STRUCTURE = {"wall", "building", "fence", "streetlight", "pole"}


def exact_ids(id2label: dict, names: set[str]) -> list[int]:
    normalized = {int(i): str(name).strip().lower() for i, name in id2label.items()}
    return [i for i, name in normalized.items() if name in names]


def sum_classes(probs: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return torch.zeros_like(probs[0])
    return probs[indices].sum(dim=0)


def build_channels(probs: torch.Tensor, groups: dict[str, list[int]]) -> np.ndarray:
    walk_raw = sum_classes(probs, groups["walkable"])
    obstacle_raw = sum_classes(probs, groups["obstacle"])
    vegetation_raw = sum_classes(probs, groups["vegetation"])
    structure_raw = sum_classes(probs, groups["structure"])
    evidence = (walk_raw + obstacle_raw).clamp_min(1e-6)
    walk = (walk_raw / evidence).clamp(0.0, 1.0)
    obstacle = (obstacle_raw / evidence).clamp(0.0, 1.0)
    vegetation = vegetation_raw.clamp(0.0, 1.0)
    structure = structure_raw.clamp(0.0, 1.0)
    dx = F.pad((walk[:, 1:] - walk[:, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((walk[1:] - walk[:-1]).abs(), (0, 0, 0, 1))
    boundary = (3.0 * torch.maximum(dx, dy)).clamp(0.0, 1.0)
    traversable = walk.detach().cpu().numpy() >= 0.5
    signed = distance_transform_edt(traversable) - distance_transform_edt(~traversable)
    signed = np.clip(signed / 18.0, -1.0, 1.0)
    channels = np.stack([
        walk.cpu().numpy(), obstacle.cpu().numpy(), vegetation.cpu().numpy(),
        structure.cpu().numpy(), boundary.cpu().numpy(), (signed + 1.0) * 0.5,
    ])
    return np.rint(np.clip(channels, 0.0, 1.0) * 255.0).astype(np.uint8)


def prepare_model(device: torch.device):
    CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE)
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE)
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    processor = SegformerImageProcessor.from_pretrained(DEFAULT_MODEL, cache_dir=CACHE, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        DEFAULT_MODEL, cache_dir=CACHE, local_files_only=True
    ).to(device).eval()
    groups = {
        "walkable": exact_ids(model.config.id2label, WALKABLE),
        "obstacle": exact_ids(model.config.id2label, OBSTACLE),
        "vegetation": exact_ids(model.config.id2label, VEGETATION),
        "structure": exact_ids(model.config.id2label, STRUCTURE),
    }
    if any(not ids for ids in groups.values()):
        raise RuntimeError(f"Empty exact semantic group: {groups}")
    return processor, model, groups


def process_image(image: Image.Image, processor, model, groups, device, map_size: int):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = F.interpolate(logits, size=(map_size, map_size), mode="bilinear", align_corners=False)
        probs = logits.softmax(dim=1)[0].float().cpu()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return build_channels(probs, groups), time.perf_counter() - start


def make_overlays(asset_dir: Path, dataset_root: Path, maps: np.ndarray, image_wh: np.ndarray, video_names: list[str]):
    out = asset_dir / "qa_overlays"
    out.mkdir(parents=True, exist_ok=True)
    train_val = np.load(DATA_DIR / "sdd_standard_train_val_data.npy", mmap_mode="r")
    test = np.load(DATA_DIR / "sdd_standard_test_data.npy", mmap_mode="r")
    train_ids = np.load(asset_dir / "sdd_train_val_scene_ids.npy", mmap_mode="r")
    test_ids = np.load(asset_dir / "sdd_test_scene_ids.npy", mmap_mode="r")
    for scene_id in range(31, len(video_names)):
        video = video_names[scene_id]
        source = dataset_root / "annotations" / video / "reference.jpg"
        image = Image.open(source).convert("RGB")
        walk = Image.fromarray(maps[scene_id, 0]).resize(image.size, Image.Resampling.BILINEAR)
        obs = Image.fromarray(maps[scene_id, 1]).resize(image.size, Image.Resampling.BILINEAR)
        tint = Image.merge("RGB", (obs, walk, Image.new("L", image.size, 0)))
        canvas = Image.blend(image, tint, 0.35)
        draw = ImageDraw.Draw(canvas)
        data, ids = (train_val, train_ids) if scene_id <= 42 else (test, test_ids)
        indices = np.flatnonzero(ids == scene_id)
        if len(indices) > 60:
            indices = indices[np.linspace(0, len(indices) - 1, 60).round().astype(int)]
        for index in indices:
            points = np.asarray(data[index, 0, :, -2:], dtype=np.float32) * 20.0
            past = [tuple(map(float, p)) for p in points[:8]]
            future = [tuple(map(float, p)) for p in points[7:]]
            draw.line(past, fill=(0, 160, 255), width=2)
            draw.line(future, fill=(255, 255, 255), width=2)
        canvas.thumbnail((960, 720), Image.Resampling.LANCZOS)
        canvas.save(out / f"scene_{scene_id:02d}_{video.replace('/', '_')}.jpg", quality=90)


def build_assets(args, device: torch.device):
    asset_dir = args.asset_dir.resolve()
    bank_path = asset_dir / "sdd_semantic_bank.npz"
    if bank_path.exists() and not args.force_assets:
        print(f"[skip assets] {bank_path}", flush=True)
        return asset_dir
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    videos = manifest["videos"]["train"] + manifest["videos"]["val"] + manifest["videos"]["test"]
    processor, model, groups = prepare_model(device)
    if args.preflight_only:
        videos = videos[:1]
    maps, sizes, times = [], [], []
    for i, video in enumerate(videos):
        image = Image.open(args.dataset_root / "annotations" / video / "reference.jpg").convert("RGB")
        channels, elapsed = process_image(image, processor, model, groups, device, args.map_size)
        maps.append(channels); sizes.append((image.width, image.height)); times.append(elapsed)
        print(f"[semantic {i + 1:02d}/{len(videos)}] {video} {elapsed:.3f}s", flush=True)
    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_ok", "maps": list(np.stack(maps).shape),
            "walkable_mean": float(np.stack(maps)[:, 0].mean() / 255.0),
            "obstacle_mean": float(np.stack(maps)[:, 1].mean() / 255.0),
            "exact_class_ids": groups,
        }, ensure_ascii=False, indent=2))
        return asset_dir

    asset_dir.mkdir(parents=True, exist_ok=True)
    maps_array = np.stack(maps)
    sizes_array = np.asarray(sizes, np.int32)
    np.savez_compressed(
        bank_path, maps=maps_array, image_wh=sizes_array, video_names=np.asarray(videos),
        channel_names=np.asarray(["walkable", "obstacle", "vegetation", "structure", "boundary", "signed_distance"]),
        pixel_scale=np.asarray([20.0], np.float32),
    )
    for name in ("sdd_train_val_scene_ids.npy", "sdd_test_scene_ids.npy"):
        np.save(asset_dir / name, np.load(OLD_ASSETS / name))
    metadata = {
        "experiment": "EXP-175A-R", "model": DEFAULT_MODEL, "offline_only": True,
        "map_size": args.map_size, "softmax_before_class_aggregation": True,
        "exact_class_ids": groups, "extractor_parameters": sum(p.numel() for p in model.parameters()),
        "preprocess_seconds_mean": float(np.mean(times)), "preprocess_seconds_total": float(np.sum(times)),
        "walkable_map_mean": float(maps_array[:, 0].mean() / 255.0),
        "obstacle_map_mean": float(maps_array[:, 1].mean() / 255.0),
    }
    (asset_dir / "scene_asset_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    make_overlays(asset_dir, args.dataset_root, maps_array, sizes_array, videos)
    print(f"[assets] {bank_path}", flush=True)
    return asset_dir


def main():
    parser = argparse.ArgumentParser(description="EXP-175A-R soft semantic repair and repeated utility audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--map-size", type=int, default=320)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--asset-dir", type=Path, default=EXP_DIR / "processed_scene_assets_soft_v2")
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "results" / "seed_1")
    parser.add_argument("--force-assets", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not args.dataset_root.exists() or not AUDIT.exists():
        raise FileNotFoundError(args.dataset_root if not args.dataset_root.exists() else AUDIT)
    device = torch.device(args.device)
    asset_dir = build_assets(args, device)
    if args.preflight_only or args.skip_audit:
        return
    command = [
        sys.executable, "-u", str(AUDIT), "--device", args.device, "--seed", str(args.seed),
        "--batch-size", str(args.batch_size), "--asset-dir", str(asset_dir),
        "--output-dir", str(args.output_dir.resolve()),
    ]
    print("[audit] " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
