"""Build aligned SDD scene sidecars and an offline semantic raster bank."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt


DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"


def video_order(manifest: dict) -> tuple[list[str], list[str]]:
    train_val = list(manifest["videos"]["train"]) + list(manifest["videos"]["val"])
    return train_val, list(manifest["videos"]["test"])


def build_scene_ids(manifest: dict, output_dir: Path) -> dict[str, int]:
    counts = {row["video"]: int(row["samples"]) for row in manifest["video_stats"]}
    train_val, test = video_order(manifest)
    all_videos = train_val + test
    ids = {video: index for index, video in enumerate(all_videos)}
    train_val_ids = np.concatenate([
        np.full(counts[video], ids[video], dtype=np.int16) for video in train_val
    ])
    test_ids = np.concatenate([
        np.full(counts[video], ids[video], dtype=np.int16) for video in test
    ])
    np.save(output_dir / "sdd_train_val_scene_ids.npy", train_val_ids)
    np.save(output_dir / "sdd_test_scene_ids.npy", test_ids)
    return ids


def category_masks(labels: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    names = {int(key): str(value).lower() for key, value in id2label.items()}
    walk_terms = ("road", "sidewalk", "floor", "earth", "grass", "path", "field", "runway", "stair", "step")
    obstacle_terms = ("building", "wall", "fence", "railing", "car", "truck", "bus", "bicycle", "motorcycle", "pole")
    vegetation_terms = ("tree", "grass", "plant", "flower")
    structure_terms = ("building", "wall", "fence", "railing", "pole", "signboard")

    def mask_for(terms: tuple[str, ...]) -> np.ndarray:
        chosen = [index for index, name in names.items() if any(term in name for term in terms)]
        return np.isin(labels, chosen)

    walkable = mask_for(walk_terms)
    obstacle = mask_for(obstacle_terms)
    vegetation = mask_for(vegetation_terms)
    structure = mask_for(structure_terms)
    boundary = np.zeros_like(walkable)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    signed = distance_transform_edt(walkable) - distance_transform_edt(~walkable)
    signed = np.clip(signed / 12.0, -1.0, 1.0)
    channels = np.stack((
        walkable, obstacle, vegetation, structure, boundary,
        (signed + 1.0) * 0.5,
    )).astype(np.float32)
    return np.rint(channels * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--map-size", type=int, default=160)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sidecars-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank_path = output / "sdd_semantic_bank.npz"
    manifest = json.loads(args.trajectory_manifest.read_text(encoding="utf-8"))
    ids = build_scene_ids(manifest, output)
    if args.sidecars_only:
        print("[done] scene-ID sidecars only", flush=True)
        return
    if bank_path.is_file() and not args.force:
        print(f"[skip] {bank_path}")
        return

    cache = Path(__file__).resolve().parent / ".cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache))
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained(args.model, cache_dir=cache)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, cache_dir=cache
    ).to(device).eval()
    all_videos = [video for video, _ in sorted(ids.items(), key=lambda item: item[1])]
    maps, sizes = [], []
    for index, video in enumerate(all_videos):
        image_path = args.dataset_root / "annotations" / video / "reference.jpg"
        image = Image.open(image_path).convert("RGB")
        sizes.append((image.width, image.height))
        inputs = processor(images=image, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits, size=(args.map_size, args.map_size),
                mode="bilinear", align_corners=False,
            )
        labels = logits.argmax(dim=1)[0].cpu().numpy()
        maps.append(category_masks(labels, model.config.id2label))
        print(f"[scene {index + 1:02d}/{len(all_videos)}] {video}", flush=True)

    np.savez_compressed(
        bank_path,
        maps=np.stack(maps),
        image_wh=np.asarray(sizes, dtype=np.int32),
        video_names=np.asarray(all_videos),
        channel_names=np.asarray([
            "walkable", "obstacle", "vegetation", "structure", "boundary", "signed_distance"
        ]),
        pixel_scale=np.asarray([float(manifest["pixel_scale"])], dtype=np.float32),
    )
    metadata = {
        "segmentation_model": args.model,
        "semantic_network_usage": "offline preprocessing only",
        "map_size": args.map_size,
        "channels": ["walkable", "obstacle", "vegetation", "structure", "boundary", "signed_distance"],
        "video_to_scene_id": ids,
        "train_val_sidecar_samples": int(sum(1 for _ in np.load(output / "sdd_train_val_scene_ids.npy", mmap_mode="r"))),
        "test_sidecar_samples": int(sum(1 for _ in np.load(output / "sdd_test_scene_ids.npy", mmap_mode="r"))),
    }
    (output / "scene_asset_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {bank_path}", flush=True)


if __name__ == "__main__":
    main()
