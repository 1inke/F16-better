"""Build leakage-safe SDD walkability and topology assets for EXP-178A."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from build_topology_graph import probability_to_topology


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
PREPROCESS = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "preprocess_sdd_standard.py"
MAP_SIZE = 160


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TinyWalkabilityCNN(nn.Module):
    """9.8K-parameter offline predictor; it is not part of online inference."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def read_image(dataset_root: Path, video: str) -> tuple[np.ndarray, tuple[int, int]]:
    image = Image.open(dataset_root / "annotations" / video / "reference.jpg").convert("RGB")
    size = image.size
    resized = np.asarray(image.resize((MAP_SIZE, MAP_SIZE), Image.Resampling.BILINEAR), np.float32) / 255.0
    return resized, size


def weak_labels(dataset_root: Path, video: str, image_wh: tuple[int, int], pre) -> tuple[np.ndarray, np.ndarray]:
    tracks = pre.read_pedestrian_tracks(dataset_root / "annotations" / video / "annotations.txt")
    positive = np.zeros((MAP_SIZE, MAP_SIZE), np.uint8)
    sx, sy = MAP_SIZE / image_wh[0], MAP_SIZE / image_wh[1]
    for points in tracks.values():
        ordered = [points[frame] for frame in sorted(points)]
        if not ordered:
            continue
        xy = np.asarray([[round(p[0] * sx), round(p[1] * sy)] for p in ordered], np.int32)
        xy[:, 0] = np.clip(xy[:, 0], 0, MAP_SIZE - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, MAP_SIZE - 1)
        if len(xy) == 1:
            cv2.circle(positive, tuple(xy[0]), 4, 1, -1)
        else:
            cv2.polylines(positive, [xy.reshape(-1, 1, 2)], False, 1, thickness=9)
    distance = cv2.distanceTransform((1 - positive).astype(np.uint8), cv2.DIST_L2, 5)
    weak_negative = distance >= 18.0
    labels = positive.astype(np.float32)
    labels[weak_negative] = 0.0
    known = positive.astype(bool) | weak_negative
    return labels, known.astype(np.float32)


def prepare(dataset_root: Path, videos: list[str], pre):
    images, labels, known, sizes = [], [], [], []
    for video in videos:
        image, size = read_image(dataset_root, video)
        label, mask = weak_labels(dataset_root, video, size, pre)
        images.append(image.transpose(2, 0, 1)); labels.append(label[None]); known.append(mask[None]); sizes.append(size)
    return (
        torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(labels)),
        torch.from_numpy(np.stack(known)), np.asarray(sizes, np.int32),
    )


def masked_loss(logits, target, known):
    positive = ((target > 0.5) & (known > 0.5)).sum().clamp_min(1)
    negative = ((target <= 0.5) & (known > 0.5)).sum().clamp_min(1)
    weight = torch.where(target > 0.5, negative.float() / positive.float(), torch.ones_like(target))
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight * known
    return raw.sum() / (weight * known).sum().clamp_min(1.0)


def fit(model, images, labels, known, indices, device, epochs, seed):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(indices), generator=generator).tolist()
        total = 0.0
        for begin in range(0, len(order), 4):
            chosen = [indices[order[x]] for x in range(begin, min(begin + 4, len(order)))]
            x, y, mask = images[chosen].to(device), labels[chosen].to(device), known[chosen].to(device)
            if (epoch + begin) % 2:
                x, y, mask = x.flip(-1), y.flip(-1), mask.flip(-1)
            gain = 0.9 + 0.2 * torch.rand((len(chosen), 1, 1, 1), generator=generator).to(device)
            x = (x * gain).clamp(0, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x), y, mask)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(chosen)
        if epoch in {0, epochs - 1} or (epoch + 1) % 10 == 0:
            print(f"[walkability] epoch={epoch + 1}/{epochs} loss={total / len(indices):.6f}", flush=True)


@torch.inference_mode()
def predict(model, images, device):
    model.eval(); output = []
    for begin in range(0, len(images), 8):
        output.append(torch.sigmoid(model(images[begin:begin + 8].to(device))).cpu().numpy()[:, 0])
    return np.concatenate(output)


def choose_threshold(probability, labels, known, indices):
    best = None
    y = labels[indices, 0].numpy(); mask = known[indices, 0].numpy() > 0.5; p = probability[indices]
    for threshold in np.linspace(0.20, 0.80, 31):
        prediction = p >= threshold
        tp = np.logical_and(prediction, y > 0.5)[mask].sum(); fp = np.logical_and(prediction, y <= 0.5)[mask].sum()
        fn = np.logical_and(~prediction, y > 0.5)[mask].sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if best is None or f1 > best[0]:
            best = (float(f1), float(threshold))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=178)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve(); bank_path = output / "topology_bank.npz"
    if bank_path.is_file() and not args.force:
        print(f"[skip] {bank_path}"); return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pre = load_module("exp178_preprocess", PREPROCESS)
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    train = list(manifest["videos"]["train"]); val = list(manifest["videos"]["val"]); test = list(manifest["videos"]["test"])
    videos = train + val + test
    images, labels, known, sizes = prepare(args.dataset_root.resolve(), videos, pre)
    calibration = list(range(0, len(train), 5)); fitting = [x for x in range(len(train)) if x not in calibration]
    model = TinyWalkabilityCNN().to(device)
    fit(model, images, labels, known, fitting, device, args.epochs, args.seed)
    probabilities = predict(model, images, device)
    f1, threshold = choose_threshold(probabilities, labels, known, calibration)
    fields = [probability_to_topology(prob, threshold) for prob in probabilities]
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "probability": probabilities.astype(np.float16), "image_wh": sizes,
        "video_names": np.asarray(videos), "threshold": np.asarray([threshold], np.float32),
    }
    for key in fields[0]:
        payload[key] = np.stack([field[key] for field in fields])
    np.savez_compressed(bank_path, **payload)
    torch.save(model.state_dict(), output / "walkability_cnn.pth")
    contract = {
        "experiment": "EXP-178A", "map_size": MAP_SIZE, "training_videos": len(fitting),
        "calibration_videos": [train[x] for x in calibration], "validation_and_test_labels_used": False,
        "threshold": threshold, "calibration_weak_label_f1": f1,
        "offline_parameters": sum(p.numel() for p in model.parameters()),
        "positive_radius_resized_px": 4, "weak_negative_min_distance_resized_px": 18,
        "epochs": args.epochs, "seed": args.seed,
        "per_video_skeleton_pixels": {
            video: int(field["skeleton"].sum()) for video, field in zip(videos, fields)
        },
        "empty_topology_videos": [
            video for video, field in zip(videos, fields) if int(field["skeleton"].sum()) == 0
        ],
    }
    (output / "asset_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
