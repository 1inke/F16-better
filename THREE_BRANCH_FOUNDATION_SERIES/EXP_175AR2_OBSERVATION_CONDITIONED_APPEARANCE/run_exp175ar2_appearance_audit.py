from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter
from scipy.stats import rankdata
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
DATASET = Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset")
DATA_DIR = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
SIDE_DIR = ROOT / "SCENE_QUERY_CURRENT_DECODER_SDD_SERIES" / "processed_scene_assets"
BASE_AUDIT = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175A_SCENE_UTILITY_AUDIT" / "run_exp175a_scene_utility_audit.py"
VARIANTS = ("appearance_similarity", "appearance_plus_continuity", "endpoint_appearance")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_map(image: Image.Image, size: int) -> np.ndarray:
    rgb = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV").resize((size, size), Image.Resampling.BILINEAR), np.float32) / 255.0
    gray = rgb.mean(axis=-1)
    mean = uniform_filter(gray, size=7, mode="reflect")
    variance = np.maximum(uniform_filter(gray * gray, size=7, mode="reflect") - mean * mean, 0.0)
    texture = np.sqrt(variance)
    gy, gx = np.gradient(gray)
    gradient = np.sqrt(gx * gx + gy * gy)
    texture = np.clip(texture / 0.20, 0.0, 1.0)
    gradient = np.clip(gradient / 0.25, 0.0, 1.0)
    channels = np.concatenate((rgb, hsv, texture[..., None], gradient[..., None]), axis=-1)
    return np.rint(channels.transpose(2, 0, 1) * 255.0).astype(np.uint8)


def ensure_bank(path: Path, dataset_root: Path, map_size: int, force: bool, preflight: bool):
    if path.exists() and not force and not preflight:
        print(f"[skip bank] {path}", flush=True); return
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    videos = manifest["videos"]["train"] + manifest["videos"]["val"] + manifest["videos"]["test"]
    selected = videos[:1] if preflight else videos
    maps, sizes = [], []
    for i, video in enumerate(selected):
        image = Image.open(dataset_root / "annotations" / video / "reference.jpg").convert("RGB")
        maps.append(make_map(image, map_size)); sizes.append((image.width, image.height))
        print(f"[appearance {i + 1:02d}/{len(selected)}] {video}", flush=True)
    if preflight:
        print(json.dumps({"status": "bank_preflight_ok", "shape": list(np.stack(maps).shape)}, indent=2)); return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, maps=np.stack(maps), image_wh=np.asarray(sizes, np.float32),
        video_names=np.asarray(videos), pixel_scale=np.asarray([20.0], np.float32),
        channel_names=np.asarray(["r", "g", "b", "h", "s", "v", "texture", "gradient"]),
    )


class AppearanceBank:
    def __init__(self, path: Path, device: torch.device):
        bank = np.load(path, allow_pickle=False)
        self.maps = torch.from_numpy(np.asarray(bank["maps"], np.uint8)).to(device)
        self.image_wh = torch.from_numpy(np.asarray(bank["image_wh"], np.float32)).to(device)
        self.video_names = [str(x) for x in bank["video_names"].tolist()]
        self.pixel_scale = float(np.asarray(bank["pixel_scale"]).reshape(-1)[0])

    def query(self, points: torch.Tensor, scene_ids: torch.Tensor):
        sizes = self.image_wh[scene_ids]
        pixels = points * self.pixel_scale
        sizes = sizes.view(points.shape[0], *([1] * (points.ndim - 2)), 2)
        x = 2.0 * pixels[..., 0] / (sizes[..., 0] - 1.0) - 1.0
        y = 2.0 * pixels[..., 1] / (sizes[..., 1] - 1.0) - 1.0
        inside = (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
        original_shape = points.shape[:-1]
        grid = torch.stack((x, y), dim=-1).reshape(points.shape[0], -1, 1, 2)
        maps = self.maps[scene_ids].to(points.dtype) / 255.0
        values = F.grid_sample(maps, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        values = values[..., 0].transpose(1, 2).reshape(*original_shape, maps.shape[1])
        return values, inside


def embedding(values: torch.Tensor) -> torch.Tensor:
    rgb, h, s, v, texture, gradient = values[..., :3], values[..., 3], values[..., 4], values[..., 5], values[..., 6], values[..., 7]
    angle = 2.0 * torch.pi * h
    return torch.cat((
        0.45 * rgb, torch.sin(angle)[..., None], torch.cos(angle)[..., None],
        s[..., None], v[..., None], 0.60 * texture[..., None], 0.35 * gradient[..., None],
    ), dim=-1)


def appearance_scores(candidate_values, candidate_inside, prototype_values, prototype_mask):
    cand = embedding(candidate_values)
    proto = embedding(prototype_values).flatten(1, -2)
    mask = prototype_mask.flatten(1)
    distance = torch.sqrt(((cand[..., None, :] - proto[:, None, None]) ** 2).mean(dim=-1).clamp_min(1e-10))
    distance = distance.masked_fill(~mask[:, None, None], float("inf"))
    nearest = distance.amin(dim=-1)
    point_similarity = torch.exp(-nearest / 0.18) * candidate_inside.to(cand.dtype)
    path_similarity = point_similarity.mean(dim=-1)
    change = torch.sqrt(((cand[..., 1:, :] - cand[..., :-1, :]) ** 2).mean(dim=-1).clamp_min(1e-10))
    continuity = torch.exp(-change / 0.18).mean(dim=-1)
    return {
        "appearance_similarity": path_similarity,
        "appearance_plus_continuity": 0.80 * path_similarity + 0.20 * continuity,
        "endpoint_appearance": 0.70 * path_similarity + 0.30 * point_similarity[..., -1],
    }


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ar, br = rankdata(a, axis=1, method="average"), rankdata(b, axis=1, method="average")
    ar -= ar.mean(1, keepdims=True); br -= br.mean(1, keepdims=True)
    denominator = np.sqrt((ar * ar).sum(1) * (br * br).sum(1))
    result = np.full(len(a), np.nan)
    valid = denominator > 1e-12
    result[valid] = (ar[valid] * br[valid]).sum(1) / denominator[valid]
    return result


def collect(split, model, cfg, trainer, base, bank, device, batch_size, max_samples):
    loader = base.make_loader(split, cfg, trainer, batch_size, max_samples)
    pieces: dict[str, list[np.ndarray]] = {}
    def save(name, value): pieces.setdefault(name, []).append(value.detach().cpu().numpy())
    with torch.inference_mode():
        for absolute_past, absolute_future, num_valid, scene_ids, *_ in loader:
            absolute_past, absolute_future = absolute_past.to(device), absolute_future.to(device)
            num_valid, scene_ids = num_valid.to(device).int(), scene_ids.to(device).long()
            past, future = trainer.normalize_to_focal_heading(absolute_past, absolute_future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            prediction = (output[2] + center.unsqueeze(-3))[:, 0]
            distance = torch.linalg.vector_norm(prediction - future[:, 0, None], dim=-1)
            quality = distance.mean(-1) + 0.25 * distance[..., -1]
            global_prediction = base.local_to_global(prediction, absolute_past)
            candidate_values, candidate_inside = bank.query(global_prediction, scene_ids)
            prototype_values, prototype_inside = bank.query(absolute_past, scene_ids)
            ids = torch.arange(absolute_past.shape[1], device=device)[None] < num_valid[:, None]
            prototype_mask = prototype_inside & ids[..., None]
            scores = appearance_scores(candidate_values, candidate_inside, prototype_values, prototype_mask)
            gt_values, gt_inside = bank.query(absolute_future[:, :1], scene_ids)
            gt_scores = appearance_scores(gt_values, gt_inside, prototype_values, prototype_mask)
            save("scene_id", scene_ids); save("quality", quality)
            logits = getattr(model, "last_mode_logits")[:, 0]
            save("logits", logits); save("candidate_in_bounds", candidate_inside.float().mean(-1))
            for name, value in scores.items(): save(name, value)
            for name, value in gt_scores.items(): save("gt_" + name, value[:, 0])
    return {key: np.concatenate(value) for key, value in pieces.items()}


def group_metrics(data, mask, variant):
    if not mask.any(): return {"samples": 0}
    quality, feasibility = data["quality"][mask], data[variant][mask]
    rows = np.arange(len(quality)); oracle = quality.argmin(1); top1 = data["logits"][mask].argmax(1)
    violation = 1.0 - feasibility
    ov = violation[rows, oracle]
    remaining = (violation.sum(1) - ov) / (violation.shape[1] - 1)
    rho = spearman_rows(feasibility, -quality)
    return {
        "samples": int(mask.sum()), "spearman": float(np.nanmean(rho)),
        "valid_spearman_fraction": float(np.isfinite(rho).mean()),
        "oracle_violation": float(ov.mean()), "remaining_violation": float(remaining.mean()),
        "oracle_reduction": float((remaining.mean() - ov.mean()) / max(remaining.mean(), 1e-8)),
        "oracle_lower": bool(ov.mean() < remaining.mean()),
        "top1_violation": float(violation[rows, top1].mean()),
        "gt_feasibility": float(data["gt_" + variant][mask].mean()),
        "candidate_in_bounds": float(data["candidate_in_bounds"][mask].mean()),
    }


def analyze(split, data, bank, variant):
    ids = data["scene_id"].astype(int); rows = []
    for scene_id in sorted(np.unique(ids)):
        row = {"split": split, "variant": variant, "scene_id": int(scene_id), "video": bank.video_names[scene_id]}
        row.update(group_metrics(data, ids == scene_id, variant)); rows.append(row)
    macro_rho = float(np.nanmean([r["spearman"] for r in rows]))
    macro_reduction = float(np.mean([r["oracle_reduction"] for r in rows]))
    return {
        "overall": group_metrics(data, np.ones(len(ids), bool), variant),
        "video_macro_spearman": macro_rho, "video_macro_oracle_reduction": macro_reduction,
        "positive_spearman_videos": int(sum(r["spearman"] > 0 for r in rows)),
        "oracle_lower_videos": int(sum(r["oracle_lower"] for r in rows)), "video_count": len(rows),
    }, rows


def main():
    parser = argparse.ArgumentParser(description="EXP-175A-R2 observation-conditioned appearance audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--map-size", type=int, default=320)
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument("--bank", type=Path, default=EXP_DIR / "assets" / "appearance_bank.npz")
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "results" / "seed_1")
    parser.add_argument("--force-bank", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    ensure_bank(args.bank.resolve(), args.dataset_root.resolve(), args.map_size, args.force_bank, args.preflight_only)
    if args.preflight_only: return
    device = torch.device(args.device)
    base = load_module("exp175ar2_base", BASE_AUDIT)
    model, cfg, trainer = base.build_model(args.seed, device)
    bank = AppearanceBank(args.bank.resolve(), device)
    collected = {}
    for split in ("validation", "test"):
        print(f"[collect] {split}", flush=True)
        collected[split] = collect(split, model, cfg, trainer, base, bank, device, args.batch_size, args.max_samples)
    validation_results, test_results, all_rows = {}, {}, []
    for variant in VARIANTS:
        validation_results[variant], rows = analyze("validation", collected["validation"], bank, variant); all_rows += rows
    selected = max(VARIANTS, key=lambda v: validation_results[v]["video_macro_spearman"])
    test_results[selected], rows = analyze("test", collected["test"], bank, selected); all_rows += rows
    val, test = validation_results[selected], test_results[selected]
    gate = {
        "selected_on_validation_only": selected,
        "validation_same_direction": val["video_macro_spearman"] > 0 and val["video_macro_oracle_reduction"] > 0,
        "validation_relation_gate": val["video_macro_spearman"] >= 0.20 or val["video_macro_oracle_reduction"] >= 0.20,
        "test_12_of_17_gate": test["oracle_lower_videos"] >= 12,
    }
    gate["passed"] = bool(all(v for k, v in gate.items() if k != "selected_on_validation_only"))
    payload = {
        "experiment": "EXP-175A-R2", "training": False,
        "no_future_used_for_scene_prototype": True, "selection_protocol": "choose fixed score on validation; test once",
        "validation_all_variants": validation_results, "selected_variant": selected,
        "test_selected_variant": test_results[selected], "gate": gate,
    }
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "per_video.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    print(json.dumps(gate, ensure_ascii=False, indent=2)); print(f"[result] {out / 'summary.json'}")


if __name__ == "__main__":
    main()
