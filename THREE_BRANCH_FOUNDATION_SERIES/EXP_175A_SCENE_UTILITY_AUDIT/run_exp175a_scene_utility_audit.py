from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
ASTRA_ROOT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA_DIR = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
ASSET_DIR = ROOT / "SCENE_QUERY_CURRENT_DECODER_SDD_SERIES" / "processed_scene_assets"
BASELINE = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175S0_FINAL_JOINT_V2_SDD_BASELINE"
CFG_PATH = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
TRAINER_PATH = ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
MODEL_PATH = ROOT / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
ADAPTER_PATH = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
VARIANT = "f16_temporal_social_cross"
ANCHOR_INDICES = (3, 7, 11)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(device: torch.device):
    os.chdir(ASTRA_ROOT)
    for path in (ASTRA_ROOT, ASTRA_ROOT / "scripts", ROOT / "FRONT_ENCODER_SERIES"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20

    cfg = load_locked_q20(CFG_PATH)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 16
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    return cfg


def checkpoint(seed: int, stage: str) -> Path:
    folder = "h3_anchor_proposal" if stage == "h3" else "joint_v2"
    return (
        BASELINE / "results" / folder / VARIANT / f"seed_{seed}"
        / "val_scene_1" / "best_selected.pth"
    )


def build_model(seed: int, device: torch.device):
    cfg = configure(device)
    trainer = load_module("exp175a_trainer", TRAINER_PATH)
    models = load_module("exp175a_models", MODEL_PATH)
    adapter = load_module("exp175a_joint_v2", ADAPTER_PATH)
    h3, final = checkpoint(seed, "h3"), checkpoint(seed, "final")
    for path in (h3, final):
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3)
    os.environ["EXP173_DECODER_INIT_SEED"] = str(seed)
    model = models.FrontEncoderASTRA(cfg, VARIANT)
    adapter.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
    model.load_state_dict(torch.load(final, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval(), cfg, trainer


class SemanticBank:
    def __init__(self, path: Path, device: torch.device):
        bank = np.load(path, allow_pickle=False)
        self.maps = torch.from_numpy(np.asarray(bank["maps"], dtype=np.uint8)).to(device)
        self.image_wh = torch.from_numpy(np.asarray(bank["image_wh"], dtype=np.float32)).to(device)
        self.video_names = [str(x) for x in bank["video_names"].tolist()]
        self.channels = [str(x) for x in bank["channel_names"].tolist()]
        self.pixel_scale = float(np.asarray(bank["pixel_scale"]).reshape(-1)[0])
        expected = ["walkable", "obstacle", "vegetation", "structure", "boundary", "signed_distance"]
        if self.channels != expected or self.maps.ndim != 4:
            raise RuntimeError(f"Unexpected semantic bank layout: {self.channels}, {self.maps.shape}")

    def query(self, global_points: torch.Tensor, scene_ids: torch.Tensor):
        """Query [B,K,T,2] points; return semantics and strict in-frame mask."""
        sizes = self.image_wh[scene_ids]
        pixels = global_points * self.pixel_scale
        x = 2.0 * pixels[..., 0] / (sizes[:, None, None, 0] - 1.0) - 1.0
        y = 2.0 * pixels[..., 1] / (sizes[:, None, None, 1] - 1.0) - 1.0
        in_bounds = (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
        grid = torch.stack((x, y), dim=-1).reshape(global_points.shape[0], -1, 1, 2)
        maps = self.maps[scene_ids].to(dtype=global_points.dtype) / 255.0
        sampled = F.grid_sample(
            maps, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )[..., 0].transpose(1, 2)
        sampled = sampled.reshape(*global_points.shape[:-1], self.maps.shape[1])
        return sampled, in_bounds


def local_to_global(local: torch.Tensor, absolute_past: torch.Tensor) -> torch.Tensor:
    lag = min(3, absolute_past.shape[-2] - 1)
    displacement = absolute_past[:, 0, -1] - absolute_past[:, 0, -1 - lag]
    speed = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    fallback = torch.zeros_like(displacement)
    fallback[:, 0] = 1.0
    heading = torch.where(speed > 0.03, displacement / speed.clamp_min(1e-6), fallback)
    cosine = heading[:, 0, None, None]
    sine = heading[:, 1, None, None]
    x = local[..., 0] * cosine - local[..., 1] * sine
    y = local[..., 0] * sine + local[..., 1] * cosine
    return absolute_past[:, 0, -1, None, None] + torch.stack((x, y), dim=-1)


def scene_features(sampled: torch.Tensor, in_bounds: torch.Tensor) -> dict[str, torch.Tensor]:
    walk, obstacle, _, structure, boundary, signed = sampled.unbind(dim=-1)
    valid = in_bounds.to(sampled.dtype)
    blocking = torch.maximum(obstacle, structure)
    feasibility_t = walk * (1.0 - blocking) * valid
    signed_inside = (signed >= 0.5) & in_bounds
    offroad_t = (~signed_inside).to(sampled.dtype)
    crossings = (signed_inside[..., 1:] != signed_inside[..., :-1]).sum(dim=-1).to(sampled.dtype)
    signed_distance = (2.0 * signed - 1.0).abs() * 12.0
    signed_distance = torch.where(in_bounds, signed_distance, torch.zeros_like(signed_distance))
    anchors = torch.as_tensor(ANCHOR_INDICES, device=sampled.device)
    return {
        "feasibility": feasibility_t.mean(dim=-1),
        "violation": 1.0 - feasibility_t.mean(dim=-1),
        "walkability": (walk * valid).mean(dim=-1),
        "offroad_ratio": offroad_t.mean(dim=-1),
        "obstacle_ratio": (blocking * valid).mean(dim=-1),
        "boundary_contact_ratio": (boundary * valid).mean(dim=-1),
        "boundary_crossings": crossings,
        "endpoint_walkability": feasibility_t[..., -1],
        "anchor_walkability": feasibility_t.index_select(-1, anchors).mean(dim=-1),
        "min_boundary_distance_map_px": signed_distance.min(dim=-1).values,
        "in_bounds_ratio": valid.mean(dim=-1),
    }


def average_rank_rows(values: np.ndarray) -> np.ndarray:
    return rankdata(values, axis=1, method="average")


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ar, br = average_rank_rows(a), average_rank_rows(b)
    ar -= ar.mean(axis=1, keepdims=True)
    br -= br.mean(axis=1, keepdims=True)
    den = np.sqrt((ar * ar).sum(axis=1) * (br * br).sum(axis=1))
    out = np.full(len(a), np.nan, dtype=np.float64)
    valid = den > 1e-12
    out[valid] = (ar[valid] * br[valid]).sum(axis=1) / den[valid]
    return out


def make_loader(split: str, cfg, trainer, batch_size: int, max_samples: int):
    if split == "validation":
        prefix, sidecar = "sdd_standard_train_val", ASSET_DIR / "sdd_train_val_scene_ids.npy"
    else:
        prefix, sidecar = "sdd_standard_test", ASSET_DIR / "sdd_test_scene_ids.npy"
    dataset = trainer.TrajectoryDataset(
        DATA_DIR / f"{prefix}_data.npy", DATA_DIR / f"{prefix}_num.npy", cfg,
        sample_scene_path=sidecar,
    )
    if split == "validation":
        scene_ids = np.load(sidecar, mmap_mode="r")
        manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
        val_start = len(manifest["videos"]["train"])
        indices = np.flatnonzero(scene_ids >= val_start).tolist()
    else:
        indices = list(range(len(dataset)))
    if max_samples > 0:
        indices = indices[:max_samples]
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )


def collect(split: str, model, cfg, trainer, bank: SemanticBank, device, batch_size, max_samples):
    loader = make_loader(split, cfg, trainer, batch_size, max_samples)
    pieces: dict[str, list[np.ndarray]] = {}

    def save(name: str, value: torch.Tensor):
        pieces.setdefault(name, []).append(value.detach().cpu().numpy())

    with torch.inference_mode():
        for absolute_past, absolute_future, num_valid, scene_ids, *_ in loader:
            absolute_past = absolute_past.to(device, non_blocking=True)
            absolute_future = absolute_future.to(device, non_blocking=True)
            num_valid = num_valid.to(device, non_blocking=True).int()
            scene_ids = scene_ids.to(device, non_blocking=True).long()
            past, future = trainer.normalize_to_focal_heading(absolute_past, absolute_future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(
                past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past
            )
            prediction = output[2] if isinstance(output, (tuple, list)) else output
            if prediction.ndim != 5 or prediction.shape[2] != 20:
                raise RuntimeError(f"Expected [B,A,20,T,2], got {tuple(prediction.shape)}")
            prediction = (prediction + center.unsqueeze(-3))[:, 0]
            target = future[:, 0, None]
            distance = torch.linalg.vector_norm(prediction - target, dim=-1)
            ade, fde = distance.mean(dim=-1), distance[..., -1]
            quality = ade + 0.25 * fde
            logits = getattr(model, "last_mode_logits", None)
            if logits is None:
                raise RuntimeError("Joint-V2 did not expose last_mode_logits")
            logits = logits[:, 0]

            global_prediction = local_to_global(prediction, absolute_past)
            sampled, in_bounds = bank.query(global_prediction, scene_ids)
            features = scene_features(sampled, in_bounds)
            gt_sampled, gt_bounds = bank.query(absolute_future[:, 0, None], scene_ids)
            past_sampled, past_bounds = bank.query(absolute_past[:, 0, None], scene_ids)
            gt_feature = scene_features(gt_sampled, gt_bounds)

            save("scene_id", scene_ids)
            save("logits", logits)
            save("ade", ade)
            save("fde", fde)
            save("quality", quality)
            for name, value in features.items():
                save(name, value)
            save("gt_feasibility", gt_feature["feasibility"][:, 0])
            save("gt_in_bounds", gt_bounds.float().mean(dim=(-1, -2)))
            save("past_in_bounds", past_bounds.float().mean(dim=(-1, -2)))
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def subset_metrics(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float | int]:
    if not np.any(mask):
        return {"samples": 0}
    quality = data["quality"][mask]
    violation = data["violation"][mask]
    feasibility = data["feasibility"][mask]
    logits = data["logits"][mask]
    rows = np.arange(len(quality))
    oracle = quality.argmin(axis=1)
    top1 = logits.argmax(axis=1)
    oracle_v = violation[rows, oracle]
    top1_v = violation[rows, top1]
    remaining_v = (violation.sum(axis=1) - oracle_v) / (violation.shape[1] - 1)
    random_v = violation.mean(axis=1)
    correlations = spearman_rows(feasibility, -quality)
    rel_reduction = (remaining_v.mean() - oracle_v.mean()) / max(remaining_v.mean(), 1e-8)
    result: dict[str, float | int] = {
        "samples": int(mask.sum()),
        "scene_spearman_feasibility_neg_quality": float(np.nanmean(correlations)),
        "scene_spearman_valid_sample_fraction": float(np.isfinite(correlations).mean()),
        "oracle_violation": float(oracle_v.mean()),
        "top1_violation": float(top1_v.mean()),
        "remaining_violation": float(remaining_v.mean()),
        "random_violation": float(random_v.mean()),
        "oracle_violation_relative_reduction_vs_remaining": float(rel_reduction),
        "oracle_lower_violation_than_remaining": bool(oracle_v.mean() < remaining_v.mean()),
        "candidate_in_bounds_ratio": float(data["in_bounds_ratio"][mask].mean()),
        "gt_in_bounds_ratio": float(data["gt_in_bounds"][mask].mean()),
        "past_in_bounds_ratio": float(data["past_in_bounds"][mask].mean()),
        "gt_feasibility": float(data["gt_feasibility"][mask].mean()),
    }
    for key in (
        "walkability", "offroad_ratio", "obstacle_ratio", "boundary_contact_ratio",
        "boundary_crossings", "endpoint_walkability", "anchor_walkability",
        "min_boundary_distance_map_px",
    ):
        values = data[key][mask]
        result[f"oracle_{key}"] = float(values[rows, oracle].mean())
        result[f"top1_{key}"] = float(values[rows, top1].mean())
        result[f"remaining_{key}"] = float(
            ((values.sum(axis=1) - values[rows, oracle]) / (values.shape[1] - 1)).mean()
        )
    return result


def analyze(split: str, data: dict[str, np.ndarray], bank: SemanticBank):
    scene_ids = data["scene_id"].astype(np.int64)
    overall = subset_metrics(data, np.ones(len(scene_ids), dtype=bool))
    per_video = []
    for scene_id in sorted(np.unique(scene_ids).tolist()):
        row = {"split": split, "scene_id": scene_id, "video": bank.video_names[scene_id]}
        row.update(subset_metrics(data, scene_ids == scene_id))
        per_video.append(row)
    little = np.asarray([bank.video_names[x].startswith("little/") for x in scene_ids])
    groups = {
        "little": subset_metrics(data, little),
        "non_little": subset_metrics(data, ~little),
    }
    macro_spearman = float(np.mean([r["scene_spearman_feasibility_neg_quality"] for r in per_video]))
    macro_reduction = float(np.mean([
        r["oracle_violation_relative_reduction_vs_remaining"] for r in per_video
    ]))
    lower_count = int(sum(bool(r["oracle_lower_violation_than_remaining"]) for r in per_video))
    return {
        "overall": overall,
        "groups": groups,
        "video_macro_spearman": macro_spearman,
        "video_macro_oracle_violation_reduction": macro_reduction,
        "videos_oracle_lower_violation": lower_count,
        "video_count": len(per_video),
    }, per_video


def write_outputs(output_dir: Path, payload: dict, rows: list[dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        keys = list(rows[0])
        with (output_dir / "per_video_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def verify_inputs(seed: int):
    required = [
        ASTRA_ROOT, DATA_DIR / "manifest.json", DATA_DIR / "sdd_standard_train_val_data.npy",
        DATA_DIR / "sdd_standard_train_val_num.npy", DATA_DIR / "sdd_standard_test_data.npy",
        DATA_DIR / "sdd_standard_test_num.npy", ASSET_DIR / "sdd_semantic_bank.npz",
        ASSET_DIR / "sdd_train_val_scene_ids.npy", ASSET_DIR / "sdd_test_scene_ids.npy",
        checkpoint(seed, "h3"), checkpoint(seed, "final"), CFG_PATH, TRAINER_PATH, MODEL_PATH,
        ADAPTER_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing EXP-175A inputs:\n" + "\n".join(missing))


def main():
    global ASSET_DIR
    parser = argparse.ArgumentParser(description="EXP-175A zero-training SDD scene-utility audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples per split")
    parser.add_argument("--preflight-only", action="store_true", help="load model and audit one small test batch")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--asset-dir", type=Path, default=None, help="Override semantic bank and scene-ID sidecars")
    args = parser.parse_args()
    if args.asset_dir is not None:
        ASSET_DIR = args.asset_dir.resolve()
    if args.output_dir is None:
        args.output_dir = EXP_DIR / "results" / f"seed_{args.seed}"
    verify_inputs(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model, cfg, trainer = build_model(args.seed, device)
    bank = SemanticBank(ASSET_DIR / "sdd_semantic_bank.npz", device)

    if args.preflight_only:
        data = collect("test", model, cfg, trainer, bank, device, min(args.batch_size, 8), 8)
        result, rows = analyze("test_smoke", data, bank)
        print(json.dumps({
            "status": "preflight_ok", "checkpoint": str(checkpoint(args.seed, "final")),
            "semantic_maps": list(bank.maps.shape), "samples": result["overall"]["samples"],
            "modes": int(data["quality"].shape[1]), "videos_seen": len(rows),
            "candidate_in_bounds_ratio": result["overall"]["candidate_in_bounds_ratio"],
        }, ensure_ascii=False, indent=2))
        return

    split_results, all_rows = {}, []
    for split in ("validation", "test"):
        print(f"[collect] {split}", flush=True)
        data = collect(split, model, cfg, trainer, bank, device, args.batch_size, args.max_samples)
        result, rows = analyze(split, data, bank)
        split_results[split] = result
        all_rows.extend(rows)
        print(f"[done] {split}: samples={result['overall']['samples']}", flush=True)

    val, test = split_results["validation"], split_results["test"]
    alignment_ok = (
        val["overall"]["gt_in_bounds_ratio"] >= 0.98
        and test["overall"]["gt_in_bounds_ratio"] >= 0.98
    )
    same_direction = (
        val["video_macro_spearman"] > 0.0
        and val["video_macro_oracle_violation_reduction"] > 0.0
    )
    relation_gate = (
        val["video_macro_spearman"] >= 0.20
        or val["video_macro_oracle_violation_reduction"] >= 0.20
    )
    video_gate = test["videos_oracle_lower_violation"] >= 12
    passed = bool(alignment_ok and same_direction and relation_gate and video_gate)
    decision = "PROCEED_TO_EXP175B" if passed else (
        "STOP_AND_FIX_ALIGNMENT" if not alignment_ok else "STOP_SCENE_MODULE_AND_REVIEW_FEATURES"
    )
    payload = {
        "experiment": "EXP-175A",
        "purpose": "zero-training scene utility audit for the frozen current SDD decoder",
        "seed": args.seed,
        "model": "F16 + H3 + Joint-V2 (EXP-175S0)",
        "semantic_asset": "offline SegFormer-B0 ADE20K, 6 channels, no online vision compute",
        "quality_definition": "ADE + 0.25 * FDE in model units",
        "splits": split_results,
        "preregistered_gate": {
            "alignment_ok_gt_in_bounds_at_least_0.98": alignment_ok,
            "test_videos_oracle_lower_at_least_12_of_17": video_gate,
            "validation_same_direction": same_direction,
            "validation_macro_spearman_at_least_0.20_or_reduction_at_least_0.20": relation_gate,
            "passed": passed,
            "decision": decision,
        },
    }
    write_outputs(args.output_dir.resolve(), payload, all_rows)
    print(json.dumps(payload["preregistered_gate"], ensure_ascii=False, indent=2))
    print(f"[result] {args.output_dir.resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
