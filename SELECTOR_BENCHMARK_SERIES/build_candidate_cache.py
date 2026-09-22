#!/usr/bin/env python
"""
Build Candidate Cache for F16 Trajectory Generator.

Generates out-of-fold (OOF) candidate trajectory predictions and ground truth metrics
on test splits for ETH/UCY folds.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

COLLISION_THRESHOLD = 0.20
NEAR_THRESHOLD = 0.50


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_astra_env(repo_root: Path, astra_root: Path, device: torch.device):
    os.chdir(astra_root)
    sys.path[:0] = [
        str(astra_root),
        str(astra_root / "scripts"),
        str(repo_root / "FRONT_ENCODER_SERIES"),
    ]
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20

    cfg_path = repo_root / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
    cfg = load_locked_q20(cfg_path)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    return cfg


def build_f16_model(repo_root: Path, fold: str, seed: int, cfg, device: torch.device):
    exp173 = repo_root / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results"
    h3 = exp173 / fold / "h3" / "e3" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"
    final = exp173 / fold / "v2" / "e3" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"

    if not h3.is_file():
        raise FileNotFoundError(f"Missing H3 checkpoint: {h3}")
    if not final.is_file():
        raise FileNotFoundError(f"Missing V2 checkpoint: {final}")

    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3)
    os.environ["EXP173_DECODER_INIT_SEED"] = str(seed)

    model_file = repo_root / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
    adapter_file = repo_root / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
    models_mod = load_module("front_encoder_models", model_file)
    adapter_mod = load_module("joint_v2_adapter", adapter_file)

    model = models_mod.FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    adapter_mod.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
    state_dict = torch.load(final, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def atomic_save_torch(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Build F16 Candidate Cache")
    parser.add_argument("--model", default="f16", choices=["f16"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--save-neighbor-distribution", action="store_true")
    parser.add_argument("--output-root", type=str, default="SELECTOR_BENCHMARK_SERIES/cache")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    astra_root = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"

    out_dir = (repo_root / args.output_root / args.model / f"seed_{args.seed}" / args.fold).resolve()
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists() and not args.force:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "complete":
                print(f"[CACHE] Already complete: {out_dir} ({manifest.get('total_samples')} samples)")
                return
        except Exception:
            pass

    print(f"[CACHE] Building candidate cache for fold={args.fold}, seed={args.seed}, split={args.split}...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = configure_astra_env(repo_root, astra_root, device)

    trainer_file = repo_root / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    trainer_mod = load_module("train_front_variant", trainer_file)

    model = build_f16_model(repo_root, args.fold, args.seed, cfg, device)

    data_file = data_dir / f"{args.fold}_data_{args.split}.npy"
    num_file = data_dir / f"{args.fold}_num_{args.split}.npy"
    if not data_file.exists() or not num_file.exists():
        raise FileNotFoundError(f"Data files missing: {data_file}, {num_file}")

    dataset = trainer_mod.TrajectoryDataset(data_file, num_file, cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    total_samples = len(dataset)
    print(f"[CACHE] Total samples: {total_samples} (device={device})")

    current_shard_samples: List[Dict[str, Any]] = []
    shard_files: List[str] = []
    shard_idx = 0
    global_idx = 0

    started = time.time()
    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            bs = past.shape[0]
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()

            past, future = trainer_mod.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            raw = output[2] if isinstance(output, (tuple, list)) else output
            pred = raw + center.unsqueeze(-3)  # (B, agents, 20, 12, 2)
            logits = model.last_mode_logits    # (B, agents, 20)

            focal_preds = pred[:, 0]    # (B, 20, 12, 2)
            focal_logits = logits[:, 0]  # (B, 20)
            focal_gt = future[:, 0]     # (B, 12, 2)

            nbr_preds = pred[:, 1:]     # (B, N-1, 20, 12, 2)
            nbr_logits = logits[:, 1:]   # (B, N-1, 20)
            nbr_gt = future[:, 1:]       # (B, N-1, 12, 2)

            # Privileged metrics computation
            cand_ade = torch.linalg.vector_norm(focal_preds - focal_gt[:, None, :, :], dim=-1).mean(dim=-1)  # (B, 20)
            cand_fde = torch.linalg.vector_norm(focal_preds[:, :, -1, :] - focal_gt[:, None, -1, :], dim=-1)  # (B, 20)

            for b in range(bs):
                nv = int(num_valid[b].item())
                valid_nbr_count = max(0, nv - 1)

                focal_b = focal_preds[b]  # (20, 12, 2)
                if valid_nbr_count > 0:
                    real_nbrs = nbr_gt[b, :valid_nbr_count]  # (valid_nbr_count, 12, 2)
                    dist = torch.linalg.vector_norm(focal_b[:, None, :, :] - real_nbrs[None, :, :, :], dim=-1)  # (20, valid_nbr_count, 12)
                    min_dist_per_step = dist.min(dim=-1).values  # (20, valid_nbr_count)
                    cand_min_dist = min_dist_per_step.min(dim=-1).values  # (20,)
                    cand_coll = (cand_min_dist < COLLISION_THRESHOLD)     # (20,)
                else:
                    cand_min_dist = torch.full((20,), 10.0, device=device)
                    cand_coll = torch.zeros((20,), dtype=torch.bool, device=device)

                sample = {
                    "sample_id": global_idx,
                    "fold": args.fold,
                    "seed": args.seed,
                    "past": past[b].float().cpu(),
                    "num_valid": nv,
                    "focal_candidates": focal_b.float().cpu(),
                    "focal_logits": focal_logits[b].float().cpu(),
                    "neighbor_candidates": nbr_preds[b, :valid_nbr_count].float().cpu() if valid_nbr_count > 0 else torch.empty((0, 20, 12, 2)),
                    "neighbor_logits": nbr_logits[b, :valid_nbr_count].float().cpu() if valid_nbr_count > 0 else torch.empty((0, 20)),
                    "focal_future_gt": focal_gt[b].float().cpu(),
                    "neighbor_future_gt": nbr_gt[b, :valid_nbr_count].float().cpu() if valid_nbr_count > 0 else torch.empty((0, 12, 2)),
                    "candidate_ade": cand_ade[b].float().cpu(),
                    "candidate_fde": cand_fde[b].float().cpu(),
                    "candidate_gt_min_clearance": cand_min_dist.float().cpu(),
                    "candidate_gt_collision": cand_coll.cpu(),
                }
                current_shard_samples.append(sample)
                global_idx += 1

                if len(current_shard_samples) >= args.shard_size:
                    shard_name = f"shard_{shard_idx:05d}.pt"
                    shard_path = out_dir / shard_name
                    atomic_save_torch(shard_path, current_shard_samples)
                    shard_files.append(shard_name)
                    print(f"  Saved {shard_name} ({len(current_shard_samples)} samples, total {global_idx}/{total_samples})")
                    current_shard_samples = []
                    shard_idx += 1

        if current_shard_samples:
            shard_name = f"shard_{shard_idx:05d}.pt"
            shard_path = out_dir / shard_name
            atomic_save_torch(shard_path, current_shard_samples)
            shard_files.append(shard_name)
            print(f"  Saved final {shard_name} ({len(current_shard_samples)} samples, total {global_idx}/{total_samples})")

    elapsed = time.time() - started
    manifest = {
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "split": args.split,
        "total_samples": global_idx,
        "shard_size": args.shard_size,
        "shards": shard_files,
        "status": "complete",
        "elapsed_seconds": elapsed,
        "timestamp": now_iso(),
    }
    atomic_write_json(manifest_path, manifest)
    print(f"[CACHE OK] Complete: {out_dir} ({global_idx} samples in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
