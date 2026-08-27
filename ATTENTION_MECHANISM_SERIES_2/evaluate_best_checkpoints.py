from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
DECODER = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS"
    / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)
RUNS = [
    ("f16_temporal_social_cross", "EXP_025_TEMPORAL_SOCIAL_CROSS_ATTENTION"),
    ("f35_temporal_half_pool", "EXP_052_TEMPORAL_HALF_POOL"),
    ("f38_recent_exact_early_pool", "EXP_057_RECENT_EXACT_EARLY_POOL"),
    ("f39_motion_event_sampling", "EXP_058_MOTION_EVENT_SAMPLING"),
    ("f40_motion_token_squeeze", "EXP_059_MOTION_TOKEN_SQUEEZE"),
    ("f41_edge_list_attention", "EXP_060_EDGE_LIST_ATTENTION"),
]
sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
os.chdir(PROJECT)

import sitecustomize  # noqa: E402,F401
from experiment_config import load_locked_q20  # noqa: E402
from front_encoder_models import FrontEncoderASTRA  # noqa: E402
from train_front_variant import TrajectoryDataset, evaluate, load_verifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=[name for name, _ in RUNS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--val-scenes", nargs="+", type=int, default=[1])
    args = parser.parse_args()
    device = torch.device("cuda:0")
    cfg = load_locked_q20(PROJECT / "configs/motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.device = str(device)
    cfg.device_list = [0]
    data_path = PROJECT / "datasets/eth_ucy/processed_data_diverse" / f"{cfg.SUBSET}_data_train.npy"
    num_path = PROJECT / "datasets/eth_ucy/processed_data_diverse" / f"{cfg.SUBSET}_num_train.npy"
    raw = np.load(data_path, mmap_mode="r")
    scene_ids = np.asarray(raw[:, 0, 0, 0], dtype=np.int64)
    dataset = TrajectoryDataset(data_path, num_path, cfg)
    spec = importlib.util.spec_from_file_location("experiment_decoder_models", DECODER)
    if spec is None or spec.loader is None:
        raise ImportError(DECODER)
    decoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder_module)

    selected = set(args.variants)
    for scene in args.val_scenes:
        validation = np.flatnonzero(scene_ids == scene)
        loader = DataLoader(
            Subset(dataset, validation.tolist()), batch_size=64, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        for seed in args.seeds:
            verifier_path = (
                PROJECT / "results/social_verifier_crossval/scale_0p030"
                / f"seed_{seed}/val_scene_{scene}/verifier_best.pth"
            )
            verifier = load_verifier(verifier_path, device) if verifier_path.is_file() else None
            for variant, folder in RUNS:
                if variant not in selected:
                    continue
                if variant == "f16_temporal_social_cross":
                    if scene != 1:
                        folder = "EXP_040_F16_CROSS_SCENE_VALIDATION"
                    elif seed != 1:
                        folder = "EXP_035_TEMPORAL_SOCIAL_CROSS_REPLICATION"
                elif variant == "f35_temporal_half_pool":
                    if scene != 1:
                        folder = "EXP_054_HALF_POOL_CROSS_SCENE"
                    elif seed != 1:
                        folder = "EXP_053_HALF_POOL_REPLICATION"
                run_dir = (
                    WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / folder / "results"
                    / variant / f"seed_{seed}" / f"val_scene_{scene}"
                )
                checkpoint = run_dir / "best_original_ade.pth"
                if not checkpoint.is_file():
                    print(f"[skip] missing {checkpoint}")
                    continue
                model = FrontEncoderASTRA(cfg, variant)
                decoder_module.replace_decoder(model, cfg, "kinematic_k20")
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                model = model.to(device)
                metrics = evaluate(model, loader, device, verifier, 0.30)
                output = {
                    "variant": variant,
                    "seed": seed,
                    "validation_scene_id": scene,
                    "checkpoint": str(checkpoint),
                    "checkpoint_selection": "minimum_validation_original_ade_at_5_epoch_interval",
                    "evaluation": metrics,
                }
                (run_dir / "best_metrics.json").write_text(
                    json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                fused = metrics["fused"]
                print(
                    variant, f"seed={seed}", f"scene={scene}",
                    f"ADE={metrics['original']['ade']:.4f}",
                    f"FDE={metrics['original']['fde']:.4f}",
                    f"FusedADE={fused['ade']:.4f}" if fused else "FusedADE=NA",
                    f"FusedFDE={fused['fde']:.4f}" if fused else "FusedFDE=NA",
                )


if __name__ == "__main__":
    main()
