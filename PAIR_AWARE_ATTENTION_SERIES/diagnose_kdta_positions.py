from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"

sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
os.chdir(PROJECT)
from experiment_config import load_locked_q20  # noqa: E402
from front_encoder_models import FrontEncoderASTRA  # noqa: E402
from train_front_variant import TrajectoryDataset, normalize_to_focal_heading  # noqa: E402


def diagnose(scene: int) -> dict:
    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.device = "cuda:0"
    cfg.device_list = [0]
    model = FrontEncoderASTRA(cfg, "f52_pair_deformable_temporal")
    spec = importlib.util.spec_from_file_location("experiment_decoder_models", DECODER)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    module.replace_decoder(model, cfg, "kinematic_k20")
    checkpoint = ROOT / "EXP_113_KDTA_BASE" / "results" / "f52_pair_deformable_temporal" / "seed_1" / f"val_scene_{scene}" / "best_selected.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = model.cuda().eval()

    data_path = PROJECT / "datasets/eth_ucy/processed_data_diverse" / f"{cfg.SUBSET}_data_train.npy"
    num_path = PROJECT / "datasets/eth_ucy/processed_data_diverse" / f"{cfg.SUBSET}_num_train.npy"
    raw = np.load(data_path, mmap_mode="r")
    scene_ids = np.asarray(raw[:, 0, 0, 0], dtype=np.int64)
    indices = np.flatnonzero(scene_ids == scene).tolist()
    loader = DataLoader(Subset(TrajectoryDataset(data_path, num_path, cfg), indices), batch_size=128, shuffle=False)

    collected = []
    with torch.no_grad():
        for past, future, num_valid, *_ in loader:
            past = past.cuda(non_blocking=True)
            future = future.cuda(non_blocking=True)
            num_valid = num_valid.cuda().int()
            past, _ = normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            encoder = model.agent_aware_trans_encoder
            positions = encoder.last_kdta_positions
            allowed = encoder.last_allowed
            collected.append(positions[allowed].cpu())
    values = torch.cat(collected, dim=0)
    learned = values[:, :2]
    quantiles = torch.quantile(learned, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]), dim=0)
    return {
        "scene": scene,
        "allowed_pairs": int(values.shape[0]),
        "mean": learned.mean(dim=0).tolist(),
        "std": learned.std(dim=0).tolist(),
        "min": learned.min(dim=0).values.tolist(),
        "max": learned.max(dim=0).values.tolist(),
        "quantiles_05_25_50_75_95": quantiles.tolist(),
    }


if __name__ == "__main__":
    result = [diagnose(1), diagnose(3)]
    path = ROOT / "EXP_113_KDTA_BASE" / "position_diagnostics.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
