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
EXPERIMENTS = {
    "matks": ("EXP_117_MATKS", "f56_motion_adaptive_key_smoothing"),
    "rgva": ("EXP_118_RGVA_G4", "f57_relation_grouped_vector_attention"),
    "cdsa": ("EXP_119_CDSA", "f58_compute_neutral_differential"),
}

sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
os.chdir(PROJECT)
from experiment_config import load_locked_q20  # noqa: E402
from front_encoder_models import FrontEncoderASTRA  # noqa: E402
from train_front_variant import TrajectoryDataset, normalize_to_focal_heading  # noqa: E402


def load_model(label: str, scene: int):
    folder, variant = EXPERIMENTS[label]
    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.device = "cuda:0"
    cfg.device_list = [0]
    model = FrontEncoderASTRA(cfg, variant)
    spec = importlib.util.spec_from_file_location("experiment_decoder_models", DECODER)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    module.replace_decoder(model, cfg, "kinematic_k20")
    checkpoint = (
        ROOT / folder / "results" / variant / "seed_1"
        / f"val_scene_{scene}" / "best_selected.pth"
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.cuda().eval(), cfg


def loader_for(cfg, scene: int):
    data_path = PROJECT / "datasets/eth_ucy/processed_data_diverse" / f"{cfg.SUBSET}_data_train.npy"
    num_path = PROJECT / "datasets/eth_ucy/processed_data_diverse" / f"{cfg.SUBSET}_num_train.npy"
    raw = np.load(data_path, mmap_mode="r")
    scene_ids = np.asarray(raw[:, 0, 0, 0], dtype=np.int64)
    indices = np.flatnonzero(scene_ids == scene).tolist()
    return DataLoader(
        Subset(TrajectoryDataset(data_path, num_path, cfg), indices),
        batch_size=128, shuffle=False,
    )


def diagnose(label: str, scene: int) -> dict:
    model, cfg = load_model(label, scene)
    encoder = model.agent_aware_trans_encoder
    alpha_values = []
    negative_mass = []
    with torch.no_grad():
        for past, future, num_valid, *_ in loader_for(cfg, scene):
            past = past.cuda(non_blocking=True)
            future = future.cuda(non_blocking=True)
            num_valid = num_valid.cuda().int()
            past, _ = normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            if label == "matks":
                valid = torch.arange(past.shape[1], device=past.device)[None] < num_valid[:, None]
                alpha_values.append(encoder.last_key_smoothing[valid].cpu())
            elif label == "cdsa":
                negative_mass.append(float(encoder.last_differential_negative_mass))
    result = {"scene": scene}
    if label == "matks":
        values = torch.cat(alpha_values, dim=0)
        result.update({
            "valid_agent_histories": int(values.shape[0]),
            "mean_left_right": values.mean(dim=(0, 1)).tolist(),
            "mean_by_time_left_right": values.mean(dim=0).tolist(),
            "max": float(values.max()),
        })
    elif label == "rgva":
        result.update({
            "bounded_group_strengths": torch.tanh(
                encoder.group_correction_strength
            ).detach().cpu().tolist(),
            "relation_correction_weight_norm": float(
                encoder.relation_group_correction.weight.detach().norm().cpu()
            ),
        })
    else:
        result.update({
            "head_lambdas": torch.sigmoid(
                encoder.differential_lambda_logit
            ).clamp(max=0.20).detach().cpu().tolist(),
            "mean_negative_mass": float(np.mean(negative_mass)),
        })
    return result


if __name__ == "__main__":
    for experiment_label, (folder, _) in EXPERIMENTS.items():
        result = [diagnose(experiment_label, scene) for scene in (1, 3)]
        path = ROOT / folder / "mechanism_diagnostics.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(experiment_label, json.dumps(result))
