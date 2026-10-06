"""One-real-batch check that both IRA arms start from exact C00 inference."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASTRA = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
sys.path[:0] = [
    str(REPO / "SELECTOR_BENCHMARK_SERIES"),
    str(REPO / "FRONT_ENCODER_SERIES"),
]

from build_candidate_cache import configure_astra_env, load_module


C00_RUN = (
    REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
    / "results" / "eth" / "v2" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1"
)
C00_CACHE = (
    REPO / "DIVERSITY_STAGE_ATTRIBUTION_SERIES" / "results"
    / "stage_attribution_analysis" / "validation_candidate_cache" / "c00.npz"
)
STATS = REPO / "PHYSICAL_INTENT_FLOW_SERIES" / "calibration" / "intent_stats.json"


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = configure_astra_env(REPO, ASTRA, device)
    trainer = load_module(
        "ira_smoke_trainer", REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    )
    models = load_module(
        "ira_smoke_models", REPO / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
    )
    ira = load_module("ira_smoke_decoder", HERE / "intent_aux_models.py")
    os.environ["IRA_SOURCE_CHECKPOINT"] = str(C00_RUN / "best_selected.pth")
    os.environ["IRA_STATS_JSON"] = str(STATS)

    with np.load(C00_CACHE) as archive:
        cache = {key: archive[key] for key in archive.files}
    sample_ids = cache["sample_id"][:2].astype(np.int64)
    data = ASTRA / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer.TrajectoryDataset(
        data / "eth_data_train.npy", data / "eth_num_train.npy", cfg
    )
    past, future, num_valid, *_ = next(iter(DataLoader(
        Subset(dataset, sample_ids.tolist()), batch_size=2, shuffle=False
    )))
    past, future = past.to(device), future.to(device)
    num_valid = num_valid.to(device).int()
    past, future = trainer.normalize_to_focal_heading(past, future)
    center = past.mean(dim=-2, keepdim=True)

    errors = {}
    counts = {}
    arm_outputs = {}
    for name in ira.EXPERIMENTS:
        torch.manual_seed(20260923)
        model = models.FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
        ira.replace_decoder(model, cfg, name)
        model = model.to(device).eval()
        with torch.inference_mode():
            prediction = model(
                past - center, None, None, mode="testing",
                num_valid=num_valid, graph_loc=past,
            )[2] + center.unsqueeze(-3)
        candidate = prediction[:, 0].cpu().numpy()
        logits = model.last_mode_logits[:, 0].cpu().numpy()
        errors[name] = {
            "candidate_max_abs_error_vs_c00": float(
                np.max(np.abs(candidate - cache["candidate"][:2]))
            ),
            "logit_max_abs_error_vs_c00": float(
                np.max(np.abs(logits - cache["logits"][:2]))
            ),
        }
        counts[name] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        arm_outputs[name] = (candidate, logits)
        del model

    names = list(ira.EXPERIMENTS)
    cross_candidate = float(np.max(np.abs(arm_outputs[names[0]][0] - arm_outputs[names[1]][0])))
    cross_logit = float(np.max(np.abs(arm_outputs[names[0]][1] - arm_outputs[names[1]][1])))
    tolerance = 2.0e-5
    if any(
        item[metric] > tolerance
        for item in errors.values()
        for metric in item
    ):
        raise RuntimeError(f"IRA initialization does not reproduce C00: {errors}")
    if cross_candidate != 0.0 or cross_logit != 0.0:
        raise RuntimeError("IRA arms differ before training")
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"IRA arms have unequal trainable counts: {counts}")
    print(json.dumps({
        "device": str(device),
        "sample_ids": sample_ids.tolist(),
        "errors": errors,
        "cross_arm_candidate_max_abs_error": cross_candidate,
        "cross_arm_logit_max_abs_error": cross_logit,
        "trainable_parameter_counts": counts,
        "contract_passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()

