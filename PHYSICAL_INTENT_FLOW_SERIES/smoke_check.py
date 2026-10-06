"""One-real-batch forward/backward contract check for PIF1."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASTRA = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
sys.path[:0] = [str(REPO / "SELECTOR_BENCHMARK_SERIES"), str(REPO / "FRONT_ENCODER_SERIES")]

from build_candidate_cache import configure_astra_env, load_module


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = configure_astra_env(REPO, ASTRA, device)
    trainer = load_module("pif_smoke_trainer", REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py")
    models = load_module("pif_smoke_models", REPO / "FRONT_ENCODER_SERIES" / "front_encoder_models.py")
    pif = load_module("pif_smoke_decoder", HERE / "physical_intent_flow_models.py")
    os.environ["PIF_SOURCE_CHECKPOINT"] = str(
        REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
        / "results" / "eth" / "v2" / "e3" / "f16_temporal_social_cross"
        / "seed_1" / "val_scene_-1" / "best_selected.pth"
    )
    os.environ["PIF_STATS_JSON"] = str(HERE / "calibration" / "intent_stats.json")
    model = models.FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    pif.replace_decoder(model, cfg, pif.EXPERIMENT)
    model = model.to(device)
    data = ASTRA / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer.TrajectoryDataset(data / "eth_data_train.npy", data / "eth_num_train.npy", cfg)
    ids = torch.from_numpy(__import__("numpy").load(
        REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
        / "results" / "eth" / "v2" / "e3" / "f16_temporal_social_cross"
        / "seed_1" / "val_scene_-1" / "split_indices.npz"
    )["train"][:2]).tolist()
    past, future, num_valid, *_ = next(iter(DataLoader(Subset(dataset, ids), batch_size=2)))
    past, future, num_valid = past.to(device), future.to(device), num_valid.to(device).int()
    past, future = trainer.normalize_to_focal_heading(past, future)
    center = past.mean(dim=-2, keepdim=True)
    model.decoder.set_training_target(future - center)
    prediction = model(past - center, future - center, None, mode="train", num_valid=num_valid, graph_loc=past)[2]
    logits = model.last_mode_logits
    loss_module = load_module("pif_smoke_loss", REPO / "FRONT_ENCODER_SERIES" / "locked_mode_query_loss.py").ModeQueryLoss(cfg).to(device)
    loss, _ = loss_module(prediction[:, :1], (future - center)[:, :1], logits[:, :1], cfg)
    loss = loss + model.decoder.auxiliary_training_loss(prediction[:, :1], (future - center)[:, :1], logits[:, :1])
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite smoke loss")
    loss.backward()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or not all(name.startswith("decoder.flow.") for name in trainable):
        raise RuntimeError(f"Unexpected trainable parameters: {trainable}")
    grad_norm = sum(
        float(parameter.grad.norm().detach().cpu())
        for name, parameter in model.named_parameters()
        if name in trainable and parameter.grad is not None
    )
    if not grad_norm > 0:
        raise RuntimeError("Flow received no gradient")
    payload = {
        "device": str(device), "loss": float(loss.detach().cpu()),
        "prediction_shape": list(prediction[:, :1].shape),
        "logit_shape": list(logits[:, :1].shape),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "trainable_scope_exact": True, "gradient_norm_sum": grad_norm,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

