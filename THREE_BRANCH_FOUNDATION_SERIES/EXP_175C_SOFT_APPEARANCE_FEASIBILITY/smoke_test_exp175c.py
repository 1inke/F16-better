"""Real-SDD-batch preflight for EXP-175C."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175A_SCENE_UTILITY_AUDIT" / "run_exp175a_scene_utility_audit.py"
ADAPTER = HERE / "scene_feasibility_adapter.py"
S0 = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175S0_FINAL_JOINT_V2_SDD_BASELINE"
R2 = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175AR2_OBSERVATION_CONDITIONED_APPEARANCE"
VARIANT = "f16_temporal_social_cross"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scene-loss-weight", type=float, default=0.02)
    args = parser.parse_args()
    device = torch.device(args.device)
    base = load("exp175c_smoke_base", BASE)
    adapter = load("exp175c_smoke_adapter", ADAPTER)
    baseline, cfg, trainer = base.build_model(args.seed, device)
    model, _, _ = base.build_model(args.seed, device)
    source = S0 / "results" / "joint_v2" / VARIANT / f"seed_{args.seed}" / "val_scene_1" / "best_selected.pth"
    bank = R2 / "assets" / "appearance_bank.npz"
    os.environ["EXP175C_SOURCE_CHECKPOINT"] = str(source)
    os.environ["EXP175C_APPEARANCE_BANK"] = str(bank)
    os.environ["EXP175C_SCENE_LOSS_WEIGHT"] = str(args.scene_loss_weight)
    adapter.replace_decoder(model, cfg, adapter.EXPERIMENT)
    model = model.to(device)
    loader = base.make_loader("validation", cfg, trainer, args.batch_size, args.batch_size)
    absolute_past, absolute_future, num_valid, scene_ids, *_ = next(iter(loader))
    absolute_past, absolute_future = absolute_past.to(device), absolute_future.to(device)
    num_valid, scene_ids = num_valid.to(device).int(), scene_ids.to(device).long()
    past, future = trainer.normalize_to_focal_heading(absolute_past, absolute_future)
    center = past.mean(dim=-2, keepdim=True)
    baseline.eval(); model.eval(); model.decoder.set_scene_batch(scene_ids, absolute_past)
    with torch.inference_mode():
        baseline_prediction = baseline(past-center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)[2]
        scene_prediction = model(past-center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)[2]
    identity_error = float((baseline_prediction-scene_prediction).abs().max().cpu())
    if identity_error > 1e-5:
        raise RuntimeError(f"Zero-init identity failed: {identity_error}")
    model.decoder.apply_training_policy(model); model.decoder.set_scene_batch(scene_ids, absolute_past)
    prediction = model(past-center, future-center, None, mode="train", num_valid=num_valid, graph_loc=past)[2]
    logits = model.last_mode_logits
    auxiliary = model.decoder.auxiliary_training_loss(prediction[:, :1], (future-center)[:, :1], logits[:, :1])
    if not torch.isfinite(auxiliary) or not auxiliary.item() > 0:
        raise RuntimeError(f"Invalid auxiliary loss: {auxiliary}")
    auxiliary.backward()
    grads = [p.grad.norm() for p in model.decoder.scene_residual.parameters() if p.grad is not None]
    grad_norm = float(torch.stack(grads).norm().cpu()) if grads else 0.0
    if not grad_norm > 0:
        raise RuntimeError("Auxiliary loss produced no scene-branch gradient")
    payload = {
        "status": "preflight_passed", "actual_sdd_batch": len(absolute_past),
        "prediction_shape": list(prediction.shape), "initial_identity_max_abs_difference": identity_error,
        "weighted_auxiliary_loss": float(auxiliary.detach().cpu()),
        "raw_scene_violation_loss": float(model.decoder.last_scene_auxiliary_raw.cpu()),
        "scene_branch_gradient_norm": grad_norm,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "future_is_training_neighbourhood_supervision_not_model_input": True,
    }
    (HERE / "preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

