"""Actual-data forward/backward preflight for EXP-175B-R2."""
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
ADAPTER = HERE / "scene_appearance_anchor_adapter.py"
S0 = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175S0_FINAL_JOINT_V2_SDD_BASELINE"
R2 = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175AR2_OBSERVATION_CONDITIONED_APPEARANCE"
VARIANT = "f16_temporal_social_cross"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    base = load("exp175br2_smoke_base", BASE)
    adapter = load("exp175br2_smoke_adapter", ADAPTER)
    baseline, cfg, trainer = base.build_model(args.seed, device)
    scene_model, _, _ = base.build_model(args.seed, device)
    source = S0 / "results" / "joint_v2" / VARIANT / f"seed_{args.seed}" / "val_scene_1" / "best_selected.pth"
    bank = R2 / "assets" / "appearance_bank.npz"
    os.environ["EXP175BR2_SOURCE_CHECKPOINT"] = str(source)
    os.environ["EXP175BR2_APPEARANCE_BANK"] = str(bank)
    adapter.replace_decoder(scene_model, cfg, adapter.EXPERIMENT)
    scene_model = scene_model.to(device)

    loader = base.make_loader("validation", cfg, trainer, args.batch_size, args.batch_size)
    absolute_past, absolute_future, num_valid, scene_ids, *_ = next(iter(loader))
    absolute_past = absolute_past.to(device)
    absolute_future = absolute_future.to(device)
    num_valid = num_valid.to(device).int()
    scene_ids = scene_ids.to(device).long()
    past, future = trainer.normalize_to_focal_heading(absolute_past, absolute_future)
    center = past.mean(dim=-2, keepdim=True)

    baseline.eval(); scene_model.eval()
    scene_model.decoder.set_scene_batch(scene_ids, absolute_past)
    with torch.inference_mode():
        base_prediction = baseline(
            past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past
        )[2]
        scene_prediction = scene_model(
            past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past
        )[2]
    initial_max_difference = float((base_prediction - scene_prediction).abs().max().cpu())
    # Separately instantiated CUDA models can differ by a few FP32 ULPs even
    # with identical weights; the scene correction itself is exactly zero.
    if initial_max_difference > 1e-5:
        raise RuntimeError(f"Zero-init identity failed: max diff={initial_max_difference}")

    scene_model.decoder.apply_training_policy(scene_model)
    scene_model.decoder.set_scene_batch(scene_ids, absolute_past)
    prediction = scene_model(
        past - center, future - center, None, mode="train", num_valid=num_valid, graph_loc=past
    )[2]
    target = (future - center)[:, :1, None]
    distance = torch.linalg.vector_norm(prediction[:, :1] - target, dim=-1)
    quality = distance.mean(dim=-1) + 0.25 * distance[..., -1]
    weights = torch.softmax(-quality.detach() / 0.10, dim=-1)
    loss = (weights * quality).sum(dim=-1).mean()
    loss.backward()
    gradients = [
        p.grad.detach().norm() for p in scene_model.decoder.scene_residual.parameters()
        if p.grad is not None
    ]
    gradient_norm = float(torch.stack(gradients).norm().cpu()) if gradients else 0.0
    if not gradient_norm > 0.0:
        raise RuntimeError("Scene branch received no gradient")
    trainable = sum(p.numel() for p in scene_model.parameters() if p.requires_grad)
    non_scene_trainable = [
        name for name, p in scene_model.named_parameters()
        if p.requires_grad and not name.startswith("decoder.scene_residual.")
    ]
    if non_scene_trainable:
        raise RuntimeError(f"Unexpected trainable base parameters: {non_scene_trainable[:8]}")
    payload = {
        "status": "preflight_passed",
        "actual_sdd_batch": int(len(absolute_past)),
        "prediction_shape": list(prediction.shape),
        "initial_identity_max_abs_difference": initial_max_difference,
        "scene_trainable_parameters": trainable,
        "gradient_norm": gradient_norm,
        "scene_correction_caps": [0.05, 0.08, 0.15],
        "no_future_used_for_scene_query": True,
    }
    (HERE / "preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
