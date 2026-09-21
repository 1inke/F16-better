"""
Evaluation script for Quality-Aware F16 Series.

Evaluates any checkpoint (Original, QA0, QA1, QA2) on Validation and Test sets.
Computes:
A. Generation Quality: minADE20, minFDE20, endpoint/trajectory diversity
B. Selection Quality: Selected ADE, Selected FDE, Regret, Exact Oracle Hit, NearOracle@5cm
C. Probability Quality: Exact Oracle NLL, Soft CE (tau=0.10), Brier Score, Top-1 ECE, Mean Max Prob, Mean Entropy
Outputs JSON and CSV metrics with single source of truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [
    str(REPO),
    str(REPO / "SELECTOR_BENCHMARK_SERIES"),
    str(REPO / "FRONT_ENCODER_SERIES"),
    str(REPO / "E2_F16_END_TO_END_COMPARISON_SERIES"),
]

if sys.platform == "win32":
    torch_lib = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import torch
from torch.utils.data import DataLoader, Subset

from build_candidate_cache import configure_astra_env, load_module
from QUALITY_AWARE_F16_SERIES.common.probability_metrics import (
    compute_generation_metrics,
    compute_selection_metrics,
    compute_probability_metrics,
)


def load_f16_checkpoint(
    repo_root: Path,
    fold: str,
    seed: int,
    checkpoint_path: Path,
    cfg,
    device: torch.device,
) -> torch.nn.Module:
    exp173 = repo_root / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results"
    h3_ckpt = exp173 / fold / "h3" / "e3" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"

    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3_ckpt)
    os.environ["EXP173_DECODER_INIT_SEED"] = str(seed)

    model_file = repo_root / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
    adapter_file = repo_root / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
    models_mod = load_module("front_encoder_models", model_file)
    adapter_mod = load_module("joint_v2_adapter", adapter_file)

    model = models_mod.FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    adapter_mod.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature_scale: float = 1.0,
    tau_eval: float = 0.10,
) -> Dict[str, Any]:
    trainer_file = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    trainer_mod = load_module("train_front_variant", trainer_file)

    all_cands = []
    all_ades = []
    all_fdes = []
    all_logits = []
    all_probs = []
    all_selected_idx = []

    with torch.no_grad():
        for batch in loader:
            past, future, num_valid, *_ = batch
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()

            past, future = trainer_mod.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)

            output = model(
                past - center, future - center, None, mode="val",
                num_valid=num_valid, graph_loc=past,
            )
            raw_pred = output[2] if isinstance(output, (tuple, list)) else output
            pred = raw_pred + center.unsqueeze(-3)
            logits = model.last_mode_logits

            focal_pred = pred[:, 0]     # [B, 20, 12, 2]
            focal_logits = logits[:, 0] # [B, 20]
            focal_gt = future[:, 0]     # [B, 12, 2]

            err = torch.linalg.vector_norm(focal_pred - focal_gt[:, None], dim=-1)
            ade = err.mean(dim=-1) # [B, 20]
            fde = err[:, :, -1]    # [B, 20]

            scaled_logits = focal_logits / float(temperature_scale)
            probs = torch.softmax(scaled_logits, dim=-1)
            selected_idx = probs.argmax(dim=-1)

            all_cands.append(focal_pred.cpu().numpy())
            all_ades.append(ade.cpu().numpy())
            all_fdes.append(fde.cpu().numpy())
            all_logits.append(focal_logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_selected_idx.append(selected_idx.cpu().numpy())

    cands = np.concatenate(all_cands, axis=0)
    ades = np.concatenate(all_ades, axis=0)
    fdes = np.concatenate(all_fdes, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    selected_idx = np.concatenate(all_selected_idx, axis=0)

    gen_m = compute_generation_metrics(cands, ades, fdes)
    sel_m = compute_selection_metrics(selected_idx, ades, fdes)
    prob_m = compute_probability_metrics(probs, ades, tau_eval=tau_eval)

    return {
        "generation": gen_m,
        "selection": sel_m,
        "probability": prob_m,
        "raw": {
            "samples": len(selected_idx),
            "mean_logit": float(np.mean(logits)),
            "std_logit": float(np.std(logits)),
            "min_logit": float(np.min(logits)),
            "max_logit": float(np.max(logits)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quality-Aware F16 Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--fold", type=str, default="eth")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temperature-scale", type=float, default=1.0)
    parser.add_argument("--tau-eval", type=float, default=0.10)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = (REPO / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_p = None
    if args.output_json:
        out_p = (REPO / args.output_json).resolve() if not Path(args.output_json).is_absolute() else Path(args.output_json).resolve()

    astra_root = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
    cfg = configure_astra_env(REPO, astra_root, device)

    trainer_file = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    trainer_mod = load_module("train_front_variant", trainer_file)

    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    base_dataset = trainer_mod.TrajectoryDataset(
        data_dir / f"{args.fold}_data_train.npy", data_dir / f"{args.fold}_num_train.npy", cfg
    )

    split_path = REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results" / args.fold / "v2" / "e3" / "f16_temporal_social_cross" / f"seed_{args.seed}" / "val_scene_-1" / "split_indices.npz"
    split_data = np.load(split_path)
    val_indices = split_data["validation"]

    val_loader = DataLoader(
        Subset(base_dataset, val_indices.tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )

    test_dataset = trainer_mod.TrajectoryDataset(
        data_dir / f"{args.fold}_data_test.npy", data_dir / f"{args.fold}_num_test.npy", cfg
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda", drop_last=False,
    )

    model = load_f16_checkpoint(REPO, args.fold, args.seed, ckpt_path, cfg, device)

    print(f"[EVAL] Evaluating checkpoint: {ckpt_path.name} on fold={args.fold}, seed={args.seed}...")
    val_res = evaluate_dataset(model, val_loader, device, args.temperature_scale, args.tau_eval)
    test_res = evaluate_dataset(model, test_loader, device, args.temperature_scale, args.tau_eval)

    result = {
        "checkpoint": str(ckpt_path),
        "fold": args.fold,
        "seed": args.seed,
        "temperature_scale": args.temperature_scale,
        "val": val_res,
        "test": test_res,
    }

    if out_p is not None:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[EVAL] Saved results to {out_p}")

    print(f"--- VALIDATION (N={val_res['raw']['samples']}) ---")
    print(f"Selected ADE: {val_res['selection']['selected_ade']:.4f}m | FDE: {val_res['selection']['selected_fde']:.4f}m | Regret: {val_res['selection']['selection_regret']:.4f}m")
    print(f"minADE20:     {val_res['generation']['min_ade20']:.4f}m | minFDE20: {val_res['generation']['min_fde20']:.4f}m")
    print(f"Oracle NLL:   {val_res['probability']['oracle_nll']:.4f} | Soft CE: {val_res['probability']['soft_ce']:.4f} | ECE: {val_res['probability']['top1_ece']:.4f} | Entropy: {val_res['probability']['mean_entropy']:.4f}")

    print(f"--- TEST (N={test_res['raw']['samples']}) ---")
    print(f"Selected ADE: {test_res['selection']['selected_ade']:.4f}m | FDE: {test_res['selection']['selected_fde']:.4f}m | Regret: {test_res['selection']['selection_regret']:.4f}m")
    print(f"minADE20:     {test_res['generation']['min_ade20']:.4f}m | minFDE20: {test_res['generation']['min_fde20']:.4f}m")
    print(f"Oracle NLL:   {test_res['probability']['oracle_nll']:.4f} | Soft CE: {test_res['probability']['soft_ce']:.4f} | ECE: {test_res['probability']['top1_ece']:.4f} | Entropy: {test_res['probability']['mean_entropy']:.4f}")


if __name__ == "__main__":
    main()
