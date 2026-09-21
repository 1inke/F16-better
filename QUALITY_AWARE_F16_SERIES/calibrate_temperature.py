"""
Validation-only Temperature Calibration for Quality-Aware F16 Series.

Optimizes scalar temperature T_cal on the validation set only (Section 29).
Evaluates probability calibration before and after calibration on both val and test sets.
Note: T_cal does NOT change argmax(logits), preserving Top-1 ADE/FDE exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

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
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, Subset

from build_candidate_cache import configure_astra_env, load_module
from QUALITY_AWARE_F16_SERIES.common.probability_metrics import (
    compute_probability_metrics,
    compute_selection_metrics,
    compute_generation_metrics,
)
from QUALITY_AWARE_F16_SERIES.evaluate_quality_aware_f16 import load_f16_checkpoint


def extract_logits_and_ades(model, loader, device):
    trainer_file = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    trainer_mod = load_module("train_front_variant", trainer_file)

    all_cands = []
    all_ades = []
    all_fdes = []
    all_logits = []

    model.eval()
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

            focal_pred = pred[:, 0]
            focal_logits = logits[:, 0]
            focal_gt = future[:, 0]

            err = torch.linalg.vector_norm(focal_pred - focal_gt[:, None], dim=-1)
            ade = err.mean(dim=-1)
            fde = err[:, :, -1]

            all_cands.append(focal_pred.cpu().numpy())
            all_ades.append(ade.cpu().numpy())
            all_fdes.append(fde.cpu().numpy())
            all_logits.append(focal_logits.cpu().numpy())

    return (
        np.concatenate(all_cands, axis=0),
        np.concatenate(all_ades, axis=0),
        np.concatenate(all_fdes, axis=0),
        np.concatenate(all_logits, axis=0),
    )


def calibrate_and_evaluate(
    model_dir: Path,
    fold: str,
    seed: int,
    device: torch.device,
    tau_eval: float = 0.10,
) -> Dict[str, Any]:
    ckpt_path = model_dir / "best_selected.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    astra_root = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
    cfg = configure_astra_env(REPO, astra_root, device)

    trainer_file = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    trainer_mod = load_module("train_front_variant", trainer_file)

    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    base_dataset = trainer_mod.TrajectoryDataset(
        data_dir / f"{fold}_data_train.npy", data_dir / f"{fold}_num_train.npy", cfg
    )

    split_path = REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results" / fold / "v2" / "e3" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "split_indices.npz"
    split_data = np.load(split_path)
    val_indices = split_data["validation"]

    val_loader = DataLoader(
        Subset(base_dataset, val_indices.tolist()), batch_size=128,
        shuffle=False, num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )

    test_dataset = trainer_mod.TrajectoryDataset(
        data_dir / f"{fold}_data_test.npy", data_dir / f"{fold}_num_test.npy", cfg
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda", drop_last=False,
    )

    model = load_f16_checkpoint(REPO, fold, seed, ckpt_path, cfg, device)

    print("[CALIBRATE] Extracting validation representations...")
    v_cands, v_ades, v_fdes, v_logits = extract_logits_and_ades(model, val_loader, device)

    print("[CALIBRATE] Extracting test representations...")
    t_cands, t_ades, t_fdes, t_logits = extract_logits_and_ades(model, test_loader, device)

    # 1. Evaluate uncalibrated (T=1.0)
    v_probs_uncal = np.exp(v_logits - np.max(v_logits, axis=-1, keepdims=True))
    v_probs_uncal /= np.sum(v_probs_uncal, axis=-1, keepdims=True)
    v_uncal_metrics = compute_probability_metrics(v_probs_uncal, v_ades, tau_eval=tau_eval)

    t_probs_uncal = np.exp(t_logits - np.max(t_logits, axis=-1, keepdims=True))
    t_probs_uncal /= np.sum(t_probs_uncal, axis=-1, keepdims=True)
    t_uncal_metrics = compute_probability_metrics(t_probs_uncal, t_ades, tau_eval=tau_eval)

    # 2. Optimize T on VALIDATION ONLY to minimize soft CE
    k_star_val = np.argmin(v_ades, axis=-1)
    q_eval_val = np.exp(-v_ades / float(tau_eval))
    q_eval_val /= np.sum(q_eval_val, axis=-1, keepdims=True)

    def val_objective(T):
        if T <= 0.001:
            return 1e9
        scaled = v_logits / T
        log_sum_exp = np.log(np.sum(np.exp(scaled - np.max(scaled, axis=-1, keepdims=True)), axis=-1, keepdims=True)) + np.max(scaled, axis=-1, keepdims=True)
        log_p = scaled - log_sum_exp
        loss = -np.mean(np.sum(q_eval_val * log_p, axis=-1))
        return loss

    res = minimize_scalar(val_objective, bounds=(0.05, 10.0), method="bounded")
    best_t = float(res.x)
    print(f"[CALIBRATE] Optimal validation temperature T_cal = {best_t:.4f}")

    # 3. Evaluate calibrated with T_cal
    v_logits_cal = v_logits / best_t
    v_probs_cal = np.exp(v_logits_cal - np.max(v_logits_cal, axis=-1, keepdims=True))
    v_probs_cal /= np.sum(v_probs_cal, axis=-1, keepdims=True)
    v_cal_metrics = compute_probability_metrics(v_probs_cal, v_ades, tau_eval=tau_eval)

    t_logits_cal = t_logits / best_t
    t_probs_cal = np.exp(t_logits_cal - np.max(t_logits_cal, axis=-1, keepdims=True))
    t_probs_cal /= np.sum(t_probs_cal, axis=-1, keepdims=True)
    t_cal_metrics = compute_probability_metrics(t_probs_cal, t_ades, tau_eval=tau_eval)

    # Compute selection and generation (invariant under T_cal)
    sel_idx_v = np.argmax(v_probs_cal, axis=-1)
    sel_idx_t = np.argmax(t_probs_cal, axis=-1)
    v_sel = compute_selection_metrics(sel_idx_v, v_ades, v_fdes)
    t_sel = compute_selection_metrics(sel_idx_t, t_ades, t_fdes)

    v_gen = compute_generation_metrics(v_cands, v_ades, v_fdes)
    t_gen = compute_generation_metrics(t_cands, t_ades, t_fdes)

    calib_results = {
        "model_dir": str(model_dir),
        "fold": fold,
        "seed": seed,
        "best_temperature": best_t,
        "val": {
            "generation": v_gen,
            "selection": v_sel,
            "uncalibrated_probability": v_uncal_metrics,
            "calibrated_probability": v_cal_metrics,
        },
        "test": {
            "generation": t_gen,
            "selection": t_sel,
            "uncalibrated_probability": t_uncal_metrics,
            "calibrated_probability": t_cal_metrics,
        },
    }

    out_file = model_dir / "temperature_calibration_metrics.json"
    out_file.write_text(json.dumps(calib_results, indent=2), encoding="utf-8")
    print(f"[CALIBRATE] Saved calibration results to {out_file}")

    print(f"--- TEST CALIBRATION IMPACT ---")
    print(f"Uncalibrated ECE: {t_uncal_metrics['top1_ece']:.4f} -> Calibrated ECE: {t_cal_metrics['top1_ece']:.4f}")
    print(f"Uncalibrated Soft CE: {t_uncal_metrics['soft_ce']:.4f} -> Calibrated Soft CE: {t_cal_metrics['soft_ce']:.4f}")
    print(f"Uncalibrated Oracle NLL: {t_uncal_metrics['oracle_nll']:.4f} -> Calibrated Oracle NLL: {t_cal_metrics['oracle_nll']:.4f}")

    return calib_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Temperature for Quality-Aware F16")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing best_selected.pth")
    parser.add_argument("--fold", type=str, default="eth")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tau-eval", type=float, default=0.10)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    m_dir = (REPO / args.model_dir).resolve() if not Path(args.model_dir).is_absolute() else Path(args.model_dir).resolve()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    calibrate_and_evaluate(m_dir, args.fold, args.seed, dev, args.tau_eval)
