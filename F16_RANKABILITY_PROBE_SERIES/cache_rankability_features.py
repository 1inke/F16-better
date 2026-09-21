"""
Frozen Candidate Feature Cache Generator for F16 Rankability Probe Series.

Extracts all frozen representations, coarse/final candidates, and quality metrics
from the strictly verified EXP-173 V2 (f16_temporal_social_cross) model on ETH Seed 1.

Enforces Butler Rule: research.semantic_objective_alignment
All coordinates are converted to relative displacements from the focal agent's last observed position t=0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [
    str(REPO),
    str(REPO / "FRONT_ENCODER_SERIES"),
    str(REPO / "SELECTOR_BENCHMARK_SERIES"),
    str(REPO / "E2_F16_END_TO_END_COMPARISON_SERIES"),
    str(HERE),
]

if sys.platform == "win32":
    torch_lib = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from build_candidate_cache import configure_astra_env, load_module
from QUALITY_AWARE_F16_SERIES.evaluate_quality_aware_f16 import load_f16_checkpoint
from common.feature_schema import validate_cache_dict


def cache_split_features(
    fold: str = "eth",
    seed: int = 1,
    split_name: str = "val",
    checkpoint_path: str = "",
    device_str: str = "cuda:0",
    output_dir: str = "",
) -> Path:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[CACHE] Caching split='{split_name}', fold='{fold}', seed={seed} on {device}...")

    if not checkpoint_path:
        # Default to strict C0 reproduction checkpoint
        c0_ckpt = (
            REPO
            / "QUALITY_AWARE_F16_SERIES_V2"
            / "results"
            / "p0_protocol"
            / "C0_EXP173_EXACT"
            / "f16_temporal_social_cross"
            / f"seed_{seed}"
            / "val_scene_-1"
            / "best_selected.pth"
        )
        if c0_ckpt.is_file():
            ckpt_path = c0_ckpt
        else:
            exp173_ckpt = (
                REPO
                / "E2_F16_END_TO_END_COMPARISON_SERIES"
                / "EXP_173_E2_F16_END_TO_END"
                / "results"
                / fold
                / "v2"
                / "e3"
                / "f16_temporal_social_cross"
                / f"seed_{seed}"
                / "val_scene_-1"
                / "best_selected.pth"
            )
            ckpt_path = exp173_ckpt
    else:
        ckpt_path = Path(checkpoint_path).resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")
    print(f"[CACHE] Using model checkpoint: {ckpt_path}")

    # Configure ASTRA environment
    astra_root = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
    cfg = configure_astra_env(REPO, astra_root, device)

    trainer_file = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    trainer_mod = load_module("train_front_variant", trainer_file)

    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer_mod.TrajectoryDataset(
        data_dir / f"{fold}_data_train.npy", data_dir / f"{fold}_num_train.npy", cfg
    )

    split_path = (
        REPO
        / "E2_F16_END_TO_END_COMPARISON_SERIES"
        / "EXP_173_E2_F16_END_TO_END"
        / "results"
        / fold
        / "v2"
        / "e3"
        / "f16_temporal_social_cross"
        / f"seed_{seed}"
        / "val_scene_-1"
        / "split_indices.npz"
    )
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split indices file: {split_path}")
    split_data = np.load(split_path)
    target_indices = split_data["train"] if split_name == "train" else split_data["validation"]
    print(f"[CACHE] Loaded {len(target_indices)} samples for split='{split_name}'")

    loader = DataLoader(
        Subset(dataset, target_indices.tolist()),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = load_f16_checkpoint(REPO, fold, seed, ckpt_path, cfg, device)
    model.eval()

    # Register forward hooks to extract context and mode features without modifying decoder
    captured = {}
    def hook_ctx(m, inp, out):
        captured["context"] = out.detach()

    def hook_mode(m, inp, out):
        captured["mode"] = out.detach()

    h_ctx = model.decoder.context.register_forward_hook(hook_ctx)
    h_mode = model.decoder.mode_mlp.register_forward_hook(hook_mode)

    # Storage arrays
    all_sample_ids = []
    all_past_relative = []
    all_future_relative = []
    all_candidate_final = []
    all_candidate_coarse = []
    all_source_anchors_relative = []
    all_corrected_anchors_relative = []
    all_context_features = []
    all_mode_features = []
    all_native_logits = []
    all_ade_final = []
    all_fde_final = []
    all_ade_coarse = []
    all_fde_coarse = []
    all_native_selected_idx = []
    all_oracle_final_idx = []
    all_oracle_coarse_idx = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            past, future, num_valid, *rest = batch
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()

            batch_size = past.shape[0]
            start_sample_idx = b_idx * 64
            batch_sample_ids = target_indices[start_sample_idx : start_sample_idx + batch_size]

            # Focal heading normalization (locked official standard)
            past, future = trainer_mod.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            past_loc = past - center

            # Official inference forward pass
            output = model(
                past_loc,
                None,
                None,
                mode="testing",
                num_valid=num_valid,
                graph_loc=past,
            )
            raw_pred = output[2] if isinstance(output, (tuple, list)) else output
            pred = raw_pred + center.unsqueeze(-3)

            logits = model.last_mode_logits
            coarse_trajectories = model.decoder.last_coarse_trajectories
            anchor_proposals = model.decoder.last_anchor_proposals
            corrected_anchors = model.decoder.last_corrected_anchors

            ctx = captured["context"]
            mode = captured["mode"]

            # Focal agent index = 0
            last_obs = past[:, 0, -1, :]  # [B, 2]
            last_pt_loc = past_loc[:, 0, -1, :]  # [B, 2]

            # Relative coordinates to focal last observation
            past_rel = past[:, 0] - last_obs[:, None, :]  # [B, 8, 2]
            fut_rel = future[:, 0] - last_obs[:, None, :]  # [B, 12, 2]

            cand_fin_rel = pred[:, 0] - last_obs[:, None, None, :]  # [B, 20, 12, 2]
            cand_coa_rel = coarse_trajectories[:, 0] - past_loc[:, 0, None, -1:, :]  # [B, 20, 12, 2]

            src_anc_rel = anchor_proposals[:, 0] - last_pt_loc[:, None, None, :]  # [B, 20, 3, 2]
            corr_anc_rel = corrected_anchors[:, 0] - last_pt_loc[:, None, None, :]  # [B, 20, 3, 2]

            focal_ctx = ctx[:, 0]  # [B, 64]
            focal_mode = mode[:, 0]  # [B, 20, 64]
            focal_logits = logits[:, 0]  # [B, 20]

            # Quality metrics
            err_fin = torch.linalg.vector_norm(cand_fin_rel - fut_rel.unsqueeze(1), dim=-1)
            ade_fin = err_fin.mean(dim=-1)
            fde_fin = err_fin[..., -1]

            err_coa = torch.linalg.vector_norm(cand_coa_rel - fut_rel.unsqueeze(1), dim=-1)
            ade_coa = err_coa.mean(dim=-1)
            fde_coarse = err_coa[..., -1]

            sel_idx = focal_logits.argmax(dim=-1)
            ora_fin_idx = ade_fin.argmin(dim=-1)
            ora_coa_idx = ade_coa.argmin(dim=-1)

            # Append to lists
            all_sample_ids.append(batch_sample_ids)
            all_past_relative.append(past_rel.cpu().numpy())
            all_future_relative.append(fut_rel.cpu().numpy())
            all_candidate_final.append(cand_fin_rel.cpu().numpy())
            all_candidate_coarse.append(cand_coa_rel.cpu().numpy())
            all_source_anchors_relative.append(src_anc_rel.cpu().numpy())
            all_corrected_anchors_relative.append(corr_anc_rel.cpu().numpy())
            all_context_features.append(focal_ctx.cpu().numpy())
            all_mode_features.append(focal_mode.cpu().numpy())
            all_native_logits.append(focal_logits.cpu().numpy())
            all_ade_final.append(ade_fin.cpu().numpy())
            all_fde_final.append(fde_fin.cpu().numpy())
            all_ade_coarse.append(ade_coa.cpu().numpy())
            all_fde_coarse.append(fde_coarse.cpu().numpy())
            all_native_selected_idx.append(sel_idx.cpu().numpy())
            all_oracle_final_idx.append(ora_fin_idx.cpu().numpy())
            all_oracle_coarse_idx.append(ora_coa_idx.cpu().numpy())

    h_ctx.remove()
    h_mode.remove()

    elapsed = time.perf_counter() - t0
    print(f"[CACHE] Feature extraction finished in {elapsed:.1f}s")

    cache_dict = {
        "sample_id": np.concatenate(all_sample_ids, axis=0),
        "past_relative": np.concatenate(all_past_relative, axis=0),
        "future_relative": np.concatenate(all_future_relative, axis=0),
        "candidate_final": np.concatenate(all_candidate_final, axis=0),
        "candidate_coarse": np.concatenate(all_candidate_coarse, axis=0),
        "source_anchors_relative": np.concatenate(all_source_anchors_relative, axis=0),
        "corrected_anchors_relative": np.concatenate(all_corrected_anchors_relative, axis=0),
        "context_feature": np.concatenate(all_context_features, axis=0),
        "mode_feature": np.concatenate(all_mode_features, axis=0),
        "native_logits": np.concatenate(all_native_logits, axis=0),
        "ADE_final": np.concatenate(all_ade_final, axis=0),
        "FDE_final": np.concatenate(all_fde_final, axis=0),
        "ADE_coarse": np.concatenate(all_ade_coarse, axis=0),
        "FDE_coarse": np.concatenate(all_fde_coarse, axis=0),
        "native_selected_index": np.concatenate(all_native_selected_idx, axis=0),
        "oracle_final_index": np.concatenate(all_oracle_final_idx, axis=0),
        "oracle_coarse_index": np.concatenate(all_oracle_coarse_idx, axis=0),
    }

    # Validate schema
    validate_cache_dict(cache_dict)
    print(f"[CACHE] Schema validation passed! Total samples: {len(cache_dict['sample_id'])}")

    out_path = Path(output_dir) if output_dir else HERE / "results" / "cache"
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{split_name}_seed{seed}.npz"
    np.savez_compressed(file_path, **cache_dict)
    print(f"[CACHE] Saved {file_path} (size: {file_path.stat().st_size / (1024*1024):.2f} MB)")

    return file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="eth")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    cache_split_features(
        fold=args.fold,
        seed=args.seed,
        split_name=args.split,
        checkpoint_path=args.checkpoint,
        device_str=args.device,
        output_dir=args.output_dir,
    )
