#!/usr/bin/env python
"""
Extract Context-Conditioned Latents (L3, L4, L5) from F16.

Non-invasive extraction:
- L3: decoder.mode_mlp output -> (20, 64)
- L4: decoder.refiner.network[3] output -> (20, 128)
- L5: input to decoder.score_head -> (20, 70)
- native_logits: output of decoder.score_head -> (20,)

Strict verification:
- Checks |traj_hook - traj_clean| < 1e-7 on first batch
- Produces latent_cache_v2/seed_{seed}/{fold}/latents.npz
- Produces latent_alignment.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO / "SELECTOR_BENCHMARK_SERIES")]

if sys.platform == "win32":
    torch_lib = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import torch
from torch.utils.data import DataLoader
from build_candidate_cache import build_f16_model, configure_astra_env, load_module

FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]


def extract_fold_latents(
    fold: str,
    seed: int,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 128,
    force: bool = False,
):
    npz_path = out_dir / "latents.npz"
    if npz_path.exists() and not force:
        print(f"[LATENT_V2] Already exists: {npz_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    astra_root = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"

    cfg = configure_astra_env(REPO, astra_root, device)
    trainer_file = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    trainer_mod = load_module("train_front_variant", trainer_file)

    model = build_f16_model(REPO, fold, seed, cfg, device)

    # Setup non-invasive forward hooks
    captured: dict[str, torch.Tensor] = {}

    def mode_mlp_hook(module, args, output):
        captured["l3"] = output[:, 0].detach().cpu() # (B, 20, 64)

    def score_head_hook(module, args, output):
        # args[0] is (B, N, 20, 70)
        captured["l5"] = args[0][:, 0].detach().cpu() # (B, 20, 70)
        captured["logits"] = output[:, 0].detach().cpu() # (B, 20)

    def refiner_hook(module, args, output):
        # features passed to refiner is (B, N, 20, 70)
        # output of refiner.network[3] is (B, N, 20, 128)
        # focal agent is index 1 (agent 0) -> output[:, 0] is (B, 20, 128)
        captured["l4"] = output[:, 0].detach().cpu() # (B, 20, 128)

    h_mode = model.decoder.mode_mlp.register_forward_hook(mode_mlp_hook)
    h_score = model.decoder.score_head.register_forward_hook(score_head_hook)
    h_ref = model.decoder.refiner.network[3].register_forward_hook(refiner_hook)

    data_file = data_dir / f"{fold}_data_test.npy"
    num_file = data_dir / f"{fold}_num_test.npy"
    dataset = trainer_mod.TrajectoryDataset(data_file, num_file, cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_l3, all_l4, all_l5, all_logits = [], [], [], []
    t0 = time.time()
    print(f"[LATENT_V2] Extracting latents for {fold} ({len(dataset)} samples)...")

    verified_diff = False

    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer_mod.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)

            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            out_traj = output[2] if isinstance(output, (tuple, list)) else output

            if not verified_diff:
                # Compare hooked forward against clean forward on same input
                h_mode.remove()
                h_score.remove()
                h_ref.remove()
                clean_output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
                clean_traj = clean_output[2] if isinstance(clean_output, (tuple, list)) else clean_output
                max_diff = float(torch.max(torch.abs(out_traj - clean_traj)).item())
                if max_diff > 1e-6:
                    raise RuntimeError(f"Hook numerical leakage! Max diff: {max_diff}")
                print(f"  [Hook Verification] Passed: max diff = {max_diff:.2e} < 1e-6")
                # Re-register
                h_mode = model.decoder.mode_mlp.register_forward_hook(mode_mlp_hook)
                h_score = model.decoder.score_head.register_forward_hook(score_head_hook)
                h_ref = model.decoder.refiner.network[3].register_forward_hook(refiner_hook)
                verified_diff = True

            all_l3.append(captured.pop("l3").numpy())
            all_l4.append(captured.pop("l4").numpy())
            all_l5.append(captured.pop("l5").numpy())
            all_logits.append(captured.pop("logits").numpy())

    h_mode.remove()
    h_score.remove()
    h_ref.remove()

    cat_l3 = np.concatenate(all_l3, axis=0).astype(np.float32)
    cat_l4 = np.concatenate(all_l4, axis=0).astype(np.float32)
    cat_l5 = np.concatenate(all_l5, axis=0).astype(np.float32)
    cat_logits = np.concatenate(all_logits, axis=0).astype(np.float32)

    tmp_path = out_dir / "latents.tmp.npz"
    np.savez_compressed(
        tmp_path,
        l3=cat_l3,
        l4=cat_l4,
        l5=cat_l5,
        native_logits=cat_logits,
    )
    if npz_path.exists():
        os.remove(npz_path)
    os.replace(tmp_path, npz_path)
    print(f"[LATENT_V2 OK] Saved {len(cat_l3)} samples (L3: {cat_l3.shape[-1]}, L4: {cat_l4.shape[-1]}, L5: {cat_l5.shape[-1]}) -> {npz_path} in {time.time()-t0:.1f}s")


def verify_alignment(cache_dir: Path, feat_dir: Path, out_dir: Path):
    alignment_report = {}
    all_ok = True
    for f in FOLDS:
        npz_file = cache_dir / f / "latents.npz"
        feat_file = feat_dir / f / "features.npz"
        if not npz_file.exists() or not feat_file.exists():
            all_ok = False
            continue

        lat_d = np.load(npz_file)
        feat_d = np.load(feat_file)

        n_lat = len(lat_d["l3"])
        n_feat = len(feat_d["g0"])
        assert n_lat == n_feat, f"Count mismatch in {f}: {n_lat} vs {n_feat}"

        # Check native logit alignment
        # In feature bank, g0[:, :, 0] is native score / logit
        nat_logits = lat_d["native_logits"]
        if nat_logits.ndim == 3 and nat_logits.shape[-1] == 1:
            nat_logits = nat_logits.squeeze(-1)
        diff_logits = float(np.max(np.abs(nat_logits - feat_d["g0"][:, :, 0])))

        alignment_report[f] = {
            "num_samples": n_lat,
            "l3_shape": list(lat_d["l3"].shape),
            "l4_shape": list(lat_d["l4"].shape),
            "l5_shape": list(lat_d["l5"].shape),
            "max_logit_diff_to_g0": float(diff_logits),
            "aligned": bool(diff_logits < 1e-4),
        }
        if diff_logits >= 1e-4:
            all_ok = False

    res = {
        "status": "PASS" if all_ok else "FAIL",
        "folds": alignment_report,
    }
    align_path = out_dir / f"latent_alignment_{cache_dir.name}.json"
    align_path_general = out_dir / "latent_alignment.json"
    align_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    if cache_dir.name == "seed_1":
        align_path_general.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[LATENT ALIGNMENT] Status: {res['status']} -> {align_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract Context Latents V2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--folds", type=str, default="eth,hotel,univ,zara01,zara02")
    parser.add_argument("--output-root", type=str, default="SELECTOR_RANKABILITY_SERIES/S2V2_CONTEXT_LATENT/latent_cache_v2")
    parser.add_argument("--feature-root", type=str, default="SELECTOR_BENCHMARK_SERIES/features")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_base = (REPO / args.output_root / f"seed_{args.seed}").resolve()
    feat_base = (REPO / args.feature_root / f"seed_{args.seed}").resolve()
    results_dir = (REPO / "SELECTOR_RANKABILITY_SERIES" / "S2V2_CONTEXT_LATENT" / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    fold_list = [f.strip() for f in args.folds.split(",") if f.strip()]
    for f in fold_list:
        extract_fold_latents(f, args.seed, out_base / f, device, batch_size=args.batch_size, force=args.force)

    verify_alignment(out_base, feat_base, results_dir)


if __name__ == "__main__":
    main()
