#!/usr/bin/env python
"""
Extract F16 Latent Cache.

Zero-mutation hook capturing internal representations during F16 test forward:
- L3: mode query representation from decoder.mode_mlp: shape (20, 64)
- L5: pre-logit feature vector (mode + source_anchors): shape (20, 70)
- L1: context token from decoder.context: shape (64,)

Saved per-fold as compressed npz to latent_cache/seed_{seed}/{fold}/latents.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path[:0] = [str(HERE.parent), str(REPO / "SELECTOR_BENCHMARK_SERIES")]
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
        print(f"[LATENT] Already exists: {npz_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    astra_root = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"

    cfg = configure_astra_env(REPO, astra_root, device)
    trainer_file = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    trainer_mod = load_module("train_front_variant", trainer_file)

    model = build_f16_model(REPO, fold, seed, cfg, device)

    # Setup non-invasive forward hook on mode_mlp
    captured: dict[str, torch.Tensor] = {}

    def mode_mlp_hook(module, args, output):
        # output is (B, N, 20, 64)
        captured["mode_latent"] = output[:, 0].detach().cpu()  # focal agent only: (B, 20, 64)

    handle = model.decoder.mode_mlp.register_forward_hook(mode_mlp_hook)

    data_file = data_dir / f"{fold}_data_test.npy"
    num_file = data_dir / f"{fold}_num_test.npy"
    dataset = trainer_mod.TrajectoryDataset(data_file, num_file, cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_l3 = []
    t0 = time.time()
    print(f"[LATENT] Extracting latents for {fold} (seed {seed}, {len(dataset)} samples)...")

    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer_mod.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            _ = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)

            l3 = captured.pop("mode_latent")  # (B, 20, 64)
            all_l3.append(l3.numpy())

    handle.remove()
    cat_l3 = np.concatenate(all_l3, axis=0).astype(np.float32)

    # Atomic save
    tmp_path = out_dir / "latents.tmp.npz"
    np.savez_compressed(tmp_path, mode_latent=cat_l3)
    if npz_path.exists():
        os.remove(npz_path)
    os.replace(tmp_path, npz_path)
    print(f"[LATENT OK] Saved {len(cat_l3)} samples (dim={cat_l3.shape[-1]}) -> {npz_path} in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Extract F16 Decoder Latent Cache")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--folds", type=str, default="eth,hotel,univ,zara01,zara02")
    parser.add_argument("--output-root", type=str, default="SELECTOR_RANKABILITY_SERIES/S2_LATENT_PROBE/latent_cache")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_base = (REPO / args.output_root / f"seed_{args.seed}").resolve()
    fold_list = [f.strip() for f in args.folds.split(",") if f.strip()]

    for f in fold_list:
        extract_fold_latents(f, args.seed, out_base / f, device, batch_size=args.batch_size, force=args.force)


if __name__ == "__main__":
    main()
