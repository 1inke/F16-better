"""
Training Runner for Rankability Probes (R1 - R4).

Protocol specifications (§17 & §18):
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Batch size: 256 samples
- Epochs: max 50, Early stopping patience: 8
- Metric for checkpoint: Validation Selected ADE
- Loss: DecisionRelevantPairwiseLoss
- Feature normalization: fit strictly on Train cache only!
- Architecture: DiagnosticProbeRanker
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
sys.path[:0] = [str(REPO), str(HERE)]

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from common.feature_schema import PROBE_FEATURE_DIMS
from common.pairwise_objective import DecisionRelevantPairwiseLoss
from ranker_models import DiagnosticProbeRanker
from tests.test_no_future_leakage import build_probe_features


def set_seed(seed: int = 1) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_probe(
    probe_name: str,
    seed: int = 1,
    fold: str = "eth",
    device_str: str = "cuda:0",
    max_epochs: int = 50,
    patience: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
) -> dict:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"[TRAIN-P2] Starting Probe Training: {probe_name} (Seed {seed}) on {device}")
    print(f"{'='*60}")

    train_cache_file = HERE / "results" / "cache" / f"train_seed{seed}.npz"
    val_cache_file = HERE / "results" / "cache" / f"val_seed{seed}.npz"

    if not train_cache_file.is_file() or not val_cache_file.is_file():
        raise FileNotFoundError(f"Missing cache files: {train_cache_file} or {val_cache_file}")

    train_data = dict(np.load(train_cache_file))
    val_data = dict(np.load(val_cache_file))

    # 1. Build features (strictly without ground truth / future)
    x_train_raw = build_probe_features(train_data, probe_name)
    x_val_raw = build_probe_features(val_data, probe_name)

    expected_dim = PROBE_FEATURE_DIMS[probe_name]
    assert x_train_raw.shape[-1] == expected_dim, (
        f"Feature dim mismatch for {probe_name}: expected {expected_dim}, got {x_train_raw.shape[-1]}"
    )

    # 2. Strict Train-only normalization
    norm_mean = x_train_raw.mean(axis=(0, 1), keepdims=True)
    norm_std = x_train_raw.std(axis=(0, 1), keepdims=True)
    norm_std[norm_std < 1e-6] = 1.0

    x_train = ((x_train_raw - norm_mean) / norm_std).astype(np.float32)
    x_val = ((x_val_raw - norm_mean) / norm_std).astype(np.float32)

    # ADE targets for loss calculation
    # For R1, does it rank coarse ADE or final ADE?
    # Guidance §11 & §14: All probes evaluate ability to select the final trajectory!
    # Primary supervision is ADE_final: ADE_k = ADE(\hat Y_k^{final}, Y)
    y_train_ade = train_data["ADE_final"].astype(np.float32)
    y_val_ade = val_data["ADE_final"].astype(np.float32)
    val_ade_final = val_data["ADE_final"]

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_ade))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    # Model and optimizer
    model = DiagnosticProbeRanker(input_dim=expected_dim, dropout=0.10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN-P2] {probe_name} Architecture instantiated with {total_params:,} parameters")

    criterion = DecisionRelevantPairwiseLoss(tie_threshold=0.02, max_diff_scale=0.20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    out_dir = HERE / "results" / "p2_rankability" / probe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Native baseline reference for validation
    native_sel_idx = val_data["native_selected_index"]
    native_val_ade = float(np.mean(val_ade_final[np.arange(len(val_ade_final)), native_sel_idx]))
    oracle_val_ade = float(np.mean(val_ade_final[np.arange(len(val_ade_final)), val_data["oracle_final_index"]]))
    print(f"[TRAIN-P2] Baseline Native Val ADE: {native_val_ade:.4f} m | Oracle: {oracle_val_ade:.4f} m")

    history = []
    best_val_ade = float("inf")
    best_epoch = -1
    best_weights = None
    best_val_scores = None
    patience_counter = 0

    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_ade_t = torch.from_numpy(y_val_ade).to(device)

    t_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        for b_x, b_y in train_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            scores = model(b_x)
            loss = criterion(scores, b_y)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            n_batches += 1

        train_loss = train_loss_accum / max(1, n_batches)

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_scores_t = model(x_val_t)
            val_loss = criterion(val_scores_t, y_val_ade_t).item()
            val_scores = val_scores_t.cpu().numpy()

        val_sel_idx = np.argmax(val_scores, axis=-1)
        val_sel_ade = float(np.mean(val_ade_final[np.arange(len(val_ade_final)), val_sel_idx]))
        val_regret = val_sel_ade - oracle_val_ade
        val_closure = (native_val_ade - val_sel_ade) / (native_val_ade - oracle_val_ade) * 100.0

        is_best = val_sel_ade < best_val_ade
        if is_best:
            best_val_ade = val_sel_ade
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_scores = val_scores.copy()
            patience_counter = 0
            mark = "★ BEST"
        else:
            patience_counter += 1
            mark = f"patience {patience_counter}/{patience}"

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_selected_ade": val_sel_ade,
            "val_regret": val_regret,
            "val_closure_pct": val_closure,
            "is_best": is_best,
        }
        history.append(epoch_record)

        if epoch % 5 == 0 or is_best or epoch == 1:
            print(
                f"  Epoch {epoch:02d}/{max_epochs:02d} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
                f"Val ADE: {val_sel_ade:.4f}m (Regret: {val_regret:.4f}m, Closure: {val_closure:+.1f}%) | {mark}"
            )

        if patience_counter >= patience:
            print(f"[TRAIN-P2] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
            break

    elapsed = time.perf_counter() - t_start
    print(f"[TRAIN-P2] Completed in {elapsed:.1f}s. Best Epoch: {best_epoch}, Best Val ADE: {best_val_ade:.4f} m")

    # Save artifacts
    torch.save(best_weights, out_dir / "best_model.pth")
    np.save(out_dir / "val_scores.npy", best_val_scores)
    np.savez_compressed(
        out_dir / "norm_stats.npz",
        mean=norm_mean,
        std=norm_std,
    )
    with open(out_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    return {
        "probe_name": probe_name,
        "input_dim": expected_dim,
        "total_params": total_params,
        "best_epoch": best_epoch,
        "best_val_ade": best_val_ade,
        "best_regret": best_val_ade - oracle_val_ade,
        "best_closure_pct": (native_val_ade - best_val_ade) / (native_val_ade - oracle_val_ade) * 100.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=["R1_exact_input", "R2_post_correction", "R3_context_post", "R4_full_final", "all"], default="all")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", default="eth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    probes = ["R1_exact_input", "R2_post_correction", "R3_context_post", "R4_full_final"] if args.probe == "all" else [args.probe]
    summary = []
    for p in probes:
        res = train_single_probe(
            probe_name=p,
            seed=args.seed,
            fold=args.fold,
            device_str=args.device,
            max_epochs=args.epochs,
            patience=args.patience,
        )
        summary.append(res)

    print("\n" + "="*60)
    print("[TRAIN-P2] ALL PROBES TRAINING COMPLETED. SUMMARY:")
    for s in summary:
        print(f"  {s['probe_name']:<20}: Val ADE = {s['best_val_ade']:.4f}m | Regret = {s['best_regret']:.4f}m | Closure = {s['best_closure_pct']:+.2f}%")
    print("="*60)
