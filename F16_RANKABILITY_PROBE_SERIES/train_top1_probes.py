"""
Training Runner for Top-1 Aligned and Set-Aware Rankability Probes (R5, R6, R7).

Follows Guide F16_RANKABILITY_CORRECTION_AND_TOP1_PROBE_GUIDE.md:
- Probe R5: Oracle-vs-Rest pairwise loss on R3 feature [context, mode, corrected_anchors]
- Probe R6: Listwise Top-1 Hard CE on R3 feature
- Probe R7: Candidate-Set-Aware Ranker (1-layer Self-Attention, < 50k params) on R3 feature with Oracle-vs-Rest loss
- Seeds: runs across ranker seeds 1, 2, 3
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

from common.pairwise_objective import OracleVsRestPairwiseLoss, ListwiseTop1Loss
from ranker_models import DiagnosticProbeRanker, SetAwareDiagnosticProbeRanker
from tests.test_no_future_leakage import build_probe_features


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_top1_probe(
    probe_name: str,
    ranker_seed: int = 1,
    cache_seed: int = 1,
    device_str: str = "cuda:0",
    max_epochs: int = 50,
    patience: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
) -> dict:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    set_seed(ranker_seed)

    print(f"\n{'='*60}")
    print(f"[TRAIN-TOP1] Starting: {probe_name} (Ranker Seed {ranker_seed}) on {device}")
    print(f"{'='*60}")

    train_cache_file = HERE / "results" / "cache" / f"train_seed{cache_seed}.npz"
    val_cache_file = HERE / "results" / "cache" / f"val_seed{cache_seed}.npz"

    train_data = dict(np.load(train_cache_file))
    val_data = dict(np.load(val_cache_file))

    # All R5, R6, R7 use R3 feature [context, mode, corrected_anchors] (134 dims)
    x_train_raw = build_probe_features(train_data, "R3_context_post")
    x_val_raw = build_probe_features(val_data, "R3_context_post")
    assert x_train_raw.shape[-1] == 134

    # Strict Train-only normalization
    norm_mean = x_train_raw.mean(axis=(0, 1), keepdims=True)
    norm_std = x_train_raw.std(axis=(0, 1), keepdims=True)
    norm_std[norm_std < 1e-6] = 1.0

    x_train = ((x_train_raw - norm_mean) / norm_std).astype(np.float32)
    x_val = ((x_val_raw - norm_mean) / norm_std).astype(np.float32)

    y_train_ade = train_data["ADE_final"].astype(np.float32)
    y_val_ade = val_data["ADE_final"].astype(np.float32)
    val_ade_final = val_data["ADE_final"]

    # Model instantiation
    if probe_name == "R7_set_aware_ranker":
        model = SetAwareDiagnosticProbeRanker(input_dim=134, dropout=0.10).to(device)
    else:
        model = DiagnosticProbeRanker(input_dim=134, dropout=0.10).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN-TOP1] {probe_name} instantiated with {total_params:,} parameters")

    # Criterion instantiation
    if probe_name in ("R5_oracle_vs_rest", "R7_set_aware_ranker"):
        criterion = OracleVsRestPairwiseLoss(tie_threshold=0.02, max_diff_scale=0.20).to(device)
    elif probe_name == "R6_listwise_top1":
        criterion = ListwiseTop1Loss(tau=None).to(device)
    else:
        raise ValueError(f"Unknown probe: {probe_name}")

    # DataLoader with ranker_seed generator
    g = torch.Generator()
    g.manual_seed(ranker_seed)
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_ade))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=g
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    out_dir = HERE / "results" / "p2_rankability" / f"{probe_name}_s{ranker_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    native_sel_idx = val_data["native_selected_index"]
    native_val_ade = float(np.mean(val_ade_final[np.arange(len(val_ade_final)), native_sel_idx]))
    oracle_val_ade = float(np.mean(val_ade_final[np.arange(len(val_ade_final)), val_data["oracle_final_index"]]))

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
            print(f"[TRAIN-TOP1] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
            break

    elapsed = time.perf_counter() - t_start
    print(f"[TRAIN-TOP1] Completed in {elapsed:.1f}s. Best Epoch: {best_epoch}, Best Val ADE: {best_val_ade:.4f} m")

    torch.save(best_weights, out_dir / "best_model.pth")
    np.save(out_dir / "val_scores.npy", best_val_scores)
    with open(out_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    return {
        "probe_name": probe_name,
        "ranker_seed": ranker_seed,
        "total_params": total_params,
        "best_epoch": best_epoch,
        "best_val_ade": best_val_ade,
        "best_regret": best_val_ade - oracle_val_ade,
        "best_closure_pct": (native_val_ade - best_val_ade) / (native_val_ade - oracle_val_ade) * 100.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", nargs="+", default=["R5_oracle_vs_rest", "R6_listwise_top1", "R7_set_aware_ranker"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    results = []
    for probe in args.probes:
        for seed in args.seeds:
            res = train_top1_probe(probe_name=probe, ranker_seed=seed, device_str=args.device)
            results.append(res)

    print("\n" + "="*70)
    print("[TRAIN-TOP1] ALL TOP-1 ALIGNED PROBES FINISHED. SUMMARY:")
    for r in results:
        print(f"  {r['probe_name']} (s{r['ranker_seed']}): ADE={r['best_val_ade']:.4f}m | Regret={r['best_regret']:.4f}m | Closure={r['best_closure_pct']:+.2f}%")
    print("="*70)
