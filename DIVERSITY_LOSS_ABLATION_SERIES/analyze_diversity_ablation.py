"""Analyze EXP-173/F16 diversity ablation checkpoints on validation only.

Outputs:
  - condition_summary.csv
  - bucket_summary.csv
  - per_sample_<condition>_seed<seed>.csv
  - paired_vs_baseline.csv
  - DIVERSITY_ABLATION_REPORT.md

No test split is touched. Hyperparameter decisions should be made on validation only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

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

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from build_candidate_cache import configure_astra_env, load_module
from QUALITY_AWARE_F16_SERIES.evaluate_quality_aware_f16 import load_f16_checkpoint


def mean_pairwise_endpoint_spread(endpoints: np.ndarray) -> np.ndarray:
    # endpoints: [N,K,2]
    diff = endpoints[:, :, None, :] - endpoints[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    k = endpoints.shape[1]
    mask = np.triu(np.ones((k, k), dtype=bool), k=1)
    return dist[:, mask].mean(axis=1)


def score_rank_desc(logits: np.ndarray, index: np.ndarray) -> np.ndarray:
    order = np.argsort(-logits, axis=1)
    inv = np.empty_like(order)
    rows = np.arange(len(order))[:, None]
    inv[rows, order] = np.arange(order.shape[1])[None, :]
    return inv[np.arange(len(index)), index] + 1


def bootstrap_mean_ci(delta: np.ndarray, n: int = 1000, seed: int = 20260922):
    rng = np.random.default_rng(seed)
    m = len(delta)
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        means[i] = delta[idx].mean()
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def classify_bucket(regret: np.ndarray, minade: np.ndarray) -> np.ndarray:
    out = np.full(len(regret), "R0_mild", dtype=object)
    out[(regret >= 0.20) & (regret < 0.50)] = "R1_medium"
    r2 = regret >= 0.50
    out[r2 & (minade < 0.50)] = "A_selection_dominant"
    out[r2 & (minade >= 0.50) & (minade < 1.00)] = "B_mixed"
    out[r2 & (minade >= 1.00)] = "C_generation_dominant"
    return out


def evaluate_condition(run_dir: Path, astra_root: Path, device: torch.device, batch_size: int):
    spec = json.loads((run_dir / "diversity_ablation_spec.json").read_text(encoding="utf-8"))
    seed = int(spec["seed"])
    checkpoint = run_dir / "best_selected.pth"
    split_path = run_dir / "split_indices.npz"
    if not checkpoint.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint/split in {run_dir}")

    cfg = configure_astra_env(REPO, astra_root, device)
    trainer = load_module("train_front_variant_for_diversity_eval", REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py")
    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer.TrajectoryDataset(data_dir / "eth_data_train.npy", data_dir / "eth_num_train.npy", cfg)
    val_idx = np.load(split_path)["validation"]
    loader = DataLoader(
        Subset(dataset, val_idx.tolist()), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )

    model = load_f16_checkpoint(REPO, "eth", seed, checkpoint, cfg, device)
    model.eval()

    sample_ids, ades_all, fdes_all, logits_all, endpoints_all = [], [], [], [], []
    with torch.no_grad():
        offset = 0
        for batch in loader:
            past, future, num_valid, *_ = batch
            b = past.shape[0]
            batch_ids = val_idx[offset:offset+b]
            offset += b
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(
                past - center, None, None, mode="testing",
                num_valid=num_valid, graph_loc=past,
            )
            raw_pred = output[2] if isinstance(output, (tuple, list)) else output
            pred = raw_pred + center.unsqueeze(-3)
            logits = model.last_mode_logits[:, 0]
            focal = pred[:, 0]
            gt = future[:, 0]
            err = torch.linalg.vector_norm(focal - gt[:, None], dim=-1)
            ade = err.mean(dim=-1)
            fde = err[..., -1]
            sample_ids.append(np.asarray(batch_ids))
            ades_all.append(ade.cpu().numpy())
            fdes_all.append(fde.cpu().numpy())
            logits_all.append(logits.cpu().numpy())
            endpoints_all.append(focal[:, :, -1, :].cpu().numpy())

    sample_id = np.concatenate(sample_ids)
    ades = np.concatenate(ades_all)
    fdes = np.concatenate(fdes_all)
    logits = np.concatenate(logits_all)
    endpoints = np.concatenate(endpoints_all)
    n, k = ades.shape

    selected = np.argmax(logits, axis=1)
    oracle = np.argmin(ades, axis=1)
    minfde_idx = np.argmin(fdes, axis=1)
    row = np.arange(n)
    selected_ade = ades[row, selected]
    selected_fde = fdes[row, selected]
    minade = ades[row, oracle]
    minfde = fdes[row, minfde_idx]
    samepath_fde = fdes[row, oracle]
    regret = selected_ade - minade
    sorted_ade = np.sort(ades, axis=1)
    delta12 = sorted_ade[:, 1] - sorted_ade[:, 0]
    support05 = (fdes < 0.5).sum(axis=1)
    support10 = (fdes < 1.0).sum(axis=1)
    ade05 = (ades < 0.5).sum(axis=1)
    ade10 = (ades < 1.0).sum(axis=1)
    spread = mean_pairwise_endpoint_spread(endpoints)
    oracle_rank = score_rank_desc(logits, oracle)
    buckets = classify_bucket(regret, minade)

    records = []
    for i in range(n):
        records.append({
            "sample_id": int(sample_id[i]),
            "bucket": buckets[i],
            "selected_idx": int(selected[i]),
            "oracle_minade_idx": int(oracle[i]),
            "minfde_idx": int(minfde_idx[i]),
            "selected_ade": float(selected_ade[i]),
            "selected_fde": float(selected_fde[i]),
            "minade": float(minade[i]),
            "minfde": float(minfde[i]),
            "samepath_fde": float(samepath_fde[i]),
            "regret": float(regret[i]),
            "delta_ade_1_2": float(delta12[i]),
            "endpoint_spread": float(spread[i]),
            "support_endpoint_0p5": int(support05[i]),
            "support_endpoint_1p0": int(support10[i]),
            "count_ade_lt_0p5": int(ade05[i]),
            "count_ade_lt_1p0": int(ade10[i]),
            "oracle_score_rank": int(oracle_rank[i]),
            "minade_eq_minfde": int(oracle[i] == minfde_idx[i]),
        })

    summary = {
        "condition": spec["experiment_name"],
        "seed": seed,
        "diversity_weight": float(spec["diversity_weight"]),
        "diversity_margin": float(spec["diversity_margin"]),
        "N": n,
        "selected_ade_mean": float(selected_ade.mean()),
        "selected_fde_mean": float(selected_fde.mean()),
        "minade_mean": float(minade.mean()),
        "minfde_mean": float(minfde.mean()),
        "samepath_fde_mean": float(samepath_fde.mean()),
        "regret_mean": float(regret.mean()),
        "regret_median": float(np.median(regret)),
        "endpoint_spread_mean": float(spread.mean()),
        "endpoint_spread_median": float(np.median(spread)),
        "support1_mean": float(support10.mean()),
        "support05_mean": float(support05.mean()),
        "ade_lt1_mean": float(ade10.mean()),
        "ade_lt05_mean": float(ade05.mean()),
        "delta12_median": float(np.median(delta12)),
        "oracle_score_rank_median": float(np.median(oracle_rank)),
        "oracle_hit_rate": float(np.mean(selected == oracle)),
        "minade_eq_minfde_rate": float(np.mean(oracle == minfde_idx)),
        "r2_rate": float(np.mean(regret >= 0.50)),
        "A_rate_all": float(np.mean(buckets == "A_selection_dominant")),
        "B_rate_all": float(np.mean(buckets == "B_mixed")),
        "C_rate_all": float(np.mean(buckets == "C_generation_dominant")),
    }

    bucket_rows = []
    for bucket in ["R0_mild", "R1_medium", "A_selection_dominant", "B_mixed", "C_generation_dominant"]:
        m = buckets == bucket
        if not np.any(m):
            continue
        bucket_rows.append({
            "condition": spec["experiment_name"],
            "seed": seed,
            "bucket": bucket,
            "N": int(m.sum()),
            "pct_all": float(100.0 * m.mean()),
            "selected_ade_mean": float(selected_ade[m].mean()),
            "minade_mean": float(minade[m].mean()),
            "minade_median": float(np.median(minade[m])),
            "minfde_median": float(np.median(minfde[m])),
            "regret_mean": float(regret[m].mean()),
            "regret_median": float(np.median(regret[m])),
            "spread_median": float(np.median(spread[m])),
            "delta12_median": float(np.median(delta12[m])),
            "support05_mean": float(support05[m].mean()),
            "support10_mean": float(support10[m].mean()),
            "isolated_support10_le1_pct": float(100.0 * np.mean(support10[m] <= 1)),
            "ade05_mean_count": float(ade05[m].mean()),
            "ade10_mean_count": float(ade10[m].mean()),
            "oracle_rank_median": float(np.median(oracle_rank[m])),
            "oracle_rank_ge15_pct": float(100.0 * np.mean(oracle_rank[m] >= 15)),
            "minade_eq_minfde_pct": float(100.0 * np.mean(oracle[m] == minfde_idx[m])),
        })

    arrays = {
        "sample_id": sample_id,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "minade": minade,
        "minfde": minfde,
        "regret": regret,
        "delta12": delta12,
        "spread": spread,
        "support05": support05.astype(float),
        "support10": support10.astype(float),
        "ade05": ade05.astype(float),
        "ade10": ade10.astype(float),
        "oracle_rank": oracle_rank.astype(float),
        "bucket": buckets,
    }
    return spec, summary, bucket_rows, records, arrays


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=Path, default=HERE / "results" / "weight_sweep")
    p.add_argument("--astra-root", type=Path, default=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--baseline-condition", default="divw_0p050_m_0p400")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sweep_root = args.sweep_root.resolve()
    manifests = []
    mf = sweep_root / "sweep_manifest.json"
    if mf.is_file():
        manifests = json.loads(mf.read_text(encoding="utf-8"))
    else:
        for spec_path in sweep_root.glob("*/f16_temporal_social_cross/seed_*/val_scene_-1/diversity_ablation_spec.json"):
            manifests.append({"run_dir": str(spec_path.parent)})
    if not manifests:
        raise RuntimeError(f"No runs found under {sweep_root}")

    summaries, bucket_all, per_condition, arrays_by_key = [], [], {}, {}
    for item in manifests:
        run_dir = Path(item["run_dir"])
        spec, summary, bucket_rows, records, arrays = evaluate_condition(
            run_dir, args.astra_root.resolve(), device, args.batch_size
        )
        key = f"{spec['experiment_name']}__seed{spec['seed']}"
        summaries.append(summary)
        bucket_all.extend(bucket_rows)
        per_condition[key] = records
        arrays_by_key[key] = arrays
        write_csv(sweep_root / f"per_sample_{key}.csv", records)
        print(f"[EVAL] {key}: minADE={summary['minade_mean']:.4f}, support@1={summary['support1_mean']:.3f}, spread={summary['endpoint_spread_mean']:.3f}")

    summaries.sort(key=lambda x: (x["seed"], x["diversity_weight"], x["diversity_margin"]))
    write_csv(sweep_root / "condition_summary.csv", summaries)
    write_csv(sweep_root / "bucket_summary.csv", bucket_all)

    paired_rows = []
    by_seed = {}
    for s in summaries:
        by_seed.setdefault(int(s["seed"]), []).append(s)
    for seed, rows in by_seed.items():
        baseline_key = f"{args.baseline_condition}__seed{seed}"
        if baseline_key not in arrays_by_key:
            continue
        base = arrays_by_key[baseline_key]
        for s in rows:
            key = f"{s['condition']}__seed{seed}"
            if key == baseline_key:
                continue
            cur = arrays_by_key[key]
            if not np.array_equal(base["sample_id"], cur["sample_id"]):
                raise RuntimeError(f"Sample mismatch: {baseline_key} vs {key}")
            for metric, direction in [
                ("minade", "lower_better"),
                ("minfde", "lower_better"),
                ("selected_ade", "lower_better"),
                ("regret", "lower_better"),
                ("spread", "diagnostic"),
                ("delta12", "lower_means_less_isolated"),
                ("support10", "higher_better_for_dense_GT_coverage"),
                ("support05", "higher_better_for_dense_GT_coverage"),
                ("ade10", "higher_better_for_path_coverage"),
                ("ade05", "higher_better_for_path_coverage"),
            ]:
                # Positive delta means current - baseline.
                delta = cur[metric] - base[metric]
                mean, lo, hi = bootstrap_mean_ci(delta)
                paired_rows.append({
                    "seed": seed,
                    "condition": s["condition"],
                    "baseline": args.baseline_condition,
                    "metric": metric,
                    "interpretation": direction,
                    "mean_delta_current_minus_baseline": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                })
    write_csv(sweep_root / "paired_vs_baseline.csv", paired_rows)

    # Baseline reproducibility check against the archived EXP-173 ETH seed-1 validation facts.
    ref = {"selected_ade": 0.5388688445091248, "minade": 0.2195655256509781, "minfde": 0.39603129029273987}
    repro_lines = []
    bkey = f"{args.baseline_condition}__seed1"
    if bkey in arrays_by_key:
        b = arrays_by_key[bkey]
        got = {
            "selected_ade": float(b["selected_ade"].mean()),
            "minade": float(b["minade"].mean()),
            "minfde": float(b["minfde"].mean()),
        }
        for k, v in ref.items():
            repro_lines.append(f"- {k}: current={got[k]:.9f}, archived={v:.9f}, abs_delta={abs(got[k]-v):.3e}")

    # Compact report.
    lines = [
        "# F16 Diversity Loss Ablation Report",
        "",
        "## Protocol",
        "",
        "- EXP-173 V2 / F16 / ETH validation only",
        "- Same H3 initialization, same scene-tail split, same seed and training schedule",
        "- Only diversity weight/margin are changed",
        "- Baseline: weight=0.05, margin=0.4",
        "",
        "## Baseline reproduction check",
        "",
        *(repro_lines or ["- Baseline seed-1 run not found."]),
        "",
        "## Condition summary",
        "",
        "| condition | w | margin | Selected ADE | minADE | minFDE | regret | spread | support@1m | ADE<1m count | ΔADE1,2 median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['condition']} (s{s['seed']}) | {s['diversity_weight']:.3f} | {s['diversity_margin']:.3f} | "
            f"{s['selected_ade_mean']:.4f} | {s['minade_mean']:.4f} | {s['minfde_mean']:.4f} | "
            f"{s['regret_mean']:.4f} | {s['endpoint_spread_mean']:.3f} | {s['support1_mean']:.2f} | "
            f"{s['ade_lt1_mean']:.2f} | {s['delta12_median']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "The diversity-loss hypothesis is supported if reducing diversity consistently decreases endpoint spread and increases GT endpoint support / reduces oracle isolation, while minADE/minFDE do not materially deteriorate. A single improved scalar is insufficient; use the paired bootstrap table to inspect direction and uncertainty.",
        "",
        "Do not inspect test results until a setting is selected on validation and then confirmed across seeds.",
    ]
    (sweep_root / "DIVERSITY_ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {sweep_root / 'DIVERSITY_ABLATION_REPORT.md'}")


if __name__ == "__main__":
    main()
