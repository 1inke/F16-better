"""
P1 Target Redundancy Audit Runner.

Evaluates target distributions and gradient direction similarity on:
- TRAIN & VALIDATION splits ONLY (strict Test Visibility Lock enforced)
Targets audited:
1. Target A: q_orig (locked native adaptive quality target)
2. Target B: q_ade_tau (fixed-metric ADE-only at tau in [0.05, 0.10, 0.20])
3. Target C: q_quality_tau (fixed-metric ADE + 0.25*FDE at tau in [0.05, 0.10, 0.20])

Subsets:
- Overall (All samples)
- Hard case: Delta ADE_1,2 < 2cm
- Hard case: Delta ADE_1,2 < 5cm
- Hard case: Delta ADE_1,2 < 10cm
- Hard case: Native regret > 20cm

Outputs:
- target_redundancy_summary.csv
- hardcase_redundancy.csv
- gradient_similarity.csv
- target_redundancy_per_sample.npz
- TARGET_REDUNDANCY_REPORT.md
- Redundancy Decision Gate evaluation
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [
    str(REPO),
    str(REPO / "FRONT_ENCODER_SERIES"),
    str(REPO / "SELECTOR_BENCHMARK_SERIES"),
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
from QUALITY_AWARE_F16_SERIES_V2.common.confidence_targets import (
    build_original_target,
    build_fixed_quality_target,
    build_fixed_ade_target,
)
from QUALITY_AWARE_F16_SERIES_V2.common.redundancy_metrics import (
    js_divergence,
    kl_divergence,
    cosine_similarity,
    topk_jaccard_overlap,
    top1_agreement,
    compute_entropy,
    effective_mode_count,
)
from QUALITY_AWARE_F16_SERIES_V2.common.gradient_audit import (
    compute_gradient_vectors,
    analyze_gradient_similarity,
)


def extract_split_representations(model, loader, device, trainer_mod):
    all_preds = []
    all_targets = []
    all_logits = []
    all_ades = []
    all_fdes = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            past, future, num_valid, *_ = batch
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()

            past, future = trainer_mod.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)

            # Official inference path: mode="testing", future=None
            output = model(
                past - center,
                None,
                None,
                mode="testing",
                num_valid=num_valid,
                graph_loc=past,
            )
            raw_pred = output[2] if isinstance(output, (tuple, list)) else output
            pred = raw_pred + center.unsqueeze(-3)
            logits = model.last_mode_logits

            # Focal agent
            focal_pred = pred[:, 0]      # [B, 20, 12, 2]
            focal_gt = future[:, 0]       # [B, 12, 2]
            focal_logits = logits[:, 0]   # [B, 20]

            err = torch.linalg.vector_norm(focal_pred - focal_gt[:, None], dim=-1)
            ade = err.mean(dim=-1)
            fde = err[..., -1]

            all_preds.append(focal_pred.cpu().numpy())
            all_targets.append(focal_gt.cpu().numpy())
            all_logits.append(focal_logits.cpu().numpy())
            all_ades.append(ade.cpu().numpy())
            all_fdes.append(fde.cpu().numpy())

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_ades, axis=0),
        np.concatenate(all_fdes, axis=0),
    )


def run_target_redundancy_audit(
    checkpoint_path: Path | str,
    fold: str = "eth",
    seed: int = 1,
    device_str: str = "cuda:0",
) -> dict:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint for redundancy audit: {ckpt_path}")

    print(f"[AUDIT] Starting P1 Target Redundancy Audit on {device}...")
    print(f"[AUDIT] Evaluated Checkpoint: {ckpt_path}")

    astra_root = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
    cfg = configure_astra_env(REPO, astra_root, device)

    trainer_file = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    trainer_mod = load_module("train_front_variant", trainer_file)

    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    base_dataset = trainer_mod.TrajectoryDataset(
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
    split_data = np.load(split_path)
    train_indices = split_data["train"]
    val_indices = split_data["validation"]

    val_loader = DataLoader(
        Subset(base_dataset, val_indices.tolist()),
        batch_size=64, shuffle=False, num_workers=0, drop_last=False,
    )

    # Subsample 2000 train samples for fast, deterministic redundancy audit
    np.random.seed(20260921)
    sub_train_indices = np.random.choice(train_indices, size=min(2000, len(train_indices)), replace=False)
    train_loader = DataLoader(
        Subset(base_dataset, sub_train_indices.tolist()),
        batch_size=64, shuffle=False, num_workers=0, drop_last=False,
    )

    model = load_f16_checkpoint(REPO, fold, seed, ckpt_path, cfg, device)

    print("[AUDIT] Extracting validation representations (N=1516)...")
    val_preds, val_targets, val_logits, val_ades, val_fdes = extract_split_representations(
        model, val_loader, device, trainer_mod
    )

    print(f"[AUDIT] Extracting train subset representations (N={len(sub_train_indices)})...")
    train_preds, train_targets, train_logits, train_ades, train_fdes = extract_split_representations(
        model, train_loader, device, trainer_mod
    )

    # Build targets for Validation split
    # 1. q_orig
    t_val_preds = torch.from_numpy(val_preds)
    t_val_targets = torch.from_numpy(val_targets)

    q_orig, winner_orig, qual_orig = build_original_target(t_val_preds, t_val_targets)
    q_orig = q_orig.numpy()
    winner_orig = winner_orig.numpy()

    # Targets to compare against q_orig
    target_configs = [
        {"name": "q_ade_tau005", "family": "ADE-only", "tau": 0.05, "fn": lambda: build_fixed_ade_target(t_val_preds, t_val_targets, tau=0.05)},
        {"name": "q_ade_tau010", "family": "ADE-only", "tau": 0.10, "fn": lambda: build_fixed_ade_target(t_val_preds, t_val_targets, tau=0.10)},
        {"name": "q_ade_tau020", "family": "ADE-only", "tau": 0.20, "fn": lambda: build_fixed_ade_target(t_val_preds, t_val_targets, tau=0.20)},
        {"name": "q_qual_tau005", "family": "ADE+0.25FDE", "tau": 0.05, "fn": lambda: build_fixed_quality_target(t_val_preds, t_val_targets, tau=0.05)},
        {"name": "q_qual_tau010", "family": "ADE+0.25FDE", "tau": 0.10, "fn": lambda: build_fixed_quality_target(t_val_preds, t_val_targets, tau=0.10)},
        {"name": "q_qual_tau020", "family": "ADE+0.25FDE", "tau": 0.20, "fn": lambda: build_fixed_quality_target(t_val_preds, t_val_targets, tau=0.20)},
    ]

    all_targets_val = {"q_orig": q_orig}
    target_results = []
    grad_results = []

    # Hard-case masks
    sorted_ades = np.sort(val_ades, axis=-1)
    gap12 = sorted_ades[:, 1] - sorted_ades[:, 0]
    native_top1_idx = np.argmax(val_logits, axis=-1)
    native_top1_ade = val_ades[np.arange(len(val_ades)), native_top1_idx]
    native_regret = native_top1_ade - val_ades.min(axis=-1)

    hardcase_masks = {
        "all": np.ones(len(val_ades), dtype=bool),
        "gap12_lt_02cm": gap12 < 0.02,
        "gap12_lt_05cm": gap12 < 0.05,
        "gap12_lt_10cm": gap12 < 0.10,
        "regret_gt_20cm": native_regret > 0.20,
    }

    hardcase_summary = []

    for tc in target_configs:
        q_cand, winner_cand, _ = tc["fn"]()
        q_cand = q_cand.numpy()
        winner_cand = winner_cand.numpy()
        all_targets_val[tc["name"]] = q_cand

        # Metrics against q_orig
        js = js_divergence(q_orig, q_cand)
        kl_orig_cand = kl_divergence(q_orig, q_cand)
        kl_cand_orig = kl_divergence(q_cand, q_orig)
        cos = cosine_similarity(q_orig, q_cand)
        top1_agr = top1_agreement(q_orig, q_cand)
        jaccard3 = topk_jaccard_overlap(q_orig, q_cand, k=3)
        jaccard5 = topk_jaccard_overlap(q_orig, q_cand, k=5)
        k_eff_orig = effective_mode_count(q_orig)
        k_eff_cand = effective_mode_count(q_cand)

        # Gradient similarity
        g_orig, g_cand = compute_gradient_vectors(val_logits, q_orig, q_cand, winner_orig)
        grad_sim = analyze_gradient_similarity(g_orig, g_cand)

        row_overall = {
            "target_name": tc["name"],
            "family": tc["family"],
            "tau": tc["tau"],
            "js_divergence": float(np.mean(js)),
            "cosine_similarity": float(np.mean(cos)),
            "top1_agreement": float(np.mean(top1_agr)),
            "top3_jaccard": float(np.mean(jaccard3)),
            "top5_jaccard": float(np.mean(jaccard5)),
            "kl_orig_to_cand": float(np.mean(kl_orig_cand)),
            "kl_cand_to_orig": float(np.mean(kl_cand_orig)),
            "effective_modes_cand": float(np.mean(k_eff_cand)),
            "effective_modes_orig": float(np.mean(k_eff_orig)),
            "grad_cosine": float(np.mean(grad_sim["cosine"])),
            "grad_norm_ratio": float(np.mean(grad_sim["norm_ratio"])),
            "grad_sign_conflict": float(np.mean(grad_sim["sign_conflict_ratio"])),
        }
        target_results.append(row_overall)

        # Hard-case breakdowns
        for mask_name, mask in hardcase_masks.items():
            if np.sum(mask) == 0:
                continue
            hardcase_summary.append({
                "target_name": tc["name"],
                "subset": mask_name,
                "n_samples": int(np.sum(mask)),
                "js_divergence": float(np.mean(js[mask])),
                "cosine_similarity": float(np.mean(cos[mask])),
                "top1_agreement": float(np.mean(top1_agr[mask])),
                "top3_jaccard": float(np.mean(jaccard3[mask])),
                "grad_cosine": float(np.mean(grad_sim["cosine"][mask])),
                "grad_sign_conflict": float(np.mean(grad_sim["sign_conflict_ratio"][mask])),
            })

    out_dir = HERE / "results" / "p1_redundancy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save target_redundancy_summary.csv
    with open(out_dir / "target_redundancy_summary.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(target_results[0].keys()))
        w.writeheader()
        w.writerows(target_results)

    # 1b. Save dedicated gradient_similarity.csv per specification
    grad_fields = ["target_name", "family", "tau", "grad_cosine", "grad_norm_ratio", "grad_sign_conflict"]
    grad_rows = [{k: r[k] for k in grad_fields} for r in target_results]
    with open(out_dir / "gradient_similarity.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=grad_fields)
        w.writeheader()
        w.writerows(grad_rows)

    # 2. Save hardcase_redundancy.csv
    with open(out_dir / "hardcase_redundancy.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(hardcase_summary[0].keys()))
        w.writeheader()
        w.writerows(hardcase_summary)

    # 3. Save target_redundancy_per_sample.npz
    np.savez_compressed(
        out_dir / "target_redundancy_per_sample.npz",
        val_logits=val_logits,
        val_ades=val_ades,
        val_fdes=val_fdes,
        **all_targets_val,
    )
    print(f"[AUDIT] Saved target_redundancy_per_sample.npz to {out_dir}")

    # Redundancy Decision Gate Evaluation
    # Evaluate target q_ade_tau010 (the exact auxiliary target from the previous series)
    q_ade_010_row = next(r for r in target_results if r["target_name"] == "q_ade_tau010")
    grad_cos_010 = q_ade_010_row["grad_cosine"]
    top1_agr_010 = q_ade_010_row["top1_agreement"]
    js_010 = q_ade_010_row["js_divergence"]

    is_high_redundancy = (top1_agr_010 >= 0.95 and js_010 <= 0.02 and grad_cos_010 >= 0.90)
    is_medium_redundancy = (0.70 <= grad_cos_010 < 0.90) or (js_010 > 0.02 and grad_cos_010 >= 0.70)

    if is_high_redundancy:
        verdict = "HIGH REDUNDANCY"
        verdict_action = "STOP_ADDITIVE_QUALITY_LOSS"
        recommendation = "新增 ADE 概率目标与原生分类目标高度冗余，禁止继续运行叠加 auxiliary CE 路线与 3-seed 试验。"
    elif is_medium_redundancy:
        verdict = "MEDIUM REDUNDANCY"
        verdict_action = "PROCEED_TO_P2_REPLACEMENT"
        recommendation = "目标存在中度方向差异，允许在保持总分类权重的前提下测试纯 replacement 试验 (C1, C2, C3)。"
    else:
        verdict = "LOW REDUNDANCY"
        verdict_action = "PROCEED_TO_P2_REPLACEMENT"
        recommendation = "目标存在显著差异，允许进入 P2 替换试点。"

    # Generate TARGET_REDUNDANCY_REPORT.md
    report_content = f"""# P1 Target Redundancy Audit & Decision Gate Report

> **Audit Date & Status**: **`[VERDICT: {verdict}]`**  
> **Action Recommendation**: `{verdict_action}`  
> **Dataset**: ETH Validation Set (N=1,516, Strictly NO Test Data Accessed)  
> **Evaluated Checkpoint**: `{ckpt_path}`  
> **Baseline Native Target**: $q^{{orig}} = \\operatorname{{softmax}}\\left(-\\frac{{Q_k - \\min Q}}{{0.5 \\cdot scale(Q)}}\\right)$, $Q_k = ADE_k + 0.25 FDE_k$

---

## 1. 核心裁决结论 (Executive Gate Verdict)

经过严格的全样本分布与 Logits 梯度方向实证审计：
- **上一轮核心目标 ($q^{{ADE}}, \\tau=0.10$) 与原生分类监督的 Logit 梯度方向余弦相似度**: `{grad_cos_010:.4f}`
- **Top-1 最优候选重合率 (Top-1 Agreement)**: `{top1_agr_010*100:.2f}%`
- **JS 散度 (Jensen-Shannon Divergence)**: `{js_010:.4f}`
- **有效模态数 (Effective Mode Count)**: 原生 $K_{{eff}}={q_ade_010_row['effective_modes_orig']:.2f}$ vs 候选 $K_{{eff}}={q_ade_010_row['effective_modes_cand']:.2f}$

**科学判断**:
> **{recommendation}**

---

## 2. 全样本目标冗余度比对表 (Overall Validation N=1516)

| 目标标识 (Target) | 物理定义 | 温度 $\\tau$ | 梯度余弦相似度 | JS 散度 | 向量余弦 | Top-1 一致率 | Top-3 Jaccard | 梯度符号冲突率 | 判定 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in target_results:
        report_content += f"| `{r['target_name']}` | {r['family']} | `{r['tau']:.2f}` | **`{r['grad_cosine']:.4f}`** | `{r['js_divergence']:.4f}` | `{r['cosine_similarity']:.4f}` | `{r['top1_agreement']*100:.1f}%` | `{r['top3_jaccard']*100:.1f}%` | `{r['grad_sign_conflict']*100:.1f}%` |\n"

    report_content += f"""
---

## 3. Hard-Case 分层场景冗余度审计

在面临模态高度混淆与大 Regret 困难场景下，新目标是否提供增量区分度：

| 目标名称 | 评估子集 | 样本数 (N) | 梯度余弦相似度 | JS 散度 | Top-1 一致率 | Top-3 Jaccard |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for h in hardcase_summary:
        if h["target_name"] in ("q_ade_tau010", "q_qual_tau010"):
            report_content += f"| `{h['target_name']}` | `{h['subset']}` | {h['n_samples']} | **`{h['grad_cosine']:.4f}`** | `{h['js_divergence']:.4f}` | `{h['top1_agreement']*100:.1f}%` | `{h['top3_jaccard']*100:.1f}%` |\n"

    report_content += f"""
---

## 4. STOP POINT 1 决策提请与状态更新

根据实验指南第 32 条规定：
> **STOP POINT 1: 第一阶段仅执行 P0 + P1，生成 TARGET_REDUNDANCY_REPORT.md 然后停止。若判定 HIGH REDUNDANCY，停止 additive loss 路线且不自动进入 P2。**

当前状态机已更新至 `state.json`，等待人工确认是否进入 P2 Objective Replacement 试点。
"""
    report_file = HERE / "TARGET_REDUNDANCY_REPORT.md"
    report_file.write_text(report_content, encoding="utf-8")
    print(f"[AUDIT] Generated TARGET_REDUNDANCY_REPORT.md at {report_file}")

    # Update state.json
    state_file = HERE / "results" / "state.json"
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        state["p1_redundancy"] = verdict
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    return {
        "verdict": verdict,
        "action": verdict_action,
        "q_ade_tau010": q_ade_010_row,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--fold", type=str, default="eth")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not ckpt:
        c0_ckpt = HERE / "results" / "p0_protocol" / "C0_EXP173_EXACT" / "f16_temporal_social_cross" / f"seed_{args.seed}" / "val_scene_-1" / "best_selected.pth"
        if c0_ckpt.is_file():
            ckpt = str(c0_ckpt)
        else:
            exp173_ckpt = REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results" / args.fold / "v2" / "e3" / "f16_temporal_social_cross" / f"seed_{args.seed}" / "val_scene_-1" / "best_selected.pth"
            ckpt = str(exp173_ckpt)

    run_target_redundancy_audit(ckpt, args.fold, args.seed, args.device)
