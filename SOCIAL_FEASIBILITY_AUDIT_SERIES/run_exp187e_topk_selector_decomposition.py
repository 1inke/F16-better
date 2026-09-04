"""EXP-187E: Top-K Social Support & Selector Decomposition Audit.

Investigates whether F16 collision regressions originate from:
1. SELECTOR_BOTTLENECK: Safe candidate exists in Top-20, but Top-1 failed to select it.
2. CANDIDATE_SUPPORT_BOTTLENECK: No safe candidate exists in the Top-20 pool.
3. TEMPORAL_PATH_BOTTLENECK: Endpoint is safe, but intermediate trajectory collides.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASTRA_ROOT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = ASTRA_ROOT / "datasets" / "eth_ucy" / "processed_data_diverse"
CFG_PATH = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
TRAINER = ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
MODEL_FILE = ROOT / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
ADAPTER = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
EXP187A = ROOT / "FINAL_END_TO_END_ENCODER_ABLATION_SERIES" / "EXP_187A_END_TO_END_ENCODER_ABLATION" / "results"
EXP173 = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results"
DEFAULT_OUT = HERE / "EXP_187E_TOPK_SELECTOR_DECOMPOSITION"

FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
MODELS = {
    "e0": {"variant": "f0_baseline", "root": EXP187A, "label": "e0"},
    "e1": {"variant": "f3_factorized", "root": EXP187A, "label": "e1"},
    "e2": {"variant": "f4_relation_tokens", "root": EXP173, "label": "e2"},
    "f16": {"variant": "f16_temporal_social_cross", "root": EXP173, "label": "e3"},
}
COLLISION_THRESHOLD = 0.20
NEAR_THRESHOLD = 0.50
K_LIST = (1, 5, 10, 20)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(device: torch.device):
    os.chdir(ASTRA_ROOT)
    sys.path[:0] = [str(ASTRA_ROOT), str(ASTRA_ROOT / "scripts"), str(ROOT / "FRONT_ENCODER_SERIES")]
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20

    cfg = load_locked_q20(CFG_PATH)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    return cfg


def checkpoint_paths(model_name: str, fold: str, seed: int) -> tuple[Path, Path]:
    meta = MODELS[model_name]
    base = meta["root"] / fold
    h3 = base / "h3" / meta["label"] / meta["variant"] / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"
    final = base / "v2" / meta["label"] / meta["variant"] / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"
    return h3, final


def build_model(model_name: str, fold: str, seed: int, device: torch.device, models, adapter):
    meta = MODELS[model_name]
    h3, final = checkpoint_paths(model_name, fold, seed)
    if not h3.is_file() or not final.is_file():
        raise FileNotFoundError(h3 if not h3.is_file() else final)
    cfg = configure(device)
    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3)
    os.environ["EXP173_DECODER_INIT_SEED"] = str(seed)
    model = models.FrontEncoderASTRA(cfg, meta["variant"])
    adapter.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
    model.load_state_dict(torch.load(final, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval(), cfg


def evaluate_topk_decomposition(model, loader, trainer, device: torch.device) -> Dict[str, Any]:
    """对模型的所有候选轨迹（K=20）进行细粒度社交与选择器分解分析"""
    records = []
    started = time.perf_counter()

    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            batch_size = past.shape[0]
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            raw = output[2] if isinstance(output, (tuple, list)) else output
            # (batch, agents, K, 12, 2)
            prediction = raw + center.unsqueeze(-3)
            logits = model.last_mode_logits  # (batch, agents, K)

            focal_preds = prediction[:, 0]  # (B, 20, 12, 2)
            focal_gt = future[:, 0]        # (B, 12, 2)
            nbr_gt = future[:, 1:]          # (B, N-1, 12, 2)
            focal_logits = logits[:, 0]     # (B, 20)

            # 排序：按 logit 降序排列候选 [0..19]
            sorted_indices = focal_logits.argsort(dim=-1, descending=True)  # (B, 20)

            # 计算各候选相对于真实目标的 ADE / FDE
            # diff: (B, 20, 12, 2)
            gt_diff = focal_preds - focal_gt[:, None, :, :]
            step_errors = torch.linalg.vector_norm(gt_diff, dim=-1)  # (B, 20, 12)
            cand_ade = step_errors.mean(dim=-1)                      # (B, 20)
            cand_fde = step_errors[:, :, -1]                         # (B, 20)

            num_nbrs = nbr_gt.shape[1]
            has_neighbours = (num_valid >= 2)

            for b in range(batch_size):
                rec = {
                    "has_neighbours": bool(has_neighbours[b].item()),
                    "valid_nbr_count": int(num_valid[b].item() - 1),
                }

                order_b = sorted_indices[b]  # (20,)
                cand_ade_b = cand_ade[b]      # (20,)
                cand_fde_b = cand_fde[b]      # (20,)

                # Top-1 metrics (native chosen candidate)
                top1_idx = int(order_b[0].item())
                rec["top1_ade"] = float(cand_ade_b[top1_idx].item())
                rec["top1_fde"] = float(cand_fde_b[top1_idx].item())
                rec["min_ade20"] = float(cand_ade_b.min().item())
                rec["min_fde20"] = float(cand_fde_b.min().item())

                if not rec["has_neighbours"]:
                    # 无邻居场景
                    rec["eligible"] = False
                    records.append(rec)
                    continue

                rec["eligible"] = True
                valid_k = rec["valid_nbr_count"]
                nbrs_b = nbr_gt[b, :valid_k]  # (valid_k, 12, 2)

                # 计算 20 个候选与所有邻居的时空距离: (20, valid_k, 12)
                focal_b = focal_preds[b]  # (20, 12, 2)
                dist_to_nbrs = torch.linalg.vector_norm(
                    focal_b[:, None, :, :] - nbrs_b[None, :, :, :], dim=-1
                )  # (20, valid_k, 12)

                # 每个候选在整个轨迹上的最小距离: (20,)
                cand_min_dist = dist_to_nbrs.min(dim=-1).values.min(dim=-1).values  # (20,)
                # 每个候选在终点时刻（step 11）的最小距离: (20,)
                cand_endpoint_min_dist = dist_to_nbrs[:, :, -1].min(dim=-1).values   # (20,)

                # 判定每个候选的硬安全（>=0.2m）与接近安全（>=0.5m）
                cand_hard_safe = (cand_min_dist >= COLLISION_THRESHOLD)  # (20,) bool
                cand_near_safe = (cand_min_dist >= NEAR_THRESHOLD)        # (20,) bool
                cand_endpoint_safe = (cand_endpoint_min_dist >= COLLISION_THRESHOLD)  # (20,) bool

                # Top-1 自身碰撞状态
                top1_hard_collision = not bool(cand_hard_safe[top1_idx].item())
                top1_near_collision = not bool(cand_near_safe[top1_idx].item())
                rec["top1_hard_collision"] = top1_hard_collision
                rec["top1_near_collision"] = top1_near_collision
                rec["top1_min_clearance"] = float(cand_min_dist[top1_idx].item())

                # 终点安全但路径碰撞 (Endpoint safe but intermediate path collision)
                rec["top1_endpoint_safe_path_collision"] = bool(
                    top1_hard_collision and cand_endpoint_safe[top1_idx].item()
                )

                # 计算 Top-1 发生碰撞的时间步 (first collision step) 与持续步数
                if top1_hard_collision:
                    top1_steps_dist = dist_to_nbrs[top1_idx].min(dim=0).values  # (12,)
                    colliding_steps = (top1_steps_dist < COLLISION_THRESHOLD)
                    rec["first_collision_step"] = int(torch.where(colliding_steps)[0][0].item()) + 1
                    rec["collision_duration"] = int(colliding_steps.sum().item())
                else:
                    rec["first_collision_step"] = None
                    rec["collision_duration"] = 0

                # 针对不同 K 分析候选池的安全覆盖情况
                for k in K_LIST:
                    k_cands = order_b[:k]  # 前 K 个候选索引
                    k_hard_safe = cand_hard_safe[k_cands]
                    k_near_safe = cand_near_safe[k_cands]
                    k_min_clearance = cand_min_dist[k_cands]

                    rec[f"hard_safe_exists@{k}"] = bool(k_hard_safe.any().item())
                    rec[f"near_safe_exists@{k}"] = bool(k_near_safe.any().item())
                    rec[f"max_min_clearance@{k}"] = float(k_min_clearance.max().item())

                    # 若存在安全候选，计算安全候选中的最小 ADE/FDE
                    if rec[f"hard_safe_exists@{k}"]:
                        safe_mask = k_hard_safe
                        safe_ade = cand_ade_b[k_cands][safe_mask].min().item()
                        safe_fde = cand_fde_b[k_cands][safe_mask].min().item()
                        rec[f"safe_minADE@{k}"] = float(safe_ade)
                        rec[f"safe_minFDE@{k}"] = float(safe_fde)
                        rec[f"safe_oracle_ADE_regret@{k}"] = float(safe_ade - rec["min_ade20"])
                        rec[f"safe_oracle_FDE_regret@{k}"] = float(safe_fde - rec["min_fde20"])
                    else:
                        rec[f"safe_minADE@{k}"] = None
                        rec[f"safe_minFDE@{k}"] = None
                        rec[f"safe_oracle_ADE_regret@{k}"] = None
                        rec[f"safe_oracle_FDE_regret@{k}"] = None

                # 核心分解指标：当 Top-1 发生碰撞时，Top-20 中是否存在安全替代项
                if top1_hard_collision:
                    rec["top1_coll_has_safe_alt@20"] = rec["hard_safe_exists@20"]
                else:
                    rec["top1_coll_has_safe_alt@20"] = None

                records.append(rec)

    elapsed = time.perf_counter() - started
    return {"records": records, "elapsed_seconds": elapsed}


def summarize_profile(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总单个 fold 的全量指标"""
    total_samples = len(records)
    eligible_records = [r for r in records if r.get("eligible", False)]
    eligible_count = len(eligible_records)

    # 基础精度
    top1_ade = np.mean([r["top1_ade"] for r in records])
    top1_fde = np.mean([r["top1_fde"] for r in records])
    min_ade20 = np.mean([r["min_ade20"] for r in records])
    min_fde20 = np.mean([r["min_fde20"] for r in records])

    if eligible_count == 0:
        return {
            "total_samples": total_samples, "eligible_count": 0,
            "top1_ade": float(top1_ade), "top1_fde": float(top1_fde),
            "min_ade20": float(min_ade20), "min_fde20": float(min_fde20),
        }

    # Top-1 碰撞与净空
    top1_hard_coll = np.mean([1.0 if r["top1_hard_collision"] else 0.0 for r in eligible_records])
    top1_near_coll = np.mean([1.0 if r["top1_near_collision"] else 0.0 for r in eligible_records])
    clearances = [r["top1_min_clearance"] for r in eligible_records]
    mean_clearance = float(np.mean(clearances))
    q1_clearance = float(np.percentile(clearances, 1))
    q5_clearance = float(np.percentile(clearances, 5))
    bottom_5_cvar = float(np.mean([c for c in clearances if c <= q5_clearance]))

    # 路径 vs 终点碰撞分解
    colliding_records = [r for r in eligible_records if r["top1_hard_collision"]]
    coll_count = len(colliding_records)
    if coll_count > 0:
        # 终点安全但路径碰撞占碰撞样本的比例
        endpoint_safe_path_coll_ratio = np.mean([1.0 if r["top1_endpoint_safe_path_collision"] else 0.0 for r in colliding_records])
        # Top-1 碰撞时，Top-20 存在安全替代候选的比例 (Selector Bottleneck Key Metric!)
        safe_alt_ratio = np.mean([1.0 if r["top1_coll_has_safe_alt@20"] else 0.0 for r in colliding_records])
        avg_first_step = float(np.mean([r["first_collision_step"] for r in colliding_records]))
        avg_duration = float(np.mean([r["collision_duration"] for r in colliding_records]))
    else:
        endpoint_safe_path_coll_ratio = 0.0
        safe_alt_ratio = 0.0
        avg_first_step = 0.0
        avg_duration = 0.0

    summary = {
        "total_samples": total_samples,
        "eligible_count": eligible_count,
        "top1_ade": float(top1_ade),
        "top1_fde": float(top1_fde),
        "min_ade20": float(min_ade20),
        "min_fde20": float(min_fde20),
        "top1_hard_collision_rate": float(top1_hard_coll),
        "top1_near_collision_rate": float(top1_near_coll),
        "mean_min_clearance": mean_clearance,
        "q1_clearance": q1_clearance,
        "q5_clearance": q5_clearance,
        "bottom_5_cvar": bottom_5_cvar,
        "collision_count": coll_count,
        "safe_alternative_exists_at_20_ratio": float(safe_alt_ratio),
        "endpoint_safe_path_coll_ratio": float(endpoint_safe_path_coll_ratio),
        "avg_first_collision_step": avg_first_step,
        "avg_collision_duration": avg_duration,
    }

    # 各 K 的安全覆盖
    for k in K_LIST:
        summary[f"hard_safe_exists@{k}"] = float(np.mean([1.0 if r[f"hard_safe_exists@{k}"] else 0.0 for r in eligible_records]))
        summary[f"near_safe_exists@{k}"] = float(np.mean([1.0 if r[f"near_safe_exists@{k}"] else 0.0 for r in eligible_records]))
        summary[f"max_min_clearance@{k}"] = float(np.mean([r[f"max_min_clearance@{k}"] for r in eligible_records]))

        safe_ade_vals = [r[f"safe_minADE@{k}"] for r in eligible_records if r[f"safe_minADE@{k}"] is not None]
        summary[f"safe_minADE@{k}"] = float(np.mean(safe_ade_vals)) if safe_ade_vals else None

        safe_regret_vals = [r[f"safe_oracle_ADE_regret@{k}"] for r in eligible_records if r[f"safe_oracle_ADE_regret@{k}"] is not None]
        summary[f"safe_oracle_ADE_regret@{k}"] = float(np.mean(safe_regret_vals)) if safe_regret_vals else None

    return summary


def run_experiment(device: torch.device, out_dir: Path, force: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = out_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)

    configure(device)
    models_mod = load_module("front_encoder_models", MODEL_FILE)
    adapter_mod = load_module("joint_v2_adapter", ADAPTER)
    trainer_mod = load_module("train_front_variant", TRAINER)

    all_fold_summaries = {}

    for model_name in ("e0", "e1", "e2", "f16"):
        all_fold_summaries[model_name] = {}
        for fold in FOLDS:
            cache_file = profiles_dir / f"{model_name}_{fold}_seed1.json"
            if cache_file.is_file() and not force:
                print(f"[skip] {model_name}/{fold} already computed.")
                all_fold_summaries[model_name][fold] = json.loads(cache_file.read_text(encoding="utf-8"))
                continue

            print(f"[eval] Running {model_name.upper()} on fold '{fold}'...")
            model, cfg = build_model(model_name, fold, seed=1, device=device, models=models_mod, adapter=adapter_mod)

            dataset = trainer_mod.TrajectoryDataset(DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy", cfg)
            loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

            res = evaluate_topk_decomposition(model, loader, trainer=trainer_mod, device=device)
            summary = summarize_profile(res["records"])
            summary["model"] = model_name
            summary["fold"] = fold
            summary["elapsed_seconds"] = res["elapsed_seconds"]

            cache_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            all_fold_summaries[model_name][fold] = summary
            print(f"  -> Top-1 Collision: {summary['top1_hard_collision_rate']*100:.2f}%, Safe-Alt@20: {summary['safe_alternative_exists_at_20_ratio']*100:.2f}%, EndptSafe-PathColl: {summary['endpoint_safe_path_coll_ratio']*100:.2f}%")

    # 聚合 5 折平均 (Equal-weight Macro Average)
    macro_summaries = {}
    for m in ("e0", "e1", "e2", "f16"):
        fold_res = [all_fold_summaries[m][f] for f in FOLDS]
        macro = {"model": m}
        keys = fold_res[0].keys()
        for k in keys:
            if k in ("model", "fold"):
                continue
            vals = [f_data[k] for f_data in fold_res if f_data.get(k) is not None]
            macro[k] = float(np.mean(vals)) if vals else None
        macro_summaries[m] = macro

    # 导出 CSV 产物
    # 1. per_fold.csv
    per_fold_csv = out_dir / "per_fold.csv"
    with open(per_fold_csv, "w", newline="", encoding="utf-8") as fp:
        fieldnames = ["model", "fold"] + [k for k in next(iter(all_fold_summaries["e0"].values())).keys() if k not in ("model", "fold")]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for m in ("e0", "e1", "e2", "f16"):
            for f in FOLDS:
                writer.writerow(all_fold_summaries[m][f])

    # 2. topk_feasibility.csv
    topk_csv = out_dir / "topk_feasibility.csv"
    with open(topk_csv, "w", newline="", encoding="utf-8") as fp:
        fields = ["model", "minADE20", "minFDE20", "Top1_Collision", "Safe_Alt_at_20", "Hard_Safe_at_1", "Hard_Safe_at_5", "Hard_Safe_at_10", "Hard_Safe_at_20", "Safe_minADE20", "Safe_Regret"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for m in ("e0", "e1", "e2", "f16"):
            mac = macro_summaries[m]
            writer.writerow({
                "model": m.upper(),
                "minADE20": f"{mac['min_ade20']:.4f}",
                "minFDE20": f"{mac['min_fde20']:.4f}",
                "Top1_Collision": f"{mac['top1_hard_collision_rate']*100:.2f}%",
                "Safe_Alt_at_20": f"{mac['safe_alternative_exists_at_20_ratio']*100:.2f}%",
                "Hard_Safe_at_1": f"{mac['hard_safe_exists@1']*100:.2f}%",
                "Hard_Safe_at_5": f"{mac['hard_safe_exists@5']*100:.2f}%",
                "Hard_Safe_at_10": f"{mac['hard_safe_exists@10']*100:.2f}%",
                "Hard_Safe_at_20": f"{mac['hard_safe_exists@20']*100:.2f}%",
                "Safe_minADE20": f"{mac.get('safe_minADE@20', 0):.4f}" if mac.get('safe_minADE@20') else "N/A",
                "Safe_Regret": f"{mac.get('safe_oracle_ADE_regret@20', 0):.4f}" if mac.get('safe_oracle_ADE_regret@20') else "N/A",
            })

    # 3. collision_case_decomposition.csv
    decomp_csv = out_dir / "collision_case_decomposition.csv"
    with open(decomp_csv, "w", newline="", encoding="utf-8") as fp:
        fields = ["model", "Top1_Collision_Rate", "Safe_Alt_Exist_Ratio", "Endpoint_Safe_Path_Coll_Ratio", "Avg_First_Coll_Step", "Avg_Coll_Duration", "Q1_Clearance", "Q5_Clearance", "Bottom_5%_CVaR"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for m in ("e0", "e1", "e2", "f16"):
            mac = macro_summaries[m]
            writer.writerow({
                "model": m.upper(),
                "Top1_Collision_Rate": f"{mac['top1_hard_collision_rate']*100:.2f}%",
                "Safe_Alt_Exist_Ratio": f"{mac['safe_alternative_exists_at_20_ratio']*100:.2f}%",
                "Endpoint_Safe_Path_Coll_Ratio": f"{mac['endpoint_safe_path_coll_ratio']*100:.2f}%",
                "Avg_First_Coll_Step": f"{mac['avg_first_collision_step']:.2f}",
                "Avg_Coll_Duration": f"{mac['avg_collision_duration']:.2f}",
                "Q1_Clearance": f"{mac['q1_clearance']:.3f}m",
                "Q5_Clearance": f"{mac['q5_clearance']:.3f}m",
                "Bottom_5%_CVaR": f"{mac['bottom_5_cvar']:.3f}m",
            })

    # 4. summary.json
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps({
        "experiment": "EXP-187E",
        "title": "Top-K Social Support & Selector Decomposition Audit",
        "macro_summaries": macro_summaries,
        "all_fold_summaries": all_fold_summaries
    }, indent=2), encoding="utf-8")

    # 5. 科学判决 (Root-cause decision)
    f16_mac = macro_summaries["f16"]
    e1_mac = macro_summaries["e1"]

    safe_alt = f16_mac["safe_alternative_exists_at_20_ratio"]
    endpoint_safe_path = f16_mac["endpoint_safe_path_coll_ratio"]

    if safe_alt >= 0.60:
        decision = "SELECTOR_BOTTLENECK"
        dec_desc = "F16 在超过 60% 的碰撞样本中，Top-20 候选池里均存在无碰撞的安全替代轨迹！这无可辩驳地证明 F16 的社交退化并非生成能力缺失，而是现有的 Native Score 选择器未能筛选出安全候选。"
    elif endpoint_safe_path >= 0.50:
        decision = "TEMPORAL_PATH_BOTTLENECK"
        dec_desc = "F16 碰撞中超过 50% 终点预测安全但轨迹中间步发生碰撞，表明瓶颈主要来自时域残差插值与曲率平滑，而非终点意图预测错误。"
    elif f16_mac["hard_safe_exists@20"] < e1_mac["hard_safe_exists@20"] - 0.05:
        decision = "CANDIDATE_SUPPORT_BOTTLENECK"
        dec_desc = "F16 的整个 Top-20 候选池的安全覆盖显著低于基线，证明注意力机制在候选生成阶段未能覆盖安全模态。"
    else:
        decision = "MIXED_OR_INCONCLUSIVE"
        dec_desc = "证据呈现混合特征，选择器偏置与时序轨迹插值均对社交退化产生显著影响。"

    # 6. 生成 RESULT_ANALYSIS.md
    analysis_md = out_dir / "RESULT_ANALYSIS.md"
    analysis_content = f"""# EXP-187E: Top-K Social Support & Selector Decomposition Audit

> **Protocol**: `focal_topk_decomposition_vs_gt_neighbors_v1`  
> **Status**: **COMPLETE**  
> **Scientific Decision**: **{decision}**

## 1. 核心判定与结论

**判决结果**: `{decision}`

{dec_desc}

## 2. Top-K 安全覆盖与选择器分解 (Five-Fold Macro Average)

| 模型 | minADE20 | minFDE20 | Top-1 碰撞率 | 碰撞中存在安全候选比例 (@20) | Hard-Safe@1 | Hard-Safe@5 | Hard-Safe@10 | Hard-Safe@20 | Safe-minADE20 | Safe Regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **E0** | {macro_summaries['e0']['min_ade20']:.4f} | {macro_summaries['e0']['min_fde20']:.4f} | {macro_summaries['e0']['top1_hard_collision_rate']*100:.2f}% | {macro_summaries['e0']['safe_alternative_exists_at_20_ratio']*100:.2f}% | {macro_summaries['e0']['hard_safe_exists@1']*100:.2f}% | {macro_summaries['e0']['hard_safe_exists@5']*100:.2f}% | {macro_summaries['e0']['hard_safe_exists@10']*100:.2f}% | {macro_summaries['e0']['hard_safe_exists@20']*100:.2f}% | {macro_summaries['e0'].get('safe_minADE@20', 0):.4f} | {macro_summaries['e0'].get('safe_oracle_ADE_regret@20', 0):.4f} |
| **E1** | {macro_summaries['e1']['min_ade20']:.4f} | {macro_summaries['e1']['min_fde20']:.4f} | {macro_summaries['e1']['top1_hard_collision_rate']*100:.2f}% | {macro_summaries['e1']['safe_alternative_exists_at_20_ratio']*100:.2f}% | {macro_summaries['e1']['hard_safe_exists@1']*100:.2f}% | {macro_summaries['e1']['hard_safe_exists@5']*100:.2f}% | {macro_summaries['e1']['hard_safe_exists@10']*100:.2f}% | {macro_summaries['e1']['hard_safe_exists@20']*100:.2f}% | {macro_summaries['e1'].get('safe_minADE@20', 0):.4f} | {macro_summaries['e1'].get('safe_oracle_ADE_regret@20', 0):.4f} |
| **E2** | {macro_summaries['e2']['min_ade20']:.4f} | {macro_summaries['e2']['min_fde20']:.4f} | {macro_summaries['e2']['top1_hard_collision_rate']*100:.2f}% | {macro_summaries['e2']['safe_alternative_exists_at_20_ratio']*100:.2f}% | {macro_summaries['e2']['hard_safe_exists@1']*100:.2f}% | {macro_summaries['e2']['hard_safe_exists@5']*100:.2f}% | {macro_summaries['e2']['hard_safe_exists@10']*100:.2f}% | {macro_summaries['e2']['hard_safe_exists@20']*100:.2f}% | {macro_summaries['e2'].get('safe_minADE@20', 0):.4f} | {macro_summaries['e2'].get('safe_oracle_ADE_regret@20', 0):.4f} |
| **F16** | **{macro_summaries['f16']['min_ade20']:.4f}** | **{macro_summaries['f16']['min_fde20']:.4f}** | {macro_summaries['f16']['top1_hard_collision_rate']*100:.2f}% | **{macro_summaries['f16']['safe_alternative_exists_at_20_ratio']*100:.2f}%** | {macro_summaries['f16']['hard_safe_exists@1']*100:.2f}% | {macro_summaries['f16']['hard_safe_exists@5']*100:.2f}% | {macro_summaries['f16']['hard_safe_exists@10']*100:.2f}% | **{macro_summaries['f16']['hard_safe_exists@20']*100:.2f}%** | **{macro_summaries['f16'].get('safe_minADE@20', 0):.4f}** | {macro_summaries['f16'].get('safe_oracle_ADE_regret@20', 0):.4f} |

## 3. 碰撞模式细粒度分解

| 模型 | Top-1 碰撞率 | 存在安全替代比 (@20) | 终点安全但路径碰撞比 | 平均首碰时间步 (1-12) | 平均碰撞持续步数 | Q1 最小间距 | Q5 最小间距 | 尾部 5% CVaR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E0** | {macro_summaries['e0']['top1_hard_collision_rate']*100:.2f}% | {macro_summaries['e0']['safe_alternative_exists_at_20_ratio']*100:.2f}% | {macro_summaries['e0']['endpoint_safe_path_coll_ratio']*100:.2f}% | {macro_summaries['e0']['avg_first_collision_step']:.2f} | {macro_summaries['e0']['avg_collision_duration']:.2f} | {macro_summaries['e0']['q1_clearance']:.3f} m | {macro_summaries['e0']['q5_clearance']:.3f} m | {macro_summaries['e0']['bottom_5_cvar']:.3f} m |
| **E1** | {macro_summaries['e1']['top1_hard_collision_rate']*100:.2f}% | {macro_summaries['e1']['safe_alternative_exists_at_20_ratio']*100:.2f}% | {macro_summaries['e1']['endpoint_safe_path_coll_ratio']*100:.2f}% | {macro_summaries['e1']['avg_first_collision_step']:.2f} | {macro_summaries['e1']['avg_collision_duration']:.2f} | {macro_summaries['e1']['q1_clearance']:.3f} m | {macro_summaries['e1']['q5_clearance']:.3f} m | {macro_summaries['e1']['bottom_5_cvar']:.3f} m |
| **E2** | {macro_summaries['e2']['top1_hard_collision_rate']*100:.2f}% | {macro_summaries['e2']['safe_alternative_exists_at_20_ratio']*100:.2f}% | {macro_summaries['e2']['endpoint_safe_path_coll_ratio']*100:.2f}% | {macro_summaries['e2']['avg_first_collision_step']:.2f} | {macro_summaries['e2']['avg_collision_duration']:.2f} | {macro_summaries['e2']['q1_clearance']:.3f} m | {macro_summaries['e2']['q5_clearance']:.3f} m | {macro_summaries['e2']['bottom_5_cvar']:.3f} m |
| **F16** | {macro_summaries['f16']['top1_hard_collision_rate']*100:.2f}% | {macro_summaries['f16']['safe_alternative_exists_at_20_ratio']*100:.2f}% | {macro_summaries['f16']['endpoint_safe_path_coll_ratio']*100:.2f}% | {macro_summaries['f16']['avg_first_collision_step']:.2f} | {macro_summaries['f16']['avg_collision_duration']:.2f} | {macro_summaries['f16']['q1_clearance']:.3f} m | {macro_summaries['f16']['q5_clearance']:.3f} m | {macro_summaries['f16']['bottom_5_cvar']:.3f} m |
"""
    analysis_md.write_text(analysis_content, encoding="utf-8")
    print(f"\n[COMPLETE] 实验分析与决策报告已写入: {analysis_md}")
    print(f"Scientific Decision: {decision}")


def main():
    parser = argparse.ArgumentParser(description="EXP-187E Top-K Social Support & Selector Decomposition Audit")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"EXP-187E running on device: {dev}")
    run_experiment(dev, Path(args.output_dir), force=args.force)


if __name__ == "__main__":
    main()
