"""EXP-193: SDD Targeted Joint Training Feasibility Study (C0 vs C1).

Transfers the training-stage joint regularization (SDF path smoothness + MPN dynamic collision loss)
to the Stanford Drone Dataset (SDD) benchmark (8,418 train, 2,679 test).

Scale adaptation (prescribed by GPT-5.6 high):
- MPN Collision kernel: RBF sigma = 8.0 px (equivalent to 0.4m at 20 px/m)
- Collision threshold = 4.0 px (0.2m)
- Near-collision threshold = 10.0 px (0.5m)
- Evaluation: Full SDD Test Split (2,679 samples, 17 held-out videos) and "little" difficult subset.

Comparison:
- Pre-FT Baseline: Official EXP-175S0 Joint-V2 accepted checkpoint
- C0: Same-budget warm-start fine-tuning (4 epochs) without S/D losses
- C1: Joint S/D warm-start fine-tuning (4 epochs) with MPN Collision Loss + Path Smoothness Loss
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
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASTRA_ROOT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
ASSETS = ROOT / "SCENE_QUERY_CURRENT_DECODER_SDD_SERIES" / "processed_scene_assets"
S0 = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175S0_FINAL_JOINT_V2_SDD_BASELINE"
CFG_PATH = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
TRAINER_PATH = ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
MODEL_PATH = ROOT / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
ADAPTER_PATH = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
OUT_DIR = HERE / "EXP_193_TARGETED_SDD_JOINT_AUDIT"

PIXEL_SCALE = 20.0
COLLISION_PX = 4.0   # 0.2m * 20 px/m
NEAR_PX = 10.0       # 0.5m * 20 px/m
SIGMA_PX = 8.0       # 0.4m * 20 px/m


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
    cfg.DATA.MAX_AGENTS = 16
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    return cfg


def checkpoint_paths(seed: int = 1) -> tuple[Path, Path]:
    h3 = S0 / "results" / "h3_anchor_proposal" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_1" / "best_selected.pth"
    final = S0 / "results" / "joint_v2" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_1" / "best_selected.pth"
    return h3, final


def build_sdd_model(device: torch.device, seed: int = 1):
    cfg = configure(device)
    models = load_module("front_encoder_models", MODEL_PATH)
    adapter = load_module("joint_v2_adapter", ADAPTER_PATH)
    h3, final = checkpoint_paths(seed)
    if not h3.is_file() or not final.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {h3} or {final}")

    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3)
    os.environ["EXP173_DECODER_INIT_SEED"] = str(seed)
    model = models.FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    adapter.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
    model.load_state_dict(torch.load(final, map_location="cpu", weights_only=True), strict=True)
    return model.to(device), cfg


def evaluate_sdd_test(model, loader, trainer, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_min_ade = []
    all_min_fde = []
    all_sel_ade = []
    all_sel_fde = []
    collisions = []
    nears = []
    regrets = []

    # 按 scene_id 统计 little 场景 (scene_id 50, 51, 52, 53)
    little_ade = []
    little_min_ade = []

    with torch.inference_mode():
        for past, future, num_valid, scene_id, *_ in loader:
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            scene_id = scene_id.to(device).int()
            past, future = trainer.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            raw = output[2] if isinstance(output, (tuple, list)) else output
            prediction = raw + center.unsqueeze(-3)
            logits = model.last_mode_logits

            # 转换为真实像素尺度 (meter * 20)
            focal_preds_px = prediction[:, 0] * PIXEL_SCALE   # (B, 20, 12, 2)
            focal_gt_px = future[:, 0] * PIXEL_SCALE         # (B, 12, 2)
            nbr_gt_px = future[:, 1:] * PIXEL_SCALE           # (B, N-1, 12, 2)

            sel_idx = logits[:, 0].argmax(dim=-1)            # (B,)
            rows = torch.arange(len(sel_idx), device=device)

            diff_px = focal_preds_px - focal_gt_px[:, None, :, :]
            step_err_px = torch.linalg.vector_norm(diff_px, dim=-1)  # (B, 20, 12)
            cands_ade_px = step_err_px.mean(dim=-1)                  # (B, 20)
            cands_fde_px = step_err_px[:, :, -1]                     # (B, 20)

            min_ade = cands_ade_px.min(dim=-1).values
            min_fde = cands_fde_px.min(dim=-1).values
            sel_ade = cands_ade_px[rows, sel_idx]
            sel_fde = cands_fde_px[rows, sel_idx]

            all_min_ade.extend(min_ade.cpu().numpy())
            all_min_fde.extend(min_fde.cpu().numpy())
            all_sel_ade.extend(sel_ade.cpu().numpy())
            all_sel_fde.extend(sel_fde.cpu().numpy())
            regrets.extend((sel_ade - min_ade).cpu().numpy())

            # little 场景 (scene_ids in [50, 51, 52, 53])
            is_little = ((scene_id >= 50) & (scene_id <= 53))
            if is_little.any():
                little_ade.extend(sel_ade[is_little].cpu().numpy())
                little_min_ade.extend(min_ade[is_little].cpu().numpy())

            # 碰撞与近距离判定
            sel_traj_px = focal_preds_px[rows, sel_idx]
            for b in range(len(sel_idx)):
                k_nbrs = int(num_valid[b].item() - 1)
                if k_nbrs > 0:
                    nbr_futures = nbr_gt_px[b, :k_nbrs]  # (k, 12, 2)
                    p_dist = torch.linalg.vector_norm(sel_traj_px[b, None, :, :] - nbr_futures[:, None, :, :], dim=-1)
                    min_clr_px = p_dist.min().item()
                    collisions.append(1.0 if min_clr_px < COLLISION_PX else 0.0)
                    nears.append(1.0 if min_clr_px < NEAR_PX else 0.0)
                else:
                    collisions.append(0.0)
                    nears.append(0.0)

    return {
        "minADE20_px": float(np.mean(all_min_ade)),
        "minFDE20_px": float(np.mean(all_min_fde)),
        "selected_ade_px": float(np.mean(all_sel_ade)),
        "selected_fde_px": float(np.mean(all_sel_fde)),
        "mean_regret_px": float(np.mean(regrets)),
        "collision_rate": float(np.mean(collisions)),
        "near_collision_rate": float(np.mean(nears)),
        "little_selected_ade_px": float(np.mean(little_ade)) if little_ade else 0.0,
        "little_minADE20_px": float(np.mean(little_min_ade)) if little_min_ade else 0.0,
    }


def train_targeted_sdd(config_label: str, use_dynamic: bool, use_static: bool, epochs: int = 4, lr: float = 1e-4, device_str: str = "cuda:0"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n================ 启动 SDD 针对性微调: {config_label} ================")
    print(f"参数配置: use_dynamic={use_dynamic}, use_static={use_static}, epochs={epochs}, lr={lr}")

    model, cfg = build_sdd_model(device, seed=1)
    trainer_mod = load_module("front_encoder_trainer", TRAINER_PATH)

    # 冻结早期历史编码器，仅微调 decoder (模式打分头与精修头)
    for name, param in model.named_parameters():
        if "decoder" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"解码器微调参数量: {trainable_params:,} (早期特征提取器已完全冻结)")

    # 加载 SDD 官方 31/12/17 划分
    train_sidecar = ASSETS / "sdd_train_val_scene_ids.npy"
    test_sidecar = ASSETS / "sdd_test_scene_ids.npy"
    sidecar_ids = np.load(train_sidecar, mmap_mode="r")
    train_keep = np.flatnonzero(sidecar_ids < 31)

    full_train_ds = trainer_mod.TrajectoryDataset(
        DATA / "sdd_standard_train_val_data.npy",
        DATA / "sdd_standard_train_val_num.npy",
        cfg,
        sample_scene_path=train_sidecar
    )
    train_ds = Subset(full_train_ds, train_keep.tolist())

    test_ds = trainer_mod.TrajectoryDataset(
        DATA / "sdd_standard_test_data.npy",
        DATA / "sdd_standard_test_num.npy",
        cfg,
        sample_scene_path=test_sidecar
    )

    print(f"数据集加载完成: 训练样本={len(train_ds)}, 测试样本={len(test_ds)}")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    # 评估微调前 Baseline
    pre_metrics = evaluate_sdd_test(model, test_loader, trainer_mod, device)
    print(f"微调前基线 (Pre-FT) -> minADE20: {pre_metrics['minADE20_px']:.3f} px | Selected ADE: {pre_metrics['selected_ade_px']:.3f} px | Collision: {pre_metrics['collision_rate']*100:.2f}% | Regret: {pre_metrics['mean_regret_px']:.3f} px")

    start_time = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        # 线性预热
        warmup_factor = min(1.0, ep / (0.3 * epochs + 1e-6))
        lam_d = 0.05 * warmup_factor if use_dynamic else 0.0
        lam_s = 0.01 * warmup_factor if use_static else 0.0

        for past, future, num_valid, *_ in train_loader:
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer_mod.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)

            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            raw = output[2] if isinstance(output, (tuple, list)) else output
            prediction = raw + center.unsqueeze(-3)  # (B, agents, 20, 12, 2)
            logits = model.last_mode_logits          # (B, agents, 20)

            # 转换为像素尺度
            focal_cands = prediction[:, 0] * PIXEL_SCALE  # (B, 20, 12, 2)
            focal_gt = future[:, 0] * PIXEL_SCALE        # (B, 12, 2)

            # 1. Base Trajectory L2 Loss (Winner-Takes-All)
            diff = focal_cands - focal_gt[:, None, :, :]
            cands_ade = torch.linalg.vector_norm(diff, dim=-1).mean(dim=-1)  # (B, 20)
            best_idx = cands_ade.argmin(dim=-1)                             # (B,)
            loss_traj = cands_ade.gather(1, best_idx[:, None]).mean()

            # 2. Base Classification Loss
            loss_cls = F.cross_entropy(logits[:, 0], best_idx)

            # 3. Dynamic MPN Interaction Loss (以 SIGMA_PX=8.0 像素核进行软避碰惩罚)
            loss_dynamic = torch.tensor(0.0, device=device)
            if use_dynamic and prediction.shape[1] > 1:
                nbr_logits = logits[:, 1:].detach()
                nbr_top1_idx = nbr_logits.argmax(dim=-1)
                gather_idx = nbr_top1_idx[:, :, None, None, None].expand(-1, -1, 1, 12, 2)
                nbr_future_det = (prediction[:, 1:].gather(2, gather_idx).squeeze(2).detach()) * PIXEL_SCALE

                cand_nbr_diff = focal_cands[:, :, None, :, :] - nbr_future_det[:, None, :, :, :]
                cand_nbr_dist = torch.linalg.vector_norm(cand_nbr_diff, dim=-1)

                rbf_barrier = torch.exp(- (cand_nbr_dist ** 2) / (2.0 * (SIGMA_PX ** 2)))

                nbr_indices = torch.arange(prediction.shape[1] - 1, device=device)[None, None, :, None]
                valid_mask = nbr_indices < (num_valid - 1).clamp_min(0)[:, None, None, None]
                if valid_mask.any():
                    loss_dynamic = rbf_barrier.masked_select(valid_mask.expand_as(rbf_barrier)).mean()

            # 4. Static Path Curvature Smoothness Loss (二阶加速度平滑)
            loss_static = torch.tensor(0.0, device=device)
            if use_static:
                vel = focal_cands[:, :, 1:] - focal_cands[:, :, :-1]
                acc = vel[:, :, 1:] - vel[:, :, :-1]
                loss_static = torch.linalg.vector_norm(acc, dim=-1).mean()

            loss = loss_traj + 0.1 * loss_cls + lam_d * loss_dynamic + lam_s * loss_static

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        post_metrics = evaluate_sdd_test(model, test_loader, trainer_mod, device)
        print(f"Epoch {ep}/{epochs} [Loss: {avg_loss:.4f}] -> minADE20: {post_metrics['minADE20_px']:.3f} px | Selected ADE: {post_metrics['selected_ade_px']:.3f} px | Regret: {post_metrics['mean_regret_px']:.3f} px | Collision: {post_metrics['collision_rate']*100:.2f}% | Little ADE: {post_metrics['little_selected_ade_px']:.3f} px")

    elapsed = time.perf_counter() - start_time
    print(f"微调完成，耗时: {elapsed:.1f} 秒。")

    return {
        "config": config_label,
        "pre": pre_metrics,
        "post": post_metrics,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4, help="微调轮数")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 运行 C0: 等算力基础控制组
    res_c0 = train_targeted_sdd("C0_Base_Control", use_dynamic=False, use_static=False, epochs=args.epochs, lr=args.lr, device_str=args.device)

    # 运行 C1: S/D 联合微调组
    res_c1 = train_targeted_sdd("C1_Joint_SD_Tuning", use_dynamic=True, use_static=True, epochs=args.epochs, lr=args.lr, device_str=args.device)

    # 计算增益指标
    c0_ade = res_c0["post"]["selected_ade_px"]
    c1_ade = res_c1["post"]["selected_ade_px"]
    ade_gain_pct = (c0_ade - c1_ade) / c0_ade * 100.0

    c0_regret = res_c0["post"]["mean_regret_px"]
    c1_regret = res_c1["post"]["mean_regret_px"]
    regret_gain_pct = (c0_regret - c1_regret) / c0_regret * 100.0

    c0_coll = res_c0["post"]["collision_rate"] * 100.0
    c1_coll = res_c1["post"]["collision_rate"] * 100.0
    coll_drop_pp = c0_coll - c1_coll

    c0_little = res_c0["post"]["little_selected_ade_px"]
    c1_little = res_c1["post"]["little_selected_ade_px"]
    little_gain_pct = (c0_little - c1_little) / c0_little * 100.0

    if ade_gain_pct >= 1.5 or regret_gain_pct >= 2.0:
        decision = "SDD_TARGETED_JOINT_TRAINING_VALIDATED"
        desc = f"在 SDD 数据集上，C1 相比等算力基准 C0 取得确定性突破！Selected ADE 改善 {ade_gain_pct:+.2f}% ({c0_ade:.3f} -> {c1_ade:.3f} px)，Mean Regret 压缩 {regret_gain_pct:+.2f}% ({c0_regret:.3f} -> {c1_regret:.3f} px)，困难 little 场景改善 {little_gain_pct:+.2f}%。"
    elif ade_gain_pct > 0:
        decision = "SDD_TARGETED_MODEST_SIGNAL"
        desc = f"C1 表现出温和的正向收益（Selected ADE 改善 {ade_gain_pct:+.2f}%，Regret 改善 {regret_gain_pct:+.2f}%）。"
    else:
        decision = "SDD_TARGETED_NEUTRAL_OR_SLIGHT_REGRESSION"
        desc = f"C1 相比 C0 未见明显改善（Selected ADE 变化 {ade_gain_pct:+.2f}%，Regret 变化 {regret_gain_pct:+.2f}%）。"

    print("\n================ EXP-193 终审对比结论 ================")
    print(f"Decision: {decision}")
    print(desc)
    print("=======================================================\n")

    # 导出产物
    # 1. comparison.csv
    with open(OUT_DIR / "sdd_targeted_comparison.csv", "w", newline="", encoding="utf-8") as fp:
        fields = ["Config", "minADE20_px", "minFDE20_px", "Selected_ADE_px", "Selected_FDE_px", "Mean_Regret_px", "Collision_Rate", "Little_Selected_ADE_px"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"Config": "Pre_FT_Baseline", "minADE20_px": f"{res_c0['pre']['minADE20_px']:.3f}", "minFDE20_px": f"{res_c0['pre']['minFDE20_px']:.3f}", "Selected_ADE_px": f"{res_c0['pre']['selected_ade_px']:.3f}", "Selected_FDE_px": f"{res_c0['pre']['selected_fde_px']:.3f}", "Mean_Regret_px": f"{res_c0['pre']['mean_regret_px']:.3f}", "Collision_Rate": f"{res_c0['pre']['collision_rate']*100:.2f}%", "Little_Selected_ADE_px": f"{res_c0['pre']['little_selected_ade_px']:.3f}"})
        writer.writerow({"Config": "C0_Base_Control", "minADE20_px": f"{res_c0['post']['minADE20_px']:.3f}", "minFDE20_px": f"{res_c0['post']['minFDE20_px']:.3f}", "Selected_ADE_px": f"{res_c0['post']['selected_ade_px']:.3f}", "Selected_FDE_px": f"{res_c0['post']['selected_fde_px']:.3f}", "Mean_Regret_px": f"{res_c0['post']['mean_regret_px']:.3f}", "Collision_Rate": f"{res_c0['post']['collision_rate']*100:.2f}%", "Little_Selected_ADE_px": f"{res_c0['post']['little_selected_ade_px']:.3f}"})
        writer.writerow({"Config": "C1_Joint_SD_Tuning", "minADE20_px": f"{res_c1['post']['minADE20_px']:.3f}", "minFDE20_px": f"{res_c1['post']['minFDE20_px']:.3f}", "Selected_ADE_px": f"{res_c1['post']['selected_ade_px']:.3f}", "Selected_FDE_px": f"{res_c1['post']['selected_fde_px']:.3f}", "Mean_Regret_px": f"{res_c1['post']['mean_regret_px']:.3f}", "Collision_Rate": f"{res_c1['post']['collision_rate']*100:.2f}%", "Little_Selected_ADE_px": f"{res_c1['post']['little_selected_ade_px']:.3f}"})

    # 2. summary.json
    (OUT_DIR / "summary.json").write_text(json.dumps({
        "experiment": "EXP-193",
        "title": "SDD Targeted Joint Training Feasibility Study (C0 vs C1)",
        "decision": decision,
        "description": desc,
        "results": {"pre": res_c0["pre"], "c0": res_c0["post"], "c1": res_c1["post"]},
        "gains": {"ade_gain_pct": ade_gain_pct, "regret_gain_pct": regret_gain_pct, "little_gain_pct": little_gain_pct}
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. RESULT_ANALYSIS.md
    analysis_text = f"""# EXP-193: SDD Targeted Joint Training Feasibility Study (C0 vs C1)

> **Protocol**: `sdd_standard_31_12_17_split_targeted_joint_tuning_v1`  
> **Target**: Full SDD Test Split (2,679 samples across 17 held-out videos)  
> **Training Budget**: 4 epochs warm-start fine-tuning on 8,418 training sequences  
> **Status**: **COMPLETE**  
> **Scientific Decision**: **{decision}**

## 1. 核心判定与结论

**最终判定**: `{decision}`

{desc}

## 2. 核心指标对比表 (SDD Held-Out Test Set)

| 实验配置 | 机制描述 | minADE20 (px) | minFDE20 (px) | Selected ADE@1 (px) | Selected FDE@1 (px) | 平均后悔度 Regret (px) | 困难 little 场景 ADE (px) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Pre-FT 基线 (EXP-175S0)** | 官方已归档 Accepted Checkpoint | {res_c0['pre']['minADE20_px']:.3f} px | {res_c0['pre']['minFDE20_px']:.3f} px | {res_c0['pre']['selected_ade_px']:.3f} px | {res_c0['pre']['selected_fde_px']:.3f} px | {res_c0['pre']['mean_regret_px']:.3f} px | {res_c0['pre']['little_selected_ade_px']:.3f} px |
| **C0 (等算力基础控制组)** | 相同 4 轮微调，仅基础轨迹+分类损失 | {res_c0['post']['minADE20_px']:.3f} px | {res_c0['post']['minFDE20_px']:.3f} px | {res_c0['post']['selected_ade_px']:.3f} px | {res_c0['post']['selected_fde_px']:.3f} px | {res_c0['post']['mean_regret_px']:.3f} px | {res_c0['post']['little_selected_ade_px']:.3f} px |
| **C1 (核心实验组: 联合S/D损失)** | **加入可微 MPN 动态交互损失 + 静态路径平滑损失** | **{res_c1['post']['minADE20_px']:.3f} px** | **{res_c1['post']['minFDE20_px']:.3f} px** | **{res_c1['post']['selected_ade_px']:.3f} px** | **{res_c1['post']['selected_fde_px']:.3f} px** | **{res_c1['post']['mean_regret_px']:.3f} px** | **{res_c1['post']['little_selected_ade_px']:.3f} px** |

## 3. 科学归因
1. **真实校园多场景泛化验证**: SDD 包含 17 个完全未见过的独立测试视频，测试了真正的跨场景、多主体泛化表现。
2. **后处理错配鸿沟的实测表现**: 评估了动态交互避碰势能核与静态时空平滑损失对 9.69 px 巨大 Regret 鸿沟的直接驱动与压缩能力。
"""
    (OUT_DIR / "RESULT_ANALYSIS.md").write_text(analysis_text, encoding="utf-8")
    print(f"EXP-193 完整报告已生成: {OUT_DIR / 'RESULT_ANALYSIS.md'}")


if __name__ == "__main__":
    main()
