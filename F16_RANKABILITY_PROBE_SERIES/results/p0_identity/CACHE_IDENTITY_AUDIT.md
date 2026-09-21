# P0 Cache Numerical Identity & Gate Audit Report

> **Gate Status**: **`[STATUS: PASS]`**  
> **Dataset**: ETH Validation Set (N=1516 samples)  
> **Evaluated Checkpoint**: `C:\Users\linke\Documents\New project\QUALITY_AWARE_F16_SERIES_V2\results\p0_protocol\C0_EXP173_EXACT\f16_temporal_social_cross\seed_1\val_scene_-1\best_selected.pth`  
> **Cache File**: `C:\Users\linke\Documents\New project\F16_RANKABILITY_PROBE_SERIES\results\cache\val_seed1.npz`  
> **Execution Date**: 2026-09-21  

---

## 1. 数值一致性对齐检查 (Numerical Consistency Audit)

| 检查项 | 容许误差阈值 | 实测最大偏差 / 一致率 | 判定状态 |
| :--- | :---: | :---: | :---: |
| **最终候选轨迹数值偏差** (`candidate_final`) | $< 1.0 \times 10^{-6}\,\text{m}$ | **`0.00e+00`** | `PASS` |
| **原生分类 Logits 偏差** (`native_logits`) | $< 1.0 \times 10^{-6}$ | **`0.00e+00`** | `PASS` |
| **独立重算选模重合率** (`native_selected_index`) | $100.0\%$ | **`100.00%`** | `PASS` |
| **Final Oracle 标签一致率** (`oracle_final_index == argmin(ADE)`) | $100.0\%$ | **`100.00%`** | `PASS` |
| **Coarse Oracle 标签一致率** (`oracle_coarse_index == argmin(ADE_coa)`) | $100.0\%$ | **`100.00%`** | `PASS` |

---

## 2. 基线指标基准对齐 (Baseline Benchmark Verification)

从当前缓存数据计算得到的验证集核心性能指标：
- **Native Selected ADE**: **`0.5389 m`**（与指导文档基准 `0.5389 m` 误差为 **`0.000031 m`**）
- **Oracle minADE20 (Final)**: **`0.2196 m`**（与指导文档基准 `0.2196 m` 误差为 **`0.000034 m`**）
- **Native Selection Regret**: **`0.3193 m`**（与指导文档基准 `0.3193 m` 误差为 **`0.000003 m`**）

---

## 3. P0 门禁裁决

本轮缓存抽取过程严格保留了 focal agent 最后一帧观察点的相对坐标体系（去除了绝对场景全局偏置记忆风险），且所有中间特征（context 64维、mode 64维、source/corrected anchors）均通过原生的 PyTorch Hook 零侵入提取。

**结论**: **`P0 Cache Identity Gate: PASS`**。允许推进至 **P1 Post-refinement Score Staleness Audit**。
