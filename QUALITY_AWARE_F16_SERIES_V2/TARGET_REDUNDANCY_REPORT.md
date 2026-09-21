# P1 Target Redundancy Audit & Decision Gate Report

> **Audit Date & Status**: **`[VERDICT: MEDIUM REDUNDANCY]`**  
> **Action Recommendation**: `PROCEED_TO_P2_REPLACEMENT`  
> **Dataset**: ETH Validation Set (N=1,516, Strictly NO Test Data Accessed)  
> **Evaluated Checkpoint**: `C:\Users\linke\Documents\New project\QUALITY_AWARE_F16_SERIES_V2\results\p0_protocol\C0_EXP173_EXACT\f16_temporal_social_cross\seed_1\val_scene_-1\best_selected.pth`  
> **Baseline Native Target**: $q^{orig} = \operatorname{softmax}\left(-\frac{Q_k - \min Q}{0.5 \cdot scale(Q)}\right)$, $Q_k = ADE_k + 0.25 FDE_k$

---

## 1. 核心裁决结论 (Executive Gate Verdict)

经过严格的全样本分布与 Logits 梯度方向实证审计：
- **上一轮核心目标 ($q^{ADE}, \tau=0.10$) 与原生分类监督的 Logit 梯度方向余弦相似度**: `0.8163`
- **Top-1 最优候选重合率 (Top-1 Agreement)**: `88.39%`
- **JS 散度 (Jensen-Shannon Divergence)**: `0.1571`
- **有效模态数 (Effective Mode Count)**: 原生 $K_{eff}=12.38$ vs 候选 $K_{eff}=4.16$

**科学判断**:
> **目标存在中度方向差异，允许在保持总分类权重的前提下测试纯 replacement 试验 (C1, C2, C3)。**

---

## 2. 全样本目标冗余度比对表 (Overall Validation N=1516)

| 目标标识 (Target) | 物理定义 | 温度 $\tau$ | 梯度余弦相似度 | JS 散度 | 向量余弦 | Top-1 一致率 | Top-3 Jaccard | 梯度符号冲突率 | 判定 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `q_ade_tau005` | ADE-only | `0.05` | **`0.8300`** | `0.2739` | `0.7259` | `88.4%` | `91.6%` | `16.1%` |
| `q_ade_tau010` | ADE-only | `0.10` | **`0.8163`** | `0.1571` | `0.8264` | `88.4%` | `91.6%` | `13.2%` |
| `q_ade_tau020` | ADE-only | `0.20` | **`0.7735`** | `0.0465` | `0.9393` | `88.4%` | `91.6%` | `11.1%` |
| `q_qual_tau005` | ADE+0.25FDE | `0.05` | **`0.8799`** | `0.3281` | `0.6864` | `100.0%` | `100.0%` | `17.5%` |
| `q_qual_tau010` | ADE+0.25FDE | `0.10` | **`0.8641`** | `0.2393` | `0.7588` | `100.0%` | `100.0%` | `15.2%` |
| `q_qual_tau020` | ADE+0.25FDE | `0.20` | **`0.8279`** | `0.1139` | `0.8704` | `100.0%` | `100.0%` | `12.2%` |

---

## 3. Hard-Case 分层场景冗余度审计

在面临模态高度混淆与大 Regret 困难场景下，新目标是否提供增量区分度：

| 目标名称 | 评估子集 | 样本数 (N) | 梯度余弦相似度 | JS 散度 | Top-1 一致率 | Top-3 Jaccard |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `q_ade_tau010` | `all` | 1516 | **`0.8163`** | `0.1571` | `88.4%` | `91.6%` |
| `q_ade_tau010` | `gap12_lt_02cm` | 343 | **`0.4216`** | `0.0711` | `77.3%` | `96.5%` |
| `q_ade_tau010` | `gap12_lt_05cm` | 549 | **`0.5768`** | `0.0915` | `75.4%` | `94.2%` |
| `q_ade_tau010` | `gap12_lt_10cm` | 850 | **`0.6994`** | `0.1077` | `80.9%` | `92.4%` |
| `q_ade_tau010` | `regret_gt_20cm` | 851 | **`0.9414`** | `0.2029` | `91.0%` | `90.8%` |
| `q_qual_tau010` | `all` | 1516 | **`0.8641`** | `0.2393` | `100.0%` | `100.0%` |
| `q_qual_tau010` | `gap12_lt_02cm` | 343 | **`0.5554`** | `0.1357` | `100.0%` | `100.0%` |
| `q_qual_tau010` | `gap12_lt_05cm` | 549 | **`0.6939`** | `0.1623` | `100.0%` | `100.0%` |
| `q_qual_tau010` | `gap12_lt_10cm` | 850 | **`0.7870`** | `0.1839` | `100.0%` | `100.0%` |
| `q_qual_tau010` | `regret_gt_20cm` | 851 | **`0.9567`** | `0.2884` | `100.0%` | `100.0%` |

---

## 4. STOP POINT 1 决策提请与状态更新

根据实验指南第 32 条规定：
> **STOP POINT 1: 第一阶段仅执行 P0 + P1，生成 TARGET_REDUNDANCY_REPORT.md 然后停止。若判定 HIGH REDUNDANCY，停止 additive loss 路线且不自动进入 P2。**

当前状态机已更新至 `state.json`，等待人工确认是否进入 P2 Objective Replacement 试点。
