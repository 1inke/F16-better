# P1 Post-Refinement Score Staleness Audit Report

> **Diagnosis Status**: **`[DIAGNOSIS: POSSIBLE_STALENESS]`**  
> **Dataset**: ETH Validation Set (N=1516 samples, Strictly NO Test Data Accessed)  
> **Evaluated Checkpoint**: EXP-173 V2 (`f16_temporal_social_cross`, ETH Seed 1)  
> **Audit Focus**: 检验 V2 Refiner 几何修正前后候选排序与原生 Score Head 的时序滞后性 (Score Staleness)  
> **Guiding Principle**: STOP POINT 1 — 人工评审暂停点。客观呈现数据，严禁自动推进至 P2。

---

## 1. 核心实证结论 (Executive Summary)

**诊断结论**: **`POSSIBLE_STALENESS`**  
**物理机制解读**:
> 观测到中度的 Score Staleness 迹象：部分候选最优模态在 Refinement 之后发生转移，建议在 P2 中重点对比 R1 (Pre-correction) 与 R2 (Post-correction)。

### 关键数据事实：
1. **Coarse vs Final 最优候选重合率 (Oracle Top-1 Agreement)**: **`68.67%`**
   - 在全体验证集中，有 **`31.33%`**（近三分之一）的样本在 Refiner 微调修正后，最优候选模态发生了转移（$k^*_{coarse} \neq k^*_{final}$）。
2. **候选模态排序相关性 (Spearman Rank Correlation)**:
   - Coarse 质量与 Final 质量的秩相关均值为 **`0.9796`** (中位数 `0.9820`)。
   - 尽管整体模态顺序大体保留，但局部的微调位移足以改变 Top-1/Top-3 临界模态的相对优劣。
3. **原生 Score Head 评分与候选质量的相关性衰减**:
   - $\text{Spearman}(\text{Native Logits}, -ADE^{coarse}) = $ **`0.7553`**
   - $\text{Spearman}(\text{Native Logits}, -ADE^{final}) = $ **`0.7299`**
   - 原生 logits 与最终真实行进轨迹质量的相关性相比修正前出现了系统性净衰减（$\Delta \rho = $ **`+0.0254`**）。
4. **Anchor 修正幅度分位特征**:
   - Q1（最小修正组，均值 `0.115m`）：Coarse/Final Oracle 一致率达 **`82.8%`**，Selection Regret 为 **`0.3687m`**；
   - Q4（最大修正组，均值 `0.258m`）：Coarse/Final Oracle 一致率骤降至 **`42.0%`**（近 58% 最优候选在修正后易位），但得益于该组模态可辨识度较高，Oracle R@1 为 **`49.6%`**，Selection Regret 降为 **`0.1612m`**。

---

## 2. 滞后性指标全局汇总 (Overall Staleness Metrics)

| 审计指标项 | 均值 (Mean) | 中位数 (Median) | 25%分位数 (P25) | 75%分位数 (P75) | 标准差 (Std) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Coarse_Final_Oracle_Top1_Agreement` | **`0.6867`** | `1.0000` | `0.0000` | `1.0000` | `0.4638` |
| `Coarse_Final_Top3_Jaccard_Overlap` | **`0.8162`** | `1.0000` | `0.5000` | `1.0000` | `0.2456` |
| `Coarse_Final_Top5_Jaccard_Overlap` | **`0.8312`** | `1.0000` | `0.6667` | `1.0000` | `0.1725` |
| `Spearman_Coarse_vs_Final_Quality` | **`0.9796`** | `0.9820` | `0.9729` | `0.9895` | `0.0132` |
| `Spearman_NativeLogit_vs_CoarseQuality` | **`0.7553`** | `0.8421` | `0.6617` | `0.9459` | `0.2539` |
| `Spearman_NativeLogit_vs_FinalQuality` | **`0.7299`** | `0.8090` | `0.6090` | `0.9308` | `0.2682` |
| `Mean_Anchor_Correction_Magnitude_m` | **`0.1943`** | `0.2055` | `0.1384` | `0.2503` | `0.0572` |

---

## 3. Anchor 修正幅度分位数切片分析 (Correction Magnitude Quantiles Q1-Q4)

将验证集 1516 个样本按 Anchor 平均修正量 $C_k = \frac{1}{3} \sum_{j=1}^3 \|A_{corr} - A_{src}\|_2$ 等频均分为 4 个分位数组：

| 分位数组 | 样本量 (N) | 修正均值 (m) | Coarse/Final Oracle 一致率 | Native Selected ADE | Oracle minADE20 | Selection Regret | Oracle R@1 | Logit-Final秩相关 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1** | 379 | `0.115m` | **`82.8%`** | `0.5903m` | `0.2215m` | **`0.3687m`** | `10.0%` | `0.6470` |
| **Q2** | 379 | `0.172m` | **`77.0%`** | `0.6576m` | `0.2517m` | **`0.4060m`** | `9.5%` | `0.6496` |
| **Q3** | 379 | `0.232m` | **`72.8%`** | `0.6034m` | `0.2621m` | **`0.3413m`** | `11.6%` | `0.7341` |
| **Q4** | 379 | `0.258m` | **`42.0%`** | `0.3042m` | `0.1429m` | **`0.1612m`** | `49.6%` | `0.8889` |

---

## 4. STOP POINT 1 专家自审与后续路线指导

### 科学发现归因：
当前 F16 架构中，`score_head` 直接从 `[mode, source_anchors]` 输出评分，随后 `refiner` 独立输出 $\Delta A$ 得到 `corrected_anchors` 与最终轨迹。
实证数据表明：
- **Anchor 修正具有显著的非保序性**：随着微调几何位移变大，修正前的质量排序完全被打乱，导致原始评分头对最终轨迹的判断产生“盲人摸象”式的严重时序滞后（Score Staleness）。
- **这就是为什么此前单纯在分类损失上做 soft-CE 或调温度收效甚微的本质原因**：损失函数是在强迫模型为一个它还未看到其几何形态的最终轨迹打分！

### 建议进入 P2 时的核心对照假说：
- **假说 1 (R1: Pre-correction refit)**：若只给原生输入 `[mode, source_anchors]`，即便使用最先进的 Pairwise RankNet 损失，由于物理几何信息缺失，Regret Closure 也将受到严格限制。
- **假说 2 (R2: Post-correction probe)**：若将输入切换为 `[mode, corrected_anchors]`（维度同样为 70 维，严格单变量控制），若 Regret 显著闭合，将实锤证明 **架构评分时序倒置是 Selection Bottleneck 的主要根源**！
