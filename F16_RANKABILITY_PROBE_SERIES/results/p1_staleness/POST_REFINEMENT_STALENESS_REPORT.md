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
- **微调修正后确实存在局部 Top-1 模态转移**：全体验证集中有 31.33% 的样本在 Refiner 微调后最优候选发生转移，且原生评分头对最终轨迹的秩相关出现轻微衰减（$\Delta \rho = +0.0254$），呈现出一定的 Score Staleness 现象。
- **定位为次要结构现象而非唯一主导根因**：Score staleness 是一个可观察到的结构现象，但后续 R1/R2 探针结果表明，仅显式提供 post-correction geometry 带来的性能增益较小（约 1.9 mm），因此它目前更适合被视为次要影响因素，暂不能直接断言为 Selection 问题的唯一本质根因。

### P2 对照假说验证：
- **假说 1 (R1: Pre-correction refit)**：仅重训评分损失，检验原输入信息是否已被充分利用。
- **假说 2 (R2: Post-correction probe)**：将输入替换为微调后几何 `[mode, corrected_anchors]`（严格单变量控制 70 维），精确量化后置几何修正带来的真实闭合增益。
