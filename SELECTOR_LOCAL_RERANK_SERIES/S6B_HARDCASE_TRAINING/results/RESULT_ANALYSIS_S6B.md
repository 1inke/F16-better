# EXP-S6B: Hard-Case Specialized Training Report

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples)  
> **Evaluation Protocol**: Leave-One-Fold-Out with Nested Inner Calibration  
> **Target Finalists**: 4 S6A Pareto Finalists $\times$ 3 Schemes (B0 Baseline, B1 Weighted, B2 Curriculum)

---

## 1. 方案对比排行榜 (S6B Performance Comparison)

| Base Finalist | Scheme | ADE (Macro / Micro) | FDE (Macro / Micro) | Regret Macro | Near-Oracle@5cm | False Interv. | Useful Interv. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`K5_tiny_set_attention_g0124_gl`** | B0_baseline | **0.5172m** / 0.5045m | 1.0730m / 1.0959m | 0.2990m | 35.22% | 2.14% | 8.72% |
| **`K5_tiny_set_attention_g0124_gl`** | B1_weighted | **0.5158m** / 0.5010m | 1.0690m / 1.0878m | 0.2977m | 35.38% | 1.61% | 9.19% |
| **`K5_tiny_set_attention_g0124_gl`** | B2_curriculum | **0.5149m** / 0.5037m | 1.0693m / 1.0948m | 0.2967m | 35.23% | 3.29% | 11.66% |
| **`K3_linear_residual_g012_gl`** | B0_baseline | **0.5102m** / 0.4948m | 1.0584m / 1.0788m | 0.2921m | 35.37% | 2.19% | 10.62% |
| **`K3_linear_residual_g012_gl`** | B1_weighted | **0.5112m** / 0.4938m | 1.0600m / 1.0768m | 0.2931m | 35.34% | 1.94% | 9.32% |
| **`K3_linear_residual_g012_gl`** | B2_curriculum | **0.5110m** / 0.4952m | 1.0597m / 1.0789m | 0.2929m | 35.41% | 1.95% | 9.61% |
| **`K10_tiny_set_attention_g012_gl`** | B0_baseline | **0.5129m** / 0.5072m | 1.0610m / 1.1014m | 0.2948m | 35.24% | 1.12% | 6.28% |
| **`K10_tiny_set_attention_g012_gl`** | B1_weighted | **0.5151m** / 0.5085m | 1.0659m / 1.1038m | 0.2969m | 35.46% | 1.64% | 7.62% |
| **`K10_tiny_set_attention_g012_gl`** | B2_curriculum | **0.5163m** / 0.5040m | 1.0695m / 1.0949m | 0.2982m | 35.42% | 1.48% | 7.31% |
| **`K3_linear_residual_g0124_gl`** | B0_baseline | **0.5104m** / 0.4956m | 1.0576m / 1.0796m | 0.2923m | 35.09% | 3.03% | 12.07% |
| **`K3_linear_residual_g0124_gl`** | B1_weighted | **0.5097m** / 0.4935m | 1.0542m / 1.0762m | 0.2916m | 35.44% | 2.74% | 12.47% |
| **`K3_linear_residual_g0124_gl`** | B2_curriculum | **0.5094m** / 0.4936m | 1.0556m / 1.0762m | 0.2913m | 34.95% | 2.11% | 9.66% |

---

## 2. 核心科学发现

1. **Hard-Case 样本加权收益边际客观评估**：
   - Hard-case weighting and curriculum provide marginal and configuration-dependent changes.
   - 在残差线性模型上，收益约在 0~1mm 之间；而在部分注意力模型上，加权未带来显著超越基线的稳健增益，不构成突破性改观。
2. **课程学习 (B2 Curriculum) 动态平滑能力**：
   - 先用全量样本学习全局流形，再逐步聚焦在困难样本，有助于提升决策稳定性，但对整体宏观指标影响有限。
3. **入选后续阶段 (Stage 3 Gate & Adaptive-K)**：
   - 最优线性加权版本作为代表进入 EXP-S6D 门控联合评估。
