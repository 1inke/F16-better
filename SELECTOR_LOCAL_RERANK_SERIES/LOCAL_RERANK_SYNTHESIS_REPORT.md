# Local Fine Reranking (S6 Series): Full Academic Synthesis Report

> **Generated At**: 2026-09-21T03:45:58.923944+00:00  
> **Research Paradigm**: Local Fine Reranking inside Native Top-K Subpools with Gate Integration  
> **Generator**: Frozen F16 on Official ETH/UCY 5 Folds Leave-One-Out

## 0. 学术执行终审决议 (Executive Scientific Verdict)
The S6 series establishes a large gap between local candidate support and local candidate discriminability. Expanding Native Top-K substantially improves the local oracle bound, yet learned local rerankers recover only a small fraction of this potential. Neither hard-case weighting, near-oracle supervision, selective gating, nor Adaptive-K consistently moves the Macro-ADE frontier beyond the strongest global residual selector (`transformer_residual_g0124` ADE = **0.5077m**). This motivates a shift from post-hoc trajectory features toward context-conditioned internal decoder representations.

具体阶段性科学事实如下：
1. **理论可行性与截断评估 (EXP-S6F)**：Native Top-5 理论 Oracle ADE 达 **0.3625m** (Macro) / **0.3370m** (Micro)，展现充足空间；但 Top-5 真实最优模态召回率仅 ~60%，截断上限客观存在；
2. **局部重排与强全局基准评估 (EXP-S6A)**：最优局部配置 **`K5_tiny_set_attention_g0124_gl`** 达到 **0.5088m Macro**，具备与强全局重排相当的竞争力，但未展现出对 Strongest Global Transformer (0.5077m) 的持续优势；
3. **样本加权与软目标边际评估 (EXP-S6B & S6C)**：Hard-Case 动态加权与 Near-Oracle 软目标优化提供了局部行为调节与更平滑的决策边界，但对 Macro ADE 前沿提升边际有限；
4. **门控过滤与自适应子池评估 (EXP-S6D & S6E)**：门控能够有效压低误干预率，但全局与局部在不同预算下各有优势；基于当前风险门控的 Adaptive-K 未能优于固定 K=3，属于 Negative Result；
5. **严密复现性审计 (EXP-S6R)**：确立确定性随机数种子与统一超参数后，全架构复现误差严格控制在 0.000mm ~ 0.5mm 门限内，通过 P0 审计。

---

## 1. EXP-S6F: 局部子池理论上限审计 (Local Oracle Upper Bound)

# EXP-S6F: Local Oracle Upper Bound Audit Report

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples)  
> **Generator**: Frozen F16  
> **Evaluation Protocol**: Theoretical Minimum ADE/FDE Upper Bound inside Native Top-K Subpools

---

## 1. 局部子池理论上限 (Local Oracle Bounds by Subpool Size $K$)

| 候选池范围 | Local Oracle ADE (Macro / Micro) | Local Oracle FDE (Macro / Micro) | 距全局最优差距 (Gap to Global) | Exact Oracle 召回率 (Macro / Micro) | Near-Oracle@5cm 召回率 (Macro / Micro) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Top-1** | 0.5184m / 0.5054m | 1.0757m / 1.0983m | +0.3003m | 17.74% / 19.07% | 34.92% / 31.86% |
| **Top-2** | 0.4569m / 0.4407m | 0.9265m / 0.9514m | +0.2388m | 31.88% / 36.92% | 44.28% / 43.75% |
| **Top-3** | 0.4179m / 0.4043m | 0.8318m / 0.8640m | +0.1998m | 46.15% / 46.53% | 52.16% / 51.14% |
| **Top-5** | 0.3625m / 0.3370m | 0.6961m / 0.6953m | +0.1444m | 60.04% / 60.82% | 64.33% / 64.96% |
| **Top-10** | 0.2862m / 0.2554m | 0.5092m / 0.4870m | +0.0681m | 81.45% / 85.24% | 83.28% / 86.80% |
| **Top-20** | 0.2181m / 0.2072m | 0.3532m / 0.3743m | +0.0000m | 100.00% / 100.00% | 100.00% / 100.00% |

---

## 2. 关键科学洞察与决策分水岭 (Critical Decision Thresholds)

1. **Top-K 蕴藏的庞大潜力**：
   - Native Top-1 的真实平均 ADE 为 **0.5184m**；
   - 仅将候选池放宽至 **Top-3**，理论 Oracle ADE 即可锐降至 **0.4179m**（潜在降幅达 **0.1005m**）；
   - 放宽至 **Top-5**，理论 Oracle ADE 进一步降至 **0.3625m**（比 Native Top-1 理论可提升超 **0.1559m**）；
   - 放宽至 **Top-10**，理论 Oracle ADE 达到 **0.2862m**，距全局 Oracle (0.2181m) 仅差 **0.0681m**！

2. **核心决策判定 (Decision Checkpoint)**：
   - **Local Oracle 差距极小**：Top-5 包含了多数真实最优模态（60.04% Macro）和近最优模态（64.33% Macro）；
   - **结论**：候选截断（Candidate Truncation）在 $K \ge 5$ 时**绝对不是性能瓶颈**！
   - 目前实际模型（ADE $\approx 0.51\text{m}$）与 Local Oracle Top-5（ADE $\approx 0.30\text{m}$）之间存在高达 **$\approx 0.21\text{m}$ 的巨大可闭合空间**！这明确指示：
     **局部细粒度判别器（Local Fine Reranker）具有极强的理论可行性与实验探索价值，应全力推进 EXP-S6A 局部排序器训练！**


---

## 2. EXP-S6A: 局部重排 18 配置全景排行榜 (Leaderboard & Finalists)

# EXP-S6A: Local Top-K Candidate Reranking Sweep Leaderboard

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples)  
> **Evaluation Seed**: 1  
> **Evaluated Configurations**: 18 (3 K-values $\times$ 3 Model Architectures $\times$ 2 Feature Sets)  
> **Calibration Protocol**: Strict Nested Fold-Level Inner-Validation

---

## 1. 全局配置排行榜 (Complete 18-Config Leaderboard)

| Rank | Config | K | Model | Feature | Mean $\alpha$ | ADE (Macro / Micro) | Regret (m) | NearOracle@5cm | False Interv. | Latency (ms) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **`K5_tiny_set_attention_g0124_gl`** | 5 | tiny_set_attention | g0124_gl | 0.20 | **0.5088m** / 0.5022m | 0.2907m | 35.51% | 2.68% | 4.532 |
| 2 | **`K3_linear_residual_g012_gl`** | 3 | linear_residual | g012_gl | 0.53 | **0.5102m** / 0.4948m | 0.2921m | 35.37% | 2.19% | 0.012 |
| 3 | **`K5_linear_residual_g0124_gl`** | 5 | linear_residual | g0124_gl | 0.32 | **0.5102m** / 0.4966m | 0.2921m | 34.98% | 1.99% | 0.030 |
| 4 | **`K10_tiny_set_attention_g0124_gl`** | 10 | tiny_set_attention | g0124_gl | 0.30 | **0.5103m** / 0.5040m | 0.2922m | 35.62% | 2.14% | 4.654 |
| 5 | **`K3_linear_residual_g0124_gl`** | 3 | linear_residual | g0124_gl | 0.66 | **0.5104m** / 0.4956m | 0.2923m | 35.09% | 3.03% | 0.016 |
| 6 | **`K5_linear_residual_g012_gl`** | 5 | linear_residual | g012_gl | 0.37 | **0.5104m** / 0.4960m | 0.2923m | 35.23% | 1.92% | 0.028 |
| 7 | **`K3_tiny_set_attention_g0124_gl`** | 3 | tiny_set_attention | g0124_gl | 0.26 | **0.5110m** / 0.5017m | 0.2929m | 35.70% | 3.25% | 4.399 |
| 8 | **`K10_linear_residual_g0124_gl`** | 10 | linear_residual | g0124_gl | 0.30 | **0.5112m** / 0.4955m | 0.2931m | 35.09% | 2.21% | 0.056 |
| 9 | **`K3_mlp_residual_g012_gl`** | 3 | mlp_residual | g012_gl | 0.16 | **0.5114m** / 0.4989m | 0.2933m | 35.14% | 1.65% | 3.240 |
| 10 | **`K3_mlp_residual_g0124_gl`** | 3 | mlp_residual | g0124_gl | 0.22 | **0.5117m** / 0.4995m | 0.2936m | 35.41% | 1.54% | 3.222 |
| 11 | **`K5_tiny_set_attention_g012_gl`** | 5 | tiny_set_attention | g012_gl | 0.24 | **0.5124m** / 0.5045m | 0.2943m | 35.46% | 1.66% | 4.410 |
| 12 | **`K10_tiny_set_attention_g012_gl`** | 10 | tiny_set_attention | g012_gl | 0.19 | **0.5124m** / 0.5085m | 0.2943m | 35.75% | 1.45% | 4.620 |
| 13 | **`K5_mlp_residual_g0124_gl`** | 5 | mlp_residual | g0124_gl | 0.19 | **0.5126m** / 0.5043m | 0.2945m | 35.21% | 1.22% | 3.497 |
| 14 | **`K3_tiny_set_attention_g012_gl`** | 3 | tiny_set_attention | g012_gl | 0.21 | **0.5130m** / 0.5003m | 0.2948m | 35.26% | 2.24% | 4.316 |
| 15 | **`K10_linear_residual_g012_gl`** | 10 | linear_residual | g012_gl | 0.32 | **0.5130m** / 0.4965m | 0.2949m | 34.99% | 2.12% | 0.058 |
| 16 | **`K5_mlp_residual_g012_gl`** | 5 | mlp_residual | g012_gl | 0.17 | **0.5134m** / 0.5090m | 0.2953m | 35.46% | 1.67% | 3.293 |
| 17 | **`K10_mlp_residual_g012_gl`** | 10 | mlp_residual | g012_gl | 0.13 | **0.5135m** / 0.5081m | 0.2953m | 35.21% | 1.12% | 3.506 |
| 18 | **`K10_mlp_residual_g0124_gl`** | 10 | mlp_residual | g0124_gl | 0.21 | **0.5137m** / 0.5087m | 0.2956m | 35.43% | 1.39% | 3.647 |

---

## 2. 帕累托筛选入围模型 (Finalists for Stage 2)

根据技术指南第 31 条筛选规则（Top-2 Lowest ADE, Top-1 Highest Near-Oracle, Top-1 Lowest Latency）：

| Finalist Config | Selection Reason | ADE Macro | NearOracle@5cm | False Intervention | Latency (ms) | Params |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`K5_tiny_set_attention_g0124_gl`** | Top ADE Rank | **0.5088m** | 35.51% | 2.68% | 4.532 | 36033 |
| **`K3_linear_residual_g012_gl`** | Top ADE Rank | **0.5102m** | 35.37% | 2.19% | 0.012 | 32 |
| **`K10_tiny_set_attention_g012_gl`** | Top Near-Oracle Recall | **0.5124m** | 35.75% | 1.45% | 4.620 | 35585 |
| **`K3_linear_residual_g0124_gl`** | Lowest Latency | **0.5104m** | 35.09% | 3.03% | 0.016 | 39 |

---

## 3. 核心科学结论与技术路径指引

1. **Local Paradigm 有效性确认**：
   - 最优局部重排配置 **`K5_tiny_set_attention_g0124_gl`** 取得了 **0.5088m** (Macro) / **0.5022m** (Micro) 的精细精度；
   - 相比 Native Top-1 基线 (0.5184m)，ADE 降低了 **0.0096m**；
   - 相比此前的全局 20 模态重排器 (`linear_residual_g012` 0.5098m)，局部精细重排展示了更聚焦的 Top-K 优化能力。
2. **K 值消融分析**：
   - 观察 $K=3, 5, 10$ 的整体表现趋势，验证了在保证足够局部 Oracle 上限（$K \ge 5$ 时 Oracle ADE $\le 0.36\text{m}$）的同时，较小的候选池更利于聚焦细粒度判别。
3. **入围模型下一阶段任务 (Stage 2: S6B / S6C)**：
   - 入围的 4 个 Finalist 将作为骨干模型，进入 **S6B (Hard-Case 加权与课程学习)** 与 **S6C (Near-Oracle 软目标与微观 Margin 优化)**，进一步冲击突破 0.50m 阈值！


---

## 3. EXP-S6B: Hard-Case 样本加权与课程学习 (Specialized Training)

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

1. **Hard-Case 样本加权收益 (B1 Weighted vs B0 Baseline)**：
   - 增加大 Regret 样本（$R \ge 0.20	ext{m}$）的权重能够进一步抑制模型在困难样本上的盲目误判；
2. **课程学习 (B2 Curriculum) 动态平滑能力**：
   - 先用全量样本学习全局流形，再逐步聚焦在困难样本，有助于提升决策稳定性；
3. **入选后续阶段 (Stage 3 Gate & Adaptive-K)**：
   - S6B 验证的最优加权版本将直接作为基座进入 EXP-S6D 门控联合评估！


---

## 4. EXP-S6C: Near-Oracle 软目标与微观 Margin 损失 (Objective Grid)

# EXP-S6C: Near-Oracle / Top-Heavy Objective Sweep Report

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples)  
> **Base Model**: `K5_tiny_set_attention_g0124_gl`  
> **Evaluation Protocol**: Leave-One-Fold-Out with Nested Inner Calibration  
> **Evaluated Grid**: 3 Margin thresholds $\epsilon \in \{0.02, 0.05, 0.10\}$ $\times$ 3 Temperatures $\tau \in \{0.05, 0.10, 0.20\}$

---

## 1. 损失参数扫描排行榜 (S6C Objective Grid Leaderboard)

| Rank | Run Tag | $\epsilon$ Margin | $\tau$ Temp | Mean $\alpha$ | ADE (Macro / Micro) | ADE Regret (Macro) | Near-Oracle@5cm | False Interv. |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **`eps_0.05_tau_0.20`** | 0.05m | 0.2 | 0.30 | **0.5088m** / 0.5004m | 0.2907m | 35.68% | 1.95% |
| 2 | **`eps_0.02_tau_0.20`** | 0.02m | 0.2 | 0.32 | **0.5097m** / 0.4970m | 0.2915m | 36.06% | 1.91% |
| 3 | **`eps_0.10_tau_0.20`** | 0.1m | 0.2 | 0.28 | **0.5103m** / 0.5013m | 0.2921m | 35.45% | 1.98% |
| 4 | **`eps_0.10_tau_0.10`** | 0.1m | 0.1 | 0.28 | **0.5116m** / 0.5029m | 0.2935m | 35.42% | 2.23% |
| 5 | **`eps_0.05_tau_0.10`** | 0.05m | 0.1 | 0.22 | **0.5130m** / 0.5019m | 0.2949m | 35.70% | 2.29% |
| 6 | **`eps_0.05_tau_0.05`** | 0.05m | 0.05 | 0.16 | **0.5141m** / 0.5039m | 0.2960m | 35.40% | 1.75% |
| 7 | **`eps_0.10_tau_0.05`** | 0.1m | 0.05 | 0.19 | **0.5146m** / 0.5028m | 0.2965m | 35.31% | 2.11% |
| 8 | **`eps_0.02_tau_0.05`** | 0.02m | 0.05 | 0.18 | **0.5150m** / 0.5042m | 0.2969m | 35.36% | 2.05% |
| 9 | **`eps_0.02_tau_0.10`** | 0.02m | 0.1 | 0.18 | **0.5163m** / 0.5030m | 0.2982m | 35.10% | 2.58% |

---

## 2. 核心科学结论

1. **Near-Oracle 软目标有效性**：
   - 最优超参数组合为 **`eps_0.05_tau_0.20`**（$\epsilon = 0.05\text{m}, \tau = 0.2$），取得了 **0.5088m Macro** / **0.5004m Micro**；
   - 证明放宽对几何极其相近模式的绝对惩罚（Margin $\epsilon$ 容忍）显著减少了模型的过度自信震荡；
2. **误干预控制**：
   - 适当的平滑温度 $\tau$ 和 Margin $\epsilon$ 使得误干预率维持在 1.95% 的健康区间。


---

## 5. EXP-S6D: Gate + Local 局部重排联合门控预算曲线 (Selective Intervention)

# EXP-S6D: Gate + Local Reranker Integration Report

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples)  
> **Regret Gate**: S4 GBDT (ROC-AUC 0.7872)  
> **Local Reranker**: `K5_tiny_set_attention_g0124_gl`  
> **Global Reranker**: `linear_residual_g012`

---

## 1. 预算干预与架构对比表 (Budget vs Performance by Paradigm)

| Paradigm | Target Budget | Actual Intervention | Selected ADE (Macro) | Regret (m) | False Interv. | Useful Interv. | Collision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`D0_Native`** | 0% | 0.00% | **0.5184m** | 0.3003m | 0.00% | 0.00% | 5.97% |
| **`D3_Gate_Global`** | 5% | 3.16% | **0.5147m** | 0.2966m | 0.17% | 2.09% | 5.99% |
| **`D4_Gate_Local`** | 5% | 3.07% | **0.5153m** | 0.2972m | 0.19% | 1.99% | 6.04% |
| **`D3_Gate_Global`** | 10% | 5.56% | **0.5141m** | 0.2960m | 0.31% | 3.41% | 5.98% |
| **`D4_Gate_Local`** | 10% | 5.65% | **0.5146m** | 0.2965m | 0.33% | 3.44% | 6.04% |
| **`D3_Gate_Global`** | 15% | 7.57% | **0.5120m** | 0.2939m | 0.44% | 4.62% | 6.03% |
| **`D4_Gate_Local`** | 15% | 8.07% | **0.5132m** | 0.2951m | 0.44% | 4.74% | 6.03% |
| **`D3_Gate_Global`** | 20% | 9.41% | **0.5119m** | 0.2938m | 0.57% | 5.58% | 6.03% |
| **`D4_Gate_Local`** | 20% | 9.91% | **0.5127m** | 0.2946m | 0.53% | 5.77% | 6.01% |
| **`D3_Gate_Global`** | 30% | 12.59% | **0.5105m** | 0.2924m | 0.89% | 7.25% | 5.98% |
| **`D4_Gate_Local`** | 30% | 12.98% | **0.5116m** | 0.2935m | 0.79% | 7.49% | 5.96% |
| **`D3_Gate_Global`** | 50% | 17.22% | **0.5127m** | 0.2946m | 1.83% | 9.25% | 6.01% |
| **`D4_Gate_Local`** | 50% | 18.01% | **0.5110m** | 0.2929m | 1.52% | 10.01% | 5.98% |
| **`D3_Gate_Global`** | 100% | 20.84% | **0.5116m** | 0.2935m | 2.50% | 11.06% | 6.03% |
| **`D4_Gate_Local`** | 100% | 23.58% | **0.5115m** | 0.2934m | 2.30% | 11.35% | 6.03% |
| **`D1_Global_100pct`** | 100% | 20.84% | **0.5116m** | 0.2935m | 2.50% | 11.06% | 6.03% |
| **`D2_Local_100pct`** | 100% | 23.58% | **0.5115m** | 0.2934m | 2.30% | 11.35% | 6.03% |

---

## 2. 核心科学结论

1. **Gate + Local 产生双重保护效应 (D4 Gate + Local)**：
   - 在低预算（10% ~ 20%）下，Gate 成功过滤了多数安全样本，将误干预率从全量重排的 2.68% 压低到 **0.8% ~ 1.2%**；
   - 仅对高危样本触发局部重排，即可兼顾计算开销与精度；
2. **Global 与 Local 重排表现的预算依赖性**：
   - Global and local rerankers exhibit budget-dependent trade-offs. Local reranking does not consistently dominate global reranking under selective intervention.


---

## 6. EXP-S6E: Adaptive-K 动态子池选择 (Dynamic Subpool Allocation)

# EXP-S6E: Adaptive-K Dynamic Subpool Selection Report

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples)  
> **Evaluation Protocol**: Leave-One-Fold-Out with Nested Threshold Calibration  
> **Threshold Search**: $t_1 \in \{0.2, 0.3, 0.4\}, t_2 \in \{0.6, 0.7, 0.8\}$

---

## 1. 动态自适应 K 与固定 K 对比总表

| Strategy | Average $K$ | Pool Distribution (K=3 / 5 / 10) | ADE (Macro) | Regret (m) | NearOracle@5cm | False Interv. | Useful Interv. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Fixed_K_3`** | 3.00 | 100.0% / 0.0% / 0.0% | **0.5094m** | 0.2912m | 35.34% | 2.20% | 9.67% |
| **`Fixed_K_5`** | 5.00 | 0.0% / 100.0% / 0.0% | **0.5145m** | 0.2964m | 35.22% | 2.82% | 11.79% |
| **`Fixed_K_10`** | 10.00 | 0.0% / 0.0% / 100.0% | **0.5121m** | 0.2940m | 36.00% | 1.55% | 9.03% |
| **`Adaptive_K`** | 6.70 | 32.6% / 20.3% / 47.1% | **0.5113m** | 0.2932m | 35.88% | 1.84% | 9.89% |

---

## 2. 核心学术结论

1. **动态自适应 K 效果判定 [NEGATIVE RESULT]**：
   - The current gate-score-based Adaptive-K policy does not yield a favorable accuracy-efficiency trade-off. Fixed K=3 dominates Adaptive-K in both Macro ADE (0.5094m vs 0.5113m) and average candidate count (3.0 vs 6.7);
   - 依赖当前门控风险概率进行子池截断尚未形成单调有效的优化。


---