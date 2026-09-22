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

1. **Local Paradigm 与 Strong Global 对比评估**：
   - 最优局部重排配置 **`K5_tiny_set_attention_g0124_gl`** 取得了 **0.5088m** (Macro) / **0.5022m** (Micro) 的精细精度；
   - 相比 Native Top-1 基线 (0.5184m)，ADE 降低了 **0.0096m**；
   - **与强全局基准对齐**：Local reranking reaches performance comparable to strong global residual reranking, but has not yet demonstrated a consistent Macro-ADE advantage over the strongest global Transformer residual baseline (`transformer_residual_g0124` ADE = **0.5077m**)。
2. **K 值消融分析**：
   - 观察 $K=3, 5, 10$ 的整体表现趋势，验证了在保证一定局部 Oracle 上限的同时，较小的候选池能有效压低局部计算延迟，但受限于后验特征表现力，未发生量级突破。
3. **入围模型下一阶段任务 (Stage 2: S6B / S6C)**：
   - 入围的 4 个 Finalist 将作为骨干模型，进入 **S6B (Hard-Case 加权与课程学习)** 与 **S6C (Near-Oracle 软目标与微观 Margin 优化)**。
