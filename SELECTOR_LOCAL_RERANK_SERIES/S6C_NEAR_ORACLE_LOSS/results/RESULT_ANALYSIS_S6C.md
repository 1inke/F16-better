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

1. **Near-Oracle 软目标有效性与边界**：
   - 最优超参数组合为 **`eps_0.05_tau_0.20`**（$\epsilon = 0.05\text{m}, \tau = 0.2$），取得了 **0.5088m Macro** / **0.5004m Micro**；
   - 相比 S6A baseline 并未实现 Macro ADE 的实质性突破：Near-oracle-aware supervision remains competitive and can alter intervention behavior, but does not improve the current Macro-ADE frontier.
2. **误干预控制**：
   - 适当的平滑温度 $\tau$ 和 Margin $\epsilon$ 使得误干预率维持在 1.95% 的健康区间，在保持精度的同时提供了更平缓的决策边界。
