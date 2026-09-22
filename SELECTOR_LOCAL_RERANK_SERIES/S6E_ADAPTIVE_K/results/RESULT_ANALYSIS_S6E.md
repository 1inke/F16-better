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

1. **动态子池机制 (Adaptive-K) 实际效果判定 [NEGATIVE RESULT]**：
   - The current gate-score-based Adaptive-K policy does not yield a favorable accuracy-efficiency trade-off. Fixed K=3 dominates Adaptive-K in both Macro ADE (0.5094m vs 0.5113m) and average candidate count (3.0 vs 6.7);
   - 依赖当前门控风险概率进行子池截断尚未形成单调有效的帕累托优化，暂时停止在该方向上过度调参。
