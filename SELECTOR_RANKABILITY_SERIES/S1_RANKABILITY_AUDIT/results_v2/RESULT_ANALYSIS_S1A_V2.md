# EXP-S1A-v2: Corrected Train vs Test Rankability Diagnostic Report

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples LOO)  
> **Evaluation Seed**: 1  
> **Validity Audit Status**: Aligned Train Evaluation, Corrected Pairwise Accuracy, Train-Only Scaling

---

## 1. 修正后的实证数据表 (Corrected Train vs Test Metrics)

| Method | Calibrated $\alpha$ | Train ADE (m) | Test ADE (m) | Train Regret (m) | Test Regret (m) | Train Pairwise@10 | Test Pairwise@10 | Test Spearman |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **linear_residual_g012** | 0.58 | 0.4743 | 0.5118 | 0.2737 | 0.2936 | 84.79% | 84.60% | 0.7579 |
| **mlp_residual_g0124** | 1.00 | 0.4635 | 0.5113 | 0.2629 | 0.2932 | 85.40% | 84.57% | 0.7559 |
| **transformer_residual_g0124** | 1.00 | 0.4628 | 0.5119 | 0.2622 | 0.2938 | 85.41% | 84.89% | 0.7591 |
| **mlp_pairwise_g0124** | 1.00 | 0.4602 | 0.5165 | 0.2596 | 0.2984 | 85.75% | 84.63% | 0.7590 |
| **mlp_listwise_g0124** | 1.00 | 0.4711 | 0.5511 | 0.2705 | 0.3330 | 82.64% | 80.04% | 0.6805 |

---

## 2. 核心科学观察与工作假设调整

1. **Pairwise 正向指标大幅修正**：
   在修正指标方向后，主流残差选择器的 Pairwise Accuracy 实际达到 **84%~85%**，Spearman 相关系数达到 **~0.75**！
   这表明：**候选集合在宏观上具有极强的整体可排序性（Global Ranking is Strong），绝非随机乱排！**
2. **Top-of-List Fine Discrimination 成为核心瓶颈**：
   虽然全局排序准确率超过 84%，但精确命中最优候选（Oracle Hit）依然只有约 17%~18%，Selected ADE 仍收敛在 0.50m~0.51m 附近。
   说明瓶颈主要集中在排名前列的优质候选之间（Top-K fine discrimination），这极力支撑了下一步推进 **EXP-S5（Native Rank Audit）** 与 **EXP-S6（Top-5 局部重排）** 的必要性。
