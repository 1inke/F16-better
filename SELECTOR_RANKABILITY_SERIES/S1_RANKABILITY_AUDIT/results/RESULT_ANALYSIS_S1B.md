# EXP-S1B: Oracle Margin & Ambiguity Audit Report

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples)  
> **Evaluation Seed**: 1

---

## 1. 核心实证结论 (Ambiguity & Sharpness Analysis)

- **Oracle 候选存在强模糊性 (Extreme Ambiguity)**：
  - **22.66%** 的样本中，第 1 候选与第 2 候选的误差差距 $\Delta_{1,2} < 2.0	ext{cm}$；
  - **37.70%** 的样本中，$\Delta_{1,2} < 5.0	ext{cm}$。
  - **科学结论**：强行训练网络去分类单一最佳索引（Argmin ADE）在物理上是病态的（Ill-posed），因为 Top-2 甚至 Top-3 候选之间的物理形态极其接近，存在严重的标签噪声与过度锐化问题。

- **Native Regret 双峰分布 (Bimodal Dilemma)**：
  - **31.86%** 的样本中，Native 自身就是 Near-Oracle（Regret < 5cm），盲目重排极易产生误干预；
  - 但在 **49.88%** 的样本中，Native 发生了严重误判（Regret > 20cm），这部分样本正是拉高全局 ADE 的根源。

---

## 2. 各折明细对比 (Per-Fold Ambiguity Table)

| Fold | Samples | Mean Oracle ADE (m) | Mean Native ADE (m) | Mean Regret (m) | Median $\Delta_{1,2}$ (m) | $P(\Delta_{1,2} < 2	ext{cm})$ | $P(	ext{Regret} < 5	ext{cm})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **eth** | 364 | 0.4086 | 0.9670 | 0.5584 | 0.1073 | 20.1% | 15.1% |
| **hotel** | 1197 | 0.1368 | 0.3067 | 0.1700 | 0.0263 | 46.4% | 55.2% |
| **univ** | 24334 | 0.2273 | 0.5587 | 0.3313 | 0.0914 | 17.1% | 26.7% |
| **zara01** | 2356 | 0.1816 | 0.4328 | 0.2512 | 0.0829 | 14.6% | 25.6% |
| **zara02** | 5910 | 0.1362 | 0.3268 | 0.1907 | 0.0280 | 44.1% | 52.0% |
