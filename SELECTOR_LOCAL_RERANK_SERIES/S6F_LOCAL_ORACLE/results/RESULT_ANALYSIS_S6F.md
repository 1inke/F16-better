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
   - 放宽至 **Top-3**，理论 Oracle ADE 降至 **0.4179m**；
   - 放宽至 **Top-5**，理论 Oracle ADE 进一步降至 **0.3625m**；
   - 放宽至 **Top-10**，理论 Oracle ADE 达到 **0.2862m**，距全局 Oracle (0.2181m) 差距缩小至 **+0.0681m**。

2. **核心决策判定 (Decision Checkpoint)**：
   - **子池上限客观评估**：Native Top-K pools preserve substantial oracle headroom. However, candidate truncation remains a meaningful ceiling, especially for K=3 and K=5. Top-10 provides substantially stronger oracle coverage than Top-5.
   - 明确具体召回率：
     - Top-5 Exact Oracle Recall: **60.04%** (Macro) / **60.82%** (Micro)
     - Top-10 Exact Oracle Recall: **81.45%** (Macro) / **85.24%** (Micro)
   - 局部重排具备充足理论探索空间，但必须正视 Top-K 截断引入的客观上限约束。
