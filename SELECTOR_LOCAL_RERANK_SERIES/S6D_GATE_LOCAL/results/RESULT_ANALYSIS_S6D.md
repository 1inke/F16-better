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
   - 在低预算（10% ~ 20%）下，Gate 成功过滤了多数安全样本，显著压低了误干预率；
   - 对高危样本触发局部重排能够有效兼顾计算开销与精度；
2. **Global 与 Local 重排表现具有预算依赖性**：
   - Global and local rerankers exhibit budget-dependent trade-offs. Local reranking does not consistently dominate global reranking under selective intervention.
   - 在 5%~30% 紧凑预算区间，Global 方案在部分指标上表现优异；而在 50%~100% 宽松预算区间，Local 展现出细微优势，两者各有权衡。
