# EXP-S5-v2: Native Rank Position & Top-K Fine Discrimination Audit Report

> **Dataset**: ETH/UCY Official 5 Folds (34,161 Total Samples)  
> **Evaluation Protocol**: Oracle & Near-Oracle Position Analysis in Native Confidence Head (Reporting Primary Macro Average + Secondary Micro Average)

---

## 1. S5A: 真实最佳候选 (Oracle) 在 Native 置信度下的位次分布

| Top-K 范围 | Oracle Recall (Macro 主指标) | Oracle Recall (Micro 汇总) | 科学发现 |
| :---: | :---: | :---: | :--- |
| **Top-1** | **17.74%** | **19.07%** | 原生首选命中率较低，存在明显优化空间 |
| **Top-2** | 31.88% | 36.92% | 前 2 名包含率提升至 35%+ |
| **Top-3** | 46.15% | 46.53% | 前 3 名包含率接近半数 |
| **Top-5** | **60.04%** | **60.82%** | **Native Top-5 包含了多数 (Majority) 的真实最优候选** |
| **Top-10** | **81.45%** | **85.24%** | Top-10 提供了极高的最优解覆盖率 |

> **精准学术表述**：  
> Native Top-5 contains a majority of exact-oracle candidates (60.04% Macro / 60.82% Micro), while Top-10 provides substantially stronger coverage at 81.45% Macro (85.24% Micro).

---

## 2. S5B: 近最优候选集合 (Near-Oracle Set $O_\epsilon$) 召回率

| 候选窗口 | Near-Oracle @ 2cm (Macro / Micro) | Near-Oracle @ 5cm (Macro / Micro) | Near-Oracle @ 10cm (Macro / Micro) |
| :---: | :---: | :---: | :---: |
| **Top-1** | 30.57% / 27.53% | 34.92% / 31.86% | 41.37% / 38.26% |
| **Top-2** | 40.53% / 40.97% | 44.28% / 43.75% | 50.07% / 48.82% |
| **Top-3** | 48.93% / 48.47% | 52.16% / 51.14% | 57.58% / 55.94% |
| **Top-5** | 61.91% / 62.51% | **64.33% / 64.96%** | 68.21% / 68.99% |
| **Top-10** | 82.40% / 85.88% | 83.28% / 86.80% | 84.91% / 88.34% |

---

## 3. S5C: 全局排序 vs 局部 Top-K 细粒度排序难度对比 (The Top-K Difficulty Gap)

| 评估候选子集 | Pairwise@10cm (Macro / Micro) | Pairwise@5cm (Macro / Micro) | Spearman (Macro / Micro) |
| :--- | :---: | :---: | :---: |
| **Native Top-3** | 60.34% / 65.60% | 58.98% / 63.62% | 0.1690 / 0.3063 |
| **Native Top-5** | 64.27% / 67.44% | 64.28% / 66.48% | 0.3206 / 0.3608 |
| **Native Top-10** | 73.70% / 72.35% | 72.55% / 71.10% | 0.5039 / 0.4670 |
| **Native Top-20** | 83.93% / 82.97% | 82.70% / 81.77% | 0.7484 / 0.7350 |

---

## 4. 科学定性总结

1. **宏观可排序性完备**：全局 20 模态下成对判别准确率达到 **83.93% Macro**，Spearman 相关性达到 **0.7484**；
2. **微观判别力瓶颈**：在 Native Top-3 内，成对正确率滑落至 **60.34%**，相关性暴跌至 **0.1690**；
3. **架构演进结论**：Native 置信度具备优秀的粗排过滤能力（Top-5 召回 **60.04%** 最优模态），下一阶段应聚焦于 **Native Top-5 局部聚焦精细重排**。
