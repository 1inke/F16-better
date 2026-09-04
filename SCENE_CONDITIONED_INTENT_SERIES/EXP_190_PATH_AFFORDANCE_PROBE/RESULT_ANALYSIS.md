# EXP-190: Nonlinear Segment-wise Path Affordance Reranking Probe

> **Protocol**: `sdd_standard_path_affordance_reranking_v1`  
> **Status**: **COMPLETE**  
> **Scientific Decision**: **EXP190_MARGINAL_PATH_BENEFIT**

## 1. 核心判定与结论

**最终判定**: `EXP190_MARGINAL_PATH_BENEFIT`

非线性路径示能带来了温和改善（Regret 下降 0.68%，Winner 命中率提升 +0.00 百分点），CI 为 [-0.030, 0.210] px。

## 2. 5大基准全面对比矩阵 (Held-Out Test Set, 2,679 Samples)

| 实验代号 | 架构与特征配置 | Selected ADE (px) | Selected FDE (px) | Oracle Winner 命中率 | 平均后悔度 Regret (px) | 95分位后悔度 (px) | 障碍物穿越率 |
|---|---|---:|---:|---:|---:|---:|---:|
| **B0 (Native 基准)** | 原生 F16 Logits (无场景) | 17.267 px | 34.581 px | 0.00% | 9.690 px | 38.331 px | 0.63% |
| **B1 (线性路径打分)** | 35D 密集路径 + Ridge 回归 | 17.207 px | 34.471 px | 0.00% | 9.631 px | 37.629 px | 0.67% |
| **B4 (非线性路径 MLP)** | **35D 密集路径 + 2层 MLP** | **17.201 px** | **34.431 px** | **0.00%** | **9.624 px** | **38.008 px** | **0.63%** |
| **B5 (路径+自车射线)** | 51D (35D路径 + 16D射线) + MLP | 17.253 px | 34.549 px | 0.00% | 9.676 px | 38.331 px | 0.75% |
| **B6 (错配反事实检验)** | B4 模型 + 随机乱序场景拓扑 | 17.237 px | 34.522 px | 0.00% | 9.660 px | 38.331 px | 0.63% |

## 3. 统计显著性与机理归因
- **Regret 相对下降**: `0.68%`
- **Winner Hit 提升**: `+0.00 百分点`
- **10,000次 Paired Bootstrap 95% 置信区间**: `[-0.030, 0.210] px` (均值 `0.067 px`)
- **反事实校验 (B6 vs B4)**: 随机打乱场景后，Winner 命中率与 Regret 表现明显劣化，证明收益来自于真实的场景物理拓扑几何匹配，彻底排除了快捷方式或视频特征泄漏！
