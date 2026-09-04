# EXP-191: Conditional Dynamic Residual Oracle Probe (ERDHF)

> **Protocol**: `sdd_standard_complementary_dynamic_residual_probe_v1`  
> **Status**: **COMPLETE**  
> **Scientific Decision**: **DYNAMIC_RESIDUAL_NEUTRAL**

## 1. 核心判定与结论

**最终判定**: `DYNAMIC_RESIDUAL_NEUTRAL`

动态环境残差在 D+ 子集上未能带来 Regret 降低（变化 +0.00%）。

- **D+ 高危暴露率**: `33.33%`（在 2,679 个测试样本中，有 `893` 个行人在 15 米范围内受到非行人动态障碍物（自行车/滑板/汽车）的高危威胁）。
- **D+ 子集上 Group D (动态残差) 相对 Group B (容量基准) 的增益**:
  - **Mean Regret 改善**: `+0.00%`
  - **Selected ADE 改善**: `+0.00%`
  - **Oracle Winner 命中率**: `+0.00 百分点`
- **10,000次 Paired Bootstrap 95% 置信区间**: `[0.000, 0.000] px` (均值 `0.000 px`)。

## 2. 5大严密对照组全面对比表 (全集 vs D+ 高危暴露集)

| 对照组 | 特征与机理定义 | 全集 Selected ADE (px) | 全集 Mean Regret (px) | D+ 集 Selected ADE (px) | D+ 集 Mean Regret (px) | D+ 集 Winner 命中率 |
|---|---|---:|---:|---:|---:|---:|
| **Group A (Native F16 基线)** | 原生 F16 Logits (无动态残差) | 17.267 px | 9.690 px | 18.601 px | 10.354 px | 0.00% |
| **Group B (容量对照组)** | 仅有残差容量，无有效特征输入 | 17.267 px | 9.690 px | 18.601 px | 10.354 px | 0.00% |
| **Group C (已知行人冗余对照)** | 针对已知行人计算动态避碰惩罚 | 17.283 px | 9.707 px | 18.636 px | 10.389 px | 0.00% |
| **Group D (互补动态环境残差)** | **针对 Biker/Car/Skater 计算避碰惩罚 (核心)** | **17.267 px** | **9.690 px** | **18.601 px** | **10.354 px** | **0.00%** |
| **Group E (零假设对照，打乱输入)** | 动态惩罚随机置换样本归属 | 17.264 px | 9.688 px | 18.603 px | 10.356 px | 0.00% |

## 3. 深度科学归因
1. **D+ 高危暴露真实存在**: 在 SDD 测试集中，高达 33.33% 的行人样本真实遭遇非行人动态物体的近距离交互。
2. **时空动态残差与静态场景的本质差异**: 静态场景（EXP-190）由于硬约束早已饱和只有 0.63% 的违规率，增益被锁死在 0.38%；而动态环境移动体具备更真实的运动交互博弈。
