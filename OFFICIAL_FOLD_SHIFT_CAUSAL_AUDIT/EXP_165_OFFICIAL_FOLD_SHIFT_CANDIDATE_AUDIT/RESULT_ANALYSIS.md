# EXP-165 Official Fold Shift + Candidate Geometry Causal Audit 结果分析

## 最终判定

- Scale hypothesis：**NOT SUPPORTED**，不允许进入 EXP-166A。
- Density/lateral hypothesis：**NOT SUPPORTED**，不允许进入 EXP-167。
- 按预注册决策树，下一项应为 **EXP-168A Joint Hybrid Generator K20（10 structured + 10 free）**。

本实验没有训练新模型。速度和密度分桶边界只由每个 outer fold 的训练源数据确定，held-out future 只用于离线候选误差诊断。

## 1. ETH 并不是简单的 anchor scale mismatch

ETH 相对训练源数据的主要 observation shift 为：

| 特征 | SMD |
|---|---:|
| neighbor_count_r6 | -3.058 |
| valid_agent_count | -3.043 |
| neighbor_count_r4 | -2.594 |
| mean_acceleration | +1.780 |
| neighbor_count_r2 | -1.572 |
| max_acceleration | +1.523 |
| mean_step_speed | +0.658 |
| median_step_speed | +0.586 |
| last_step_speed | +0.609 |

速度确实偏高，但没有达到预注册的 `SMD >= 0.8`。更强的变化是邻居数量大幅下降以及加速度显著增加。因此 ETH 更符合“低密度 + 强运动变化”的联合偏移，而不是单一速度尺度偏移。

更关键的是候选误差机制与 scale hypothesis 相反：

| Source-train speed bin | K20 J | H3 J | V2 J | V2−K20 |
|---:|---:|---:|---:|---:|
| Q1 | 0.739 | 0.566 | 0.545 | -0.194 |
| Q2 | 0.819 | 0.551 | 0.568 | -0.251 |
| Q3 | 0.712 | 0.586 | 0.593 | -0.119 |
| Q4 | 1.259 | 0.849 | 0.853 | -0.406 |

最高速度段中所有生成器都更困难，但 K20 的 J 相对整体恶化 22.9%，V2 仅恶化 19.6%；V2 相对 K20 的优势反而在 Q4 最大。H3/V2 相对 K20 的 gap 不随速度单调恶化，H3→V2 correction 与速度的 Spearman 仅 0.171，也没有径向覆盖不足证据。

因此，简单按历史速度同比缩放 H3 anchor 和 V2 correction 没有因果依据。运行 EXP-166A 很可能破坏当前在高速段已经优于 K20 的结构化候选。

## 2. Univ 的问题不是横向覆盖不足

Univ 相对训练源数据确实是显著高密度场景：

| 特征 | SMD |
|---|---:|
| neighbor_count_r6 | +1.737 |
| valid_agent_count | +1.652 |
| neighbor_count_r4 | +1.637 |
| neighbor_count_r2 | +0.968 |
| mean_step_speed | -0.393 |

低速×高密度桶包含 2,225 个样本，统计量充足：

| Family | J | lateral std | longitudinal std |
|---|---:|---:|---:|
| K20 | 0.1200 | 0.909 | 1.147 |
| H3 | 0.1637 | 1.287 | 1.015 |
| V2 | 0.1534 | 1.174 | 0.985 |

H3 在目标桶中确实弱于 K20，V2也只能部分恢复；但目标桶的 `H3−K20 J = 0.0437`，小于 Univ 全体的 `0.0639`，退化并没有集中在该桶。更重要的是 H3 lateral spread 比 K20 大约 41.6%，并非横向候选不足。

因此 EXP-167 提议的 density-conditioned lateral widening 会进一步放大已经偏宽的横向候选，机制方向与数据相反。该实验不应训练。

## 3. 对当前候选生成的更新认识

1. H3/V2 在 ETH 高速样本上相对自由 K20 更有优势，说明结构化 anchor 并没有出现简单的高速半径不足。
2. H3 在 Univ 的问题不是缺少横向多样性，而是增加的多样性没有有效落在 GT 附近；问题更接近候选预算/生成机制匹配，而不是单轴宽度不足。
3. 场景条件偏移真实存在，但“某个条件发生偏移”不等于“沿该条件直接缩放候选”会有效。EXP-165成功阻止了两条缺少机制支持的训练线。

## 4. 下一步

按照指导文件的预注册顺序，EXP-166A 与 EXP-167 均未获准入，应转入 EXP-168A：共享 F16 encoder，在训练阶段直接联合优化 10 个 structured H3/V2 modes 与 10 个 free-query modes，输出始终为 K=20。

EXP-168A 必须记录两族 winner fraction、soft responsibility、梯度范数、族内/族间多样性，防止一条分支责任塌缩。第一轮只运行官方五折 seed 1，不搜索 12/8、14/6 等配额；只有达到宏平均 J 改善至少 2%、所有 fold 退化不超过 2%、ETH或Univ至少改善 4%，且两族责任均不低于 10%，才允许补 seeds 2/3。
