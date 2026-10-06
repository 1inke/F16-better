# RGDM-S 轨迹筛选分支阶段性分析

状态：`SELECTION_ONLY_COMPLETE_FULL_FACTORIAL_PENDING`

本报告只分析已完成的 `s_only_g0_s1 / rgdm_b4_full`。生成分支 `g_only_g1_s0` 和联合分支 `joint_g1_s1` 尚未运行，因此这里不能给出生成主效应、交互效应或完整 2×2 结论。所有指标来自固定的 1,516 条 ETH scene-tail validation 样本；没有访问 test。

## 训练确认

- 从锁定的 C00 `best_selected.pth` 初始化。
- C00 主干冻结，只训练 44,744 个 RGDM-S 参数；模型总参数 391,014。
- 实际完成 12 个 epoch，耗时 344.32 秒（约 5.74 分钟）。
- gate/rank 辅助损失权重均为 0，本轮只检验结构本身。
- 候选轨迹保持不变；完整验证推理的最大数值误差为 `2.861e-6`，处于 CUDA 浮点等价容差 `5e-6` 内。

## 主要结果

| 视角 | Selected ADE | 相对 C00 | Selected FDE | 相对 C00 | Recall@1 | Recall@3 |
|---|---:|---:|---:|---:|---:|---:|
| C00 baseline | 0.538869 | — | 1.187295 | — | 20.185% | 36.082% |
| RGDM-S best-selection（epoch 2） | 0.536118 | -0.002751 (-0.51%) | 1.179417 | -0.007878 (-0.66%) | 20.251% | 36.807% |
| RGDM-S oracle-criterion checkpoint（epoch 1） | 0.539283 | +0.000415 (+0.08%) | 1.187472 | +0.000177 (+0.01%) | 20.119% | 36.346% |
| RGDM-S final（epoch 12） | 0.549658 | +0.010789 (+2.00%) | 1.204336 | +0.017042 (+1.44%) | 20.910% | 37.269% |

配对 bootstrap（10,000 次）的关键区间：

- best-selection 的 Selected ADE 差值 95% CI 为 `[-0.005886, +0.000199]`，跨 0；0.51% 改善不能判为可靠。
- best-selection 的 Selected FDE 差值 95% CI 为 `[-0.015255, -0.000898]`，有小幅稳定改善。
- final 的 Selected ADE 差值 95% CI 为 `[+0.004252, +0.017210]`，确认变差。
- final 的 Selected FDE 差值 95% CI 为 `[+0.001657, +0.032554]`，确认变差。
- Recall@1 和 Recall@3 的差值区间均跨 0，不能认定命中排序得到稳定提升。

## 候选与机制诊断

`minADE=0.219566`、`minFDE=0.396031`、endpoint spread、支持度和重复率在 C00 与 RGDM-S 间不变，符合“只改筛选、不改候选”的设计。改动只作用于 score，因此 Selected ADE 的变化也等于 regret 的变化。

best-selection checkpoint 的验证后运动分组显示：

| 分组 | 样本数 | C00 ADE | RGDM-S ADE | 差值 |
|---|---:|---:|---:|---:|
| stable | 1067 | 0.423465 | 0.424506 | +0.001041 |
| transition | 226 | 0.719847 | 0.715083 | -0.004764 |
| change | 223 | 0.907635 | 0.888782 | -0.018853 |

改善主要集中在少数 change 样本，stable 多数样本反而略差。该分组使用未来真值，只能用于验证后诊断，不能直接变成部署时的 gate。

历史校准参数在 best-selection checkpoint 为 `-0.004764`，对应平均历史置信系数约 `1.000071`：符号表示轻微强化而非预期的“遗忘/衰减”，且作用量几乎为零。主要变化来自 score residual。其平均绝对值从 epoch 1 的 `0.016875` 增至 final 的 `0.122419`，同时验证 ADE 在 epoch 2 后持续恶化，表现为明显的早期过拟合和 checkpoint 敏感性。

## 阶段性判定

预注册的选择侧推进门要求 Selected ADE 至少改善 1%，且 best/final 方向一致。当前最优只有 0.51%，区间跨 0；final 反而显著变差 2.00%。因此：

`RGDM-S_STRUCTURE_GATE_FAIL_AT_SEED1_VALIDATION`

这不是“机制完全无效”的普遍结论，而是说明当前 B4 结构、12-epoch continuation 和无辅助损失设置不具备稳定推进价值。不能只报告 epoch 2 的小幅改善，也不建议基于该结果直接扩种子或访问 test。

## 完整实验状态

- C00：ready（复用，不重训）。
- RGDM-S：ready（本报告）。
- RGDM-G：pending。
- RGDM-G + RGDM-S：pending。
- 完整 factorial analysis：pending。

批量脚本的输出目录识别问题已经修复。再次执行 `run_experiment.ps1 -Phase All -Device cuda:0` 会跳过已完成的 RGDM-S，只训练生成与联合两个缺失条件，然后进行完整分析。
