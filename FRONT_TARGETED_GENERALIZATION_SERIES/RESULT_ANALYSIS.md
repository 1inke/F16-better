# EXP-138 针对性泛化系列结果分析

## 1. 实验完整性

- 24/24 次新训练完成：4 个单变量 × 2 种冻结开发划分 × 3 个种子。
- 汇总另含 6 次已有 Baseline，共 30 行结果。
- 所有比较均为相同种子配对；未访问官方 ETH/UCY 测试折。
- 以下误差指标均为越低越好。

## 2. 三种子均值

### General 开发划分

| 模型 | 参数量 | minADE20 | minFDE20 | Same-path FDE | Top-1 ADE | Top-1 FDE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 284,642 | 0.180862 | 0.331217 | 0.396436 | 0.461649 | 0.994950 |
| Residual Shape–Scale | 290,859 | **0.180547** | **0.328014** | 0.398866 | **0.451558** | **0.977990** |
| Future Difficulty Loss | 284,642 | 0.181958 | 0.332065 | 0.399775 | 0.462163 | 0.993497 |
| Wide Speed Augmentation | 284,642 | 0.181585 | 0.330225 | **0.395895** | 0.456112 | 0.987447 |
| Curvature Augmentation | 284,642 | 0.183359 | 0.334853 | 0.401650 | 0.466866 | 1.011467 |

配对百分比变化：

| 模型 | minADE | minFDE | Same-path FDE | Top-1 ADE | Top-1 FDE |
|---|---:|---:|---:|---:|---:|
| Residual Shape–Scale | **-0.17%** | **-0.96%** | +0.62% | **-2.18%** | **-1.70%** |
| Future Difficulty Loss | +0.61% | +0.26% | +0.84% | +0.12% | -0.14% |
| Wide Speed Augmentation | +0.40% | -0.29% | -0.14% | -1.19% | -0.74% |
| Curvature Augmentation | +1.38% | +1.11% | +1.32% | +1.14% | +1.67% |

### Motion-challenge 开发划分

| 模型 | 参数量 | minADE20 | minFDE20 | Same-path FDE | Top-1 ADE | Top-1 FDE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 284,642 | **0.302045** | 0.601339 | 0.707942 | 0.740569 | 1.587822 |
| Residual Shape–Scale | 290,859 | 0.302860 | **0.600018** | **0.705797** | **0.728551** | **1.572293** |
| Future Difficulty Loss | 284,642 | 0.304161 | 0.602200 | 0.712916 | 0.748169 | 1.594281 |
| Wide Speed Augmentation | 284,642 | 0.303682 | 0.602707 | 0.711416 | 0.736712 | 1.585341 |
| Curvature Augmentation | 284,642 | 0.302999 | 0.602729 | 0.709906 | 0.761188 | 1.635900 |

配对百分比变化：

| 模型 | minADE | minFDE | Same-path FDE | Top-1 ADE | Top-1 FDE |
|---|---:|---:|---:|---:|---:|
| Residual Shape–Scale | +0.27% | **-0.22%** | **-0.30%** | **-1.61%** | **-0.98%** |
| Future Difficulty Loss | +0.70% | +0.14% | +0.70% | +1.03% | +0.41% |
| Wide Speed Augmentation | +0.54% | +0.23% | +0.49% | -0.52% | -0.16% |
| Curvature Augmentation | +0.32% | +0.23% | +0.28% | +2.79% | +3.03% |

## 3. 唯一保留候选：Residual Shape–Scale

### 正向证据

- General minADE/minFDE 同时改善 0.17%/0.96%。
- General Top-1 ADE/FDE 改善 2.18%/1.70%，并且 3/3 个种子同向。
- Challenge minFDE、Same-path FDE、Top-1 ADE/FDE 分别改善 0.22%、0.30%、1.61%、0.98%。
- Challenge 上述四项均有 2/3 个种子改善。
- 与 EXP-137 的完全替换式 Shape–Scale 相比，保留 Baseline 状态的零初始化残差明显更稳，证明之前的信息损失判断成立。

### 未通过之处

- Challenge minADE 平均恶化 0.27%，只有 seed 1 和 seed 3 改善。
- seed 2 Challenge minADE 恶化 1.17%，说明残差对训练随机性较敏感。
- Challenge 的 oracle checkpoint 组合分数（minADE + 0.25×minFDE）仍比 Baseline 约差 0.11%，因此不能直接晋级。
- Same-path FDE 在 General 平均恶化 0.62%，表明部分模式的整条路径一致性被牺牲。

### 决策

将其标为 **唯一可继续校准的候选**，但当前 0.25 有界残差版本不替换正式模型，也暂不进入官方五折。下一步只针对残差幅度与正则化做小范围验证，不叠加其他失败模块。

## 4. 淘汰方案

### Future Difficulty Loss

- General 和 Challenge 的 minADE/minFDE 均变差。
- Challenge Same-path FDE、Top-1 也全部变差。
- 直接以未来困难度统一放大整个损失，会同时放大分类、回归、端点和多样性项，可能打乱原有损失平衡。

结论：当前统一重权重形式停止。若未来重访，应只对特定回归/coverage 项做归一化条件权重，而不能整体乘权。

### Wide Speed Augmentation

- General 的 Top-1 有改善，但 Challenge minADE 在 3/3 个种子均变差。
- Challenge minFDE 和 Same-path FDE 也整体变差。

结论：0.75–1.25 对当前数据过宽，停止；保留原 0.9–1.1 设置。

### Curvature Augmentation

- General minADE 在 3/3 个种子变差，平均 +1.38%。
- Challenge Top-1 ADE/FDE 在 3/3 个种子变差，平均 +2.79%/+3.03%。
- 当前几何扰动虽保持同帧相对距离，但生成的时间演化未必符合真实行人动力学，造成明显分布污染。

结论：直接停止，不降低幅度继续搜索。

## 5. 参数与运行记录

- Residual Shape–Scale：290,859 参数，比 Baseline 增加 6,217，即 +2.18%。
- 其余三个方案推理模型均保持 284,642 参数。
- Curvature General 的运行时间出现 918–2546 秒的大幅异常波动，而 Challenge 为 632–722 秒，说明运行期间存在外部负载、锁屏调度或系统状态变化。该组 elapsed time 不适合用于模型速度比较。
- 由于本轮目标是精度筛选，是否保留残差需在校准后再单独测推理延迟与 FLOPs。

## 6. 下一步建议

下一轮不再平铺新方向，只围绕 Residual Shape–Scale 做低成本校准：

1. 固定其余部分，比较残差幅度上限 0.10、0.15 与当前 0.25；
2. 增加轻量 residual-norm 正则，使残差只在确有收益的运动状态下偏离 Baseline；
3. 先用 General/Challenge 三种子筛选，目标门槛为两种划分的 minADE/minFDE 均不劣化，且 Challenge 至少 2/3 种子同向；
4. 只有通过该门槛后再进入正式五折验证。

本轮明确不建议把残差与 Future Difficulty Loss、Wide Speed 或 Curvature 组合。
