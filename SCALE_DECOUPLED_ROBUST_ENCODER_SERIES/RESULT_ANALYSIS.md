# SDRE1正式结果分析

## 1. 结论

SDRE1未通过seed1晋级标准，不进入seed2/3，也不进行SDRE2–4消融。

它在General上带来小幅FDE改善，但Challenge的两个主要oracle指标和same-path FDE同时退化。DSA提出的“尺度泄漏”是一个真实诊断现象，但本轮raw + normalized residual解耦并没有改善Challenge候选覆盖。

## 2. B0配对结果

| Split | Model | minADE20 | minFDE20 | same-path FDE | Top1 ADE | Top1 FDE |
|---|---|---:|---:|---:|---:|---:|
| General | B0 | 0.17952 | 0.31769 | 0.38101 | 0.46989 | 1.03276 |
| General | SDRE1 | 0.17956 | **0.31560** | **0.37957** | 0.47073 | 1.03401 |
| Challenge | B0 | **0.27874** | **0.48038** | **0.57057** | 0.75200 | 1.62268 |
| Challenge | SDRE1 | 0.27990 | 0.48106 | 0.57307 | **0.74979** | **1.62000** |

相对B0：

| Split | minADE20 | minFDE20 | same-path FDE |
|---|---:|---:|---:|
| General | +0.025% | **−0.659%** | **−0.378%** |
| Challenge | +0.417% | +0.142% | +0.438% |

负值为改善。General的FDE有小幅正面信号，但Challenge没有达到“minADE或minFDE改善≥0.5%”的预注册要求，且三个oracle相关指标同向退化。

## 3. 训练过程

General的SDRE1最佳oracle checkpoint出现在epoch 3；Challenge最佳checkpoint出现在epoch 6。之后没有出现向更好oracle覆盖恢复的趋势。

Challenge的Top1 ADE/FDE略有下降，而oracle minADE/minFDE变差。这表示SDRE1可能轻微改变了高分候选的排序/集中性，但没有扩展正确未来的候选覆盖；当前论文主目标是minADE/minFDE，因此不能将Top1微小变化视为成功。

## 4. 机制是否真正激活？

是。新增分支并非因为zero-init而始终失效：

- General与Challenge中motion-shape adapter及normalized relation encoder的末层权重均已从零更新；
- relation/shape gate的最终bias约为−1.95，对应平均门控强度约0.12；
- 参数量从355,048增加到360,691，仅增加5,643（约1.59%）。

因此失败的解释不是“模块没有获得梯度”，而是当前归一化坐标、局部距离回退和条件门控的组合，未能把Challenge的高速低密度偏移转化为更好的多模态轨迹覆盖。

## 5. 研究决策

1. 停止SDRE1链，不跑多种子和SDRE2–4。
2. 不继续叠加同类的scale-normalized residual、额外gate或更复杂的域适配损失。
3. 保留现有F16 + K20 + 三锚点decoder作为当前轻量模型；SDRE1可作为“编码器尺度解耦未改善Challenge oracle coverage”的负实验边界。
4. 后续若仍想改善Challenge，应优先从训练分布/数据覆盖而不是再加encoder小模块切入，例如speed×density平衡采样或更接近目标数据集的验证；这属于新的训练分布实验，不应伪装为SDRE1的局部消融。
