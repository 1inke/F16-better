# CSO1 实验结果分析

## 核心结论

CSO1-R 与 CSO1-P 均未达到晋级标准，**不进入 Seed2/3**。

- CSO1-R 几乎等价于 B0，但四个主要 oracle 结果在 General/Challenge 均没有净改善。原因不是实现失效，而是 B0 的候选本来就很少落入 0.4 repulsion margin，winner-protected repulsion 几乎没有可作用的对象。
- CSO1-P 在 Challenge 改善了 Top1、Same-path FDE 和极小幅 minFDE，但 General 的 minADE/minFDE 分别退化 0.49%/0.70%，候选 diversity 也进一步下降。Prototype 学成了稳定的 mode 均值，却没有提高候选集合覆盖。

本轮最重要的结论是：

> 当前问题不是候选之间“距离不够大”，而是候选是否覆盖了正确、合理的未来区域。无方向的 endpoint repulsion 和单均值 prototype 都不能提供这种信息。

## 1. Overall 指标

数值越低越好；变化率为负表示改善。

| Split | 模型 | minADE20 | minFDE20 | Same-path FDE | Top1 ADE | Top1 FDE |
|---|---:|---:|---:|---:|---:|---:|
| General | B0 | 0.17868 | 0.31407 | 0.37851 | 0.46739 | 1.02615 |
| General | CSO1-R | 0.17884 | 0.31501 | 0.38057 | 0.46788 | 1.02680 |
| General | CSO1-R 变化 | +0.092% | +0.300% | +0.543% | +0.106% | +0.063% |
| General | CSO1-P | 0.17955 | 0.31625 | 0.38131 | 0.46664 | 1.02591 |
| General | CSO1-P 变化 | **+0.491%** | **+0.695%** | +0.738% | -0.159% | -0.024% |
| Challenge | B0 | 0.27807 | 0.47784 | 0.56940 | 0.75006 | 1.61948 |
| Challenge | CSO1-R | 0.27844 | 0.47810 | 0.56811 | 0.75029 | 1.61921 |
| Challenge | CSO1-R 变化 | +0.131% | +0.054% | -0.226% | +0.031% | -0.017% |
| Challenge | CSO1-P | 0.27817 | 0.47759 | 0.56650 | 0.74741 | 1.61261 |
| Challenge | CSO1-P 变化 | +0.033% | -0.052% | -0.509% | -0.353% | -0.424% |

CSO1-R 的变化全部在约 ±0.5% 内，整体略偏退化。CSO1-P 的 Challenge 结果看似较好，但 General oracle 明显反向，未形成跨 split 复现。

## 2. 为什么 CSO1-R 几乎没有作用

| Split | B0 endpoint pair mean | P10 | 小于 0.4 margin 的 pairs |
|---|---:|---:|---:|
| General | 2.421 | 0.898 | **1.73%** |
| Challenge | 2.663 | 1.017 | **0.13%** |

当前候选终点平均距离远大于 0.4，甚至 P10 也达到 0.90–1.02。换言之，98.3%–99.9% 的候选对根本不会触发 margin loss。

CSO1-R 只把原来的 all-pair linear repulsion 换成 winner-protected squared-hinge repulsion；在有效 pair 极少的情况下，这个修改自然接近 no-op。它没有创造新的 coverage signal。

实际 diversity 反而轻微下降：

| Split | Endpoint | Trajectory | Anchor |
|---|---:|---:|---:|
| General CSO1-R | -0.275% | -0.316% | -0.298% |
| Challenge CSO1-R | -0.275% | -0.307% | -0.299% |

这也符合预期：排除 winner 并采用 squared hinge 后，总排斥信号比原 B0 更弱。

## 3. Prototype 的作用：改善局部选择，但收缩集合

CSO1-P 的 diversity 变化为：

| Split | Endpoint | Trajectory | Anchor |
|---|---:|---:|---:|
| General | -0.871% | -0.933% | -0.898% |
| Challenge | -0.413% | -0.464% | -0.432% |

Prototype 并未造成 MA1 那种 7%–13% 的严重收缩，但方向仍然是把同一 mode 的预测拉向跨样本均值。这解释了：

- Challenge Top1 ADE/FDE 改善 0.35%/0.42%；
- Challenge Same-path FDE 改善 0.51%；
- 但 General minADE/minFDE 退化 0.49%/0.70%。

它更像一个轻微的中心化/稳定化正则，对选择出的代表轨迹有帮助，却削弱了尾部候选覆盖。

Prototype 训练并未出现未覆盖问题：两个 split 的 20 个 modes 全部被初始化；General/Challenge 分别累计约 345k/245k winner 更新。因此失败不是 prototype 没学到，而是“每个 mode 一个 EMA 均值”不足以表示条件化、多峰的未来形状。

## 4. ADE/FDE@K 没有呈现集合优化应有的规律

### CSO1-R

- General ADE@2/@5 小幅改善 0.155%/0.148%，但 ADE@10/@20 又退化 0.116%/0.092%。
- Challenge 从 K=2 到 K=20 基本全部退化，ADE@10 退化 0.420%。

### CSO1-P

- General ADE@1 改善 0.159%，但 ADE@2/@5 退化，ADE@20 退化 0.491%。
- Challenge ADE@1 改善 0.353%，ADE@5 改善 0.061%，但 ADE@10 退化 0.148%，ADE@20 退化 0.033%。

如果 candidate-set coverage 真正改善，应当随着 K 增大逐步获得更多收益。本轮恰好相反：少量高分候选有时更好，但加入更多候选没有带来更好的 oracle coverage。

因此 CSO1-P 的收益主要来自评分/中心候选，而不是集合覆盖。

## 5. Anchor coverage

### CSO1-R

- General Step4/8/12/endpoint 分别退化 0.94%/0.46%/0.89%/0.30%。
- Challenge 仅 Step12 改善 0.02%，其余基本持平或退化。

### CSO1-P

- General Step4/8/12/endpoint 分别退化 0.64%/0.42%/0.29%/0.70%。
- Challenge Step4/8/12/endpoint 分别改善 0.03%/0.35%/0.13%/0.05%。

Prototype 在 Challenge 的三个锚点上方向一致但幅度很小，且 General 完全不复现，不能晋级。

## 6. Winner usage 保持稳定

- 所有模型在两个 split 上均为 0 个 never-win modes。
- General 始终有 19 个 modes 的 winner 占比超过 1%，Challenge 为 20 个。
- Winner entropy 两个 CSO 变体均与 B0 接近，并略有上升。

这说明实验没有造成离散 mode collapse。但 winner usage 稳定并不等价于 GT coverage 提升；本轮主要问题仍是每个 mode 覆盖的位置不够准确。

## 7. 训练曲线排除偶然 checkpoint

- CSO1-R 在 General 第 3/6/9/12 轮均没有超过 B0，Challenge 同样如此。
- CSO1-P 的 General 从第 3 轮开始就退化，之后只能部分恢复；Challenge 的小幅改善在第 9 轮最佳。
- 训练 loss 平稳，没有数值异常或优化失败。

结论不是由最终 checkpoint 偶然造成，也没有继续增加 epoch 的合理依据。

## 8. 晋级判定

| 标准 | CSO1-R | CSO1-P |
|---|---|---|
| minADE/minFDE 改善 ≥0.5% | 失败 | 失败 |
| ADE@10/@20 一致改善 | 失败 | 失败 |
| General/Challenge 同向 | 失败 | 失败 |
| Diversity 不下降 | 失败 | 失败 |
| Anchor/endpoint coverage 不下降 | 失败 | 失败 |
| Winner usage 不集中、无 dead modes | 通过 | 通过 |

综合结论：**两个变体都不进入 Seed2/3。**

## 9. 后续研究判断

不建议继续扫 margin 或提高 repulsion 权重。诊断显示候选已经足够分散，盲目扩大距离只会把候选推向未必合理的区域。这里必须区分：

> Diversity 是候选彼此不同；coverage 是候选接近可能发生的未来。前者增加不保证后者增加。

ETH/UCY 每个观测历史只有一条标注未来。仅凭单 GT，non-winner modes 没有信息判断“应该去哪里”，repulsion 只能告诉它们“不要靠近”，而不能告诉它们“哪些替代未来是合理的”。单一 EMA prototype 同样只能学习每个 mode 的平均形状。

若继续寻找较大提升，下一步需要的是 **有方向的多未来监督**，而不是更强 repulsion：

1. 从相似历史/相似交互状态构建 conditional future bank，给 non-winner candidates 提供伪多未来目标。
2. 离线聚类条件化 future shapes，学习多中心 mode prototypes，而不是每个 mode 一个全局 EMA 均值。
3. 使用 distributional decoder，例如轻量 mixture likelihood，让 mode 同时学习均值、概率和合理尺度。
4. 或重新设计 hierarchical candidate basis，让不同候选由可解释的方向/速度/转向原型生成。

CSO1 最终说明：当前 candidate family 并不缺简单的几何分散度，真正短板是 **non-winner candidates 缺少有语义、有方向的学习目标**。

## 数据来源

- `seed1_summary.json`：严格配对总体指标。
- `diagnostic/combined_summary.json`：ADE/FDE@K、pairwise distribution、diversity、coverage 和 winner usage。
- 各实验 `console.log`：验证曲线。
- Prototype checkpoint buffers：mode 覆盖与累计更新次数。
