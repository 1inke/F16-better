# CFB2 正式结果分析

## 1. 最终结论

CFB2 **未通过 seed1 晋级标准，不进入多种子实验**。

- 与实际初始化的 IR1 source baseline 严格配对后：
  - General minADE20/minFDE20 退化 **1.06% / 0.43%**；
  - Challenge 退化 **1.20% / 1.09%**。
- 与项目当前更强的 CSRS baseline 比较，General 退化 1.53%/1.59%，Challenge 退化 1.45%/1.62%。
- K=1/2/5 基本完全不变，CFB1 的中 K 收益没有保留。
- social-needed 子集在两个 split 上均退化。
- 梯度隔离本身成功；失败完全来自 mode 18/19 的候选替换方向不正确。

按预注册决策，本轮后应停止 retrieval-based pseudo-future regression，不继续 CFB2 权重、阈值或种子扫描。

## 2. 对照协议修正

初版汇总使用了 CSRS baseline 作 B0，但 CFB1/CFB2 实际从 IR1 baseline checkpoint 初始化。这两个 checkpoint 不是同一权重：

| Split | 对照 | minADE20 | minFDE20 | top-1 ADE |
|---|---|---:|---:|---:|
| General | CSRS 当前更强 B0 | 0.17868 | 0.31407 | 0.46739 |
| General | IR1 实际 source B0 | 0.17952 | 0.31769 | 0.46989 |
| Challenge | CSRS 当前更强 B0 | 0.27807 | 0.47784 | 0.75006 |
| Challenge | IR1 实际 source B0 | 0.27874 | 0.48038 | 0.75200 |

因此，方法因果效果必须与 IR1 source 比较；模型选型则仍应与更强的 CSRS baseline 比较。驱动脚本与诊断脚本已修正，后续不会再混用这两个对照。

这一修正也改变了对 CFB1 的局部解释：CFB1 相对它自己的 IR1 source 在 General 改善约 0.01%/0.25%，Challenge 改善约 0.12%/0.42%，但仍未超过项目当前 CSRS baseline，而且 social-needed 明显退化。因此 CFB1 有微弱因果信号，但没有模型选型价值。

## 3. CFB2 严格配对主结果

| Split | 模型 | minADE20 | minFDE20 | same-path FDE | top-1 ADE | top-1 FDE |
|---|---|---:|---:|---:|---:|---:|
| General | IR1 source B0 | 0.17952 | 0.31769 | 0.38101 | 0.46989 | 1.03276 |
| General | CFB2 | 0.18142 | 0.31906 | 0.38317 | 0.46989 | 1.03276 |
| General | 相对变化 | **+1.06%** | **+0.43%** | +0.57% | 约 0 | 0 |
| Challenge | IR1 source B0 | 0.27874 | 0.48038 | 0.57057 | 0.75200 | 1.62268 |
| Challenge | CFB2 | 0.28210 | 0.48559 | 0.57516 | 0.75200 | 1.62268 |
| Challenge | 相对变化 | **+1.20%** | **+1.09%** | +0.80% | 0 | 约 0 |

两个 split 方向一致，且所有主指标均退化，不是随机的单 split 波动。

## 4. 梯度隔离验证

| Split | mode 0–17 最大绝对偏移 | logits 最大偏移 | baseline top-1 平均轨迹偏移 | mode 18/19 平均偏移 |
|---|---:|---:|---:|---:|
| General | 0.000006 | 0.000005 | 0.00000018 | 0.2732 |
| Challenge | 0.000012 | 0.000007 | 0.00000035 | 0.2533 |

这证明 stop-gradient、冻结主干和关闭 oracle coverage 辅助项全部生效。CFB2 的失败不能归因于共享编码器污染；它是被适配的两个候选本身变差。

## 5. K 曲线

- General：K=1/2/5 与 source B0 数值相同，K=10 仅改善约 0.05%，K=20 退化 1.06%。
- Challenge：K=1/2/5 基本相同，K=10 变化约 0，K=20 退化 1.20%。

mode 18/19 大多位于评分尾部，所以修改它们没有影响前 5 个候选。CFB2 没有继承 CFB1 的 K=2/5/10 收益，只改变了 K=20 的尾部覆盖，并且改变方向是负面的。

## 6. social-needed

| Split | minADE20 相对变化 | minFDE20 相对变化 |
|---|---:|---:|
| General | +0.68% | +0.35% |
| Challenge | +1.38% | +1.71% |

置信度筛选没有增强社会交互分支。它仍主要刻画一般运动相似性，不能表示当前邻居对 focal future 的条件性影响。

## 7. 候选多样性与 tail coverage

- General 整轨迹多样性下降 **3.79%**；
- Challenge 下降 **3.25%**。

PFRA 选择高支持、高一致性的 cluster medoid，确实排除了极端噪声，但也把 mode 18/19 拉向高密度、已被其他模式覆盖的中心未来。CFB1 是过度扩散；CFB2 则过度中心化。

tail 11–20 成为 oracle winner 的比例略微增加：General 20.18%→20.78%，Challenge 34.23%→35.06%；但 tail 获胜时的平均 ADE 增益下降：General 0.3029→0.2845，Challenge 0.3861→0.3675。这表示新增的是更多“小幅、冗余胜出”，同时失去了原候选对少数稀有未来的“大幅覆盖收益”。

## 8. 0/1/2 pseudo 分组

| Split | pseudo 数 | 样本数 | minADE20 变化 | minFDE20 变化 |
|---|---:|---:|---:|---:|
| General | 0 | 561 | +0.07% | -0.64% |
| General | 1 | 495 | +1.91% | +1.54% |
| General | 2 | 460 | +2.03% | +1.07% |
| Challenge | 0 | 2009 | +1.03% | +0.77% |
| Challenge | 1 | 828 | +1.69% | +2.01% |
| Challenge | 2 | 195 | +1.55% | +1.27% |

真正使用 1–2 条高置信 pseudo 的样本退化最稳定，因此增加显式推理门控只能减少 0-pseudo 样本的误改，不能修复核心失败。没有必要为此再做 CFB2b。

## 9. PFRA 为什么通过，训练仍失败

PFRA 当前预测的是 pseudo future 的绝对 utility：

`pseudo 本身是否离 GT 较近`。

真正需要优化的是它对现有候选集的边际 utility：

`加入或替换该 pseudo 后，是否提供 B0 尚未覆盖的新方向`。

一条轨迹可以非常合理、离 GT 不远、邻居支持很高，但如果 B0 已有相似候选，它仍是冗余的。当前 PFRA 没有把“可靠性”和“相对 B0 的非冗余性”结合起来，这正是 oracle audit 与蒸馏收益之间的缺口。

## 10. 训练过程与成本

- 两个 split 的最优 checkpoint 都出现在 epoch 3，之后不再改善；增加 epoch 无法修复。
- 总参数从 355,048 增加到 360,428，即增加 5,380（约 1.52%）。
- 可训练参数仅 5,380，严格限制在 adapter。
- 两个 split 训练各约 3.5 分钟，训练成本低，但存在小量推理 adapter 开销。

## 11. 决策

1. CFB2 不跑 seed2/3。
2. 不扫描 confidence threshold、pseudo weight、学习率或 adapter scale。
3. 不追加仅解决推理门控的 CFB2b。
4. Future Bank 可保留一次“边际 utility”零训练审计作为研究解释，但不再进入模型训练链。
5. 下一条主要模型路线应转向无需外部伪未来的条件分布建模，例如 conditional distributional anchors、learned stochastic latent basis 或轻量 mixture likelihood，使 20 个候选共同学习覆盖分工，而不是用检索 medoid 替换已有稀有候选。

本轮最重要的结论是：**置信度可以判断一条检索未来是否合理，却不能判断它是否为当前候选集提供新的信息。对多模态轨迹预测，边际覆盖价值比单条轨迹可靠性更关键。**
