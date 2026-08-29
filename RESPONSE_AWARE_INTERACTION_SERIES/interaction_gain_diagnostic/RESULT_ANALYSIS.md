# RB1 Interaction Gain 离线诊断分析

## 最终结论

**RB1 没有在高交互或高响应样本中表现出稳定、足够大的 oracle 收益。按照预先规定的情况 C，本项目应终止 Response-Aware Attention Bias 方向，不补跑 seed 2/3。**

本结论只否定“将 observed response 作为 attention-logit bias”的当前实现，不否定交互建模。结果反而支持下一阶段将交互信息直接用于 Mode / Anchor 候选生成。

## 数据可靠性

- General：1,516 个严格配对样本；
- Challenge：3,032 个严格配对样本；
- B0/RB1 使用完全相同的验证索引；
- 重新推理的十项总体指标全部复现原 `metrics.json`；
- 最大绝对复现误差约为 `6e-8`；
- 本诊断没有重新训练，也没有根据结果事后选择 High 阈值；High Response 与 High Potential 均使用训练集 Q75。

## 总体结果

| 划分 | minADE20 | minFDE20 | same-path FDE | Top-1 ADE | Top-1 FDE |
|---|---:|---:|---:|---:|---:|
| General | +0.195% | -0.338% | -0.158% | -0.157% | -0.159% |
| Challenge | +0.216% | +0.037% | +0.086% | -0.106% | -0.044% |

负号表示 RB1 改善。总体信号仍然只有千分之一级，而且 oracle 指标跨划分不一致。

## 最关键的二维子集

### High Potential + High Response

| 划分 | 样本数 | minADE20 | minFDE20 | same-path FDE | ADE 胜率 | FDE 胜率 |
|---|---:|---:|---:|---:|---:|---:|
| General | 151 | +0.105% | -0.217% | +0.127% | 50.3% | 51.0% |
| Challenge | 617 | -0.191% | -0.716% | +0.096% | 51.5% | 54.0% |

Challenge 的 minFDE20 有 0.716% 改善，但仍远低于预设的 2%–3% 保留门槛；minADE20 仅改善 0.191%，same-path FDE 反而退化。General 中同一目标子集也没有复现该收益。

配对 bootstrap 95% 区间：

| 划分 | 指标 | 相对变化 | 95% 区间 |
|---|---|---:|---:|
| General | minADE20 | +0.105% | [-1.310%, +1.606%] |
| General | minFDE20 | -0.217% | [-2.362%, +2.014%] |
| Challenge | minADE20 | -0.191% | [-0.751%, +0.397%] |
| Challenge | minFDE20 | -0.716% | [-1.552%, +0.129%] |

所有区间均跨过零，无法排除这些微小变化来自样本波动。

## Response Gain Curve

理想结果应当是 response 越强，RB1 改善越大；实际结果不符合这一模式。

### 最高 Response 10%

| 划分 | minADE20 | minFDE20 | same-path FDE |
|---|---:|---:|---:|
| General | +0.891% | +1.475% | +1.390% |
| Challenge | +0.920% | -0.247% | -0.462% |

General 的三个 oracle 指标在最高响应样本中全部退化；Challenge 也出现 ADE/FDE 冲突。

逐样本响应强度与 RB1 误差变化的 Spearman 相关系数如下：

| 划分 | minADE20 delta | minFDE20 delta | same-path FDE delta |
|---|---:|---:|---:|
| General | +0.017 | +0.054 | +0.058 |
| Challenge | -0.020 | -0.031 | -0.051 |

绝对值均不超过 0.058，说明 response intensity 与 RB1 收益基本没有单调关系。

## 不同响应定义的最高十分位

| 特征 | General minADE/minFDE | Challenge minADE/minFDE |
|---|---:|---:|
| Response magnitude | +0.871% / +1.399% | +0.921% / -0.247% |
| Pair response | +0.762% / +1.500% | +1.310% / +0.449% |
| Lateral response | +0.349% / +1.584% | +1.057% / -0.017% |
| Opportunity-gated pair response | +1.051% / +1.494% | +0.915% / +0.716% |
| Potential score | +0.910% / +1.551% | -0.451% / -0.785% |

尤其值得注意：与 RB1 实际计算最接近的 opportunity-gated pair response，在两个划分的最高十分位中 minADE/minFDE 全部退化。这是当前机制无效的直接证据。

Potential score 的 Challenge D10 有轻微改善，但 General D10 同时明显退化，因此不能解释为可泛化的交互收益。

## 其他分桶中的孤立现象

General 的最近距离 `1.5–3.0 m` 桶中，minADE/minFDE 分别改善 2.81%/2.17%，但该桶只有 103 个样本；Challenge 的相同距离桶包含 429 个样本，却分别退化 0.47%/0.77%。这是典型的孤立分桶现象，不能用于支持 RB1。

其他 closing speed、TTC 和 neighbor count 分桶同样没有同时满足以下要求：

- General/Challenge 方向一致；
- minADE/minFDE 同时或至少不冲突；
- 随交互强度增强形成连续趋势；
- 改善幅度达到约 2%–3%。

## 机制判断

RB1 参数已经正常学习，但仅修改 attention logits 很难显著改变 20 条未来候选的覆盖范围。当前结果更符合以下解释：

1. 原 F16 relation encoder 已经隐式包含距离、相对速度和交互风险等相近信息；
2. observed response 是对历史的描述，但没有直接规定不同未来模式应如何发生位移；
3. attention 权重的小幅变化能够轻微改变共享表示和 Top-1 排序，却不足以系统性改变 mode template 与三锚点的位置；
4. 因此 oracle candidate quality 没有获得稳定提升。

## 停止判定

按实验开始前规定的条件：

- High-response subset 没有明确收益：是；
- High-potential + High-response subset 没有达到 2%–3% 且无稳定趋势：是；
- oracle 指标仍接近零改善：是。

因此执行情况 C：

> 终止 Response-Aware Attention Bias，不补 seed 2/3，不继续搜索 hidden dimension、response feature 组合、opportunity gate、bias magnitude、normalization、learning rate 或 RB2/RB3/RB4 attention-bias 变体。

## 下一研究方向

下一步应从“影响关注对象”转向“影响候选内容”：

```text
Sparse Interaction Encoder
        ↓
Interaction-conditioned Mode Representation
        ↓
Mode-specific Anchor Offset / Anchor Generation
        ↓
现有 Refinement 与候选评分
```

第一轮应保持三锚点、20 模式、细化器和训练/推理选择机制不变，只新增零初始化的 interaction-to-mode / interaction-to-anchor 条件支路。这样可以直接检验：交互信息是否必须进入候选生成，才能改善 minADE/minFDE。
