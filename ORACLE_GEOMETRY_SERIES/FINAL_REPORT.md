# ETH/UCY Oracle Candidate Geometry Series 最终报告

## 最终结论

本系列找到一个可复现的 Oracle 改进：

> **F16 temporal-social encoder + 原始 kinematic K20 + soft Oracle coverage loss，weight=0.20，temperature=0.10。**

该方案只修改训练目标，不改变推理网络、K值、运动学锚点、候选评分结构、参数量和推理计算量。模型仍为284,642参数。

## 三随机种子结果（scene=1）

| 模型 | mean minADE20 | mean same-path FDE | mean minFDE20 | mean Top-1 ADE/FDE |
|---|---:|---:|---:|---:|
| F16 K20 | 0.1641 | 0.3105 | 0.2616 | **0.4948 / 0.8770** |
| Coverage 0.15 | 0.1596 | 0.2952 | 0.2403 | 0.5491 / 0.9413 |
| Coverage 0.20 | **0.1568** | **0.2864** | **0.2335** | 0.5198 / 0.9121 |

Coverage 0.20 相对 F16 的 Oracle 改善：

- minADE20：约 4.4%；
- same-path FDE：约 7.8%；
- minFDE20：约 10.7%。

Top-1平均仍退化约4%–5%，说明评分器尚未适应新的候选分布。这不影响候选覆盖改善的结论，但意味着暂时不能直接将其作为最终Top-1模型。

## seed=1 五场景验证

| scene | F16 minADE/minFDE | Coverage 0.20 minADE/minFDE | 结果 |
|---:|---:|---:|---|
| 0 | **0.5466 / 1.0828** | 0.5609 / 1.1114 | 退化 |
| 1 | 0.1634 / 0.2619 | **0.1450 / 0.2217** | 改善 |
| 2 | 0.3057 / 0.5670 | **0.2929 / 0.5573** | 改善 |
| 3 | 0.2033 / 0.3701 | **0.1994 / 0.3583** | 改善 |
| 4 | 0.1718 / 0.3191 | **0.1706 / 0.3003** | 改善 |

按9,874个样本加权：

| 模型 | minADE20 | same-path FDE | minFDE20 | Top-1 ADE/FDE |
|---|---:|---:|---:|---:|
| F16 | 0.1996 | 0.4444 | 0.3658 | **0.4737 / 0.9802** |
| Coverage 0.20 | **0.1956** | **0.4298** | **0.3495** | 0.5037 / 1.0181 |

加权 Oracle 改善约2.0%、3.3%、4.5%。方法在4/5场景有效，但困难scene=0退化约2.6%，仍存在场景依赖。

## 几何结构实验

| 实验 | 参数 | seed=1 minADE20 | same-path FDE | minFDE20 | 结论 |
|---|---:|---:|---:|---:|---|
| F16 K20 | 284,642 | **0.1634** | 0.3151 | 0.2619 | 基线 |
| 条件锚点旋转 | 284,707 | 0.1698 | 0.3116 | 0.2594 | minADE退化 |
| 中间控制点 | 284,772 | 0.1663 | 0.3487 | 0.2650 | 整体退化 |
| 可学习基底 | 284,690 | 0.1655 | **0.3031** | **0.2562** | 单种子有信号，多种子失败 |

几何自由度越高并没有越好。固定运动学基底本身具有稳定正则作用；当前更有效的方式是改变候选训练责任分配，而不是增加解码自由参数。

## 覆盖损失定义

对每个候选计算：

`quality_k = ADE_k + 0.25 * FDE_k`

以停止梯度的责任权重：

`r_k = softmax(-quality_k / 0.10)`

增加训练项：

`L_coverage = 0.20 * sum_k(r_k * quality_k)`

原winner regression、soft/hard classification、endpoint、short-horizon和diversity loss全部保留。

## 被否决的补充实验

- coverage weight 0.05/0.10：Oracle收益低于0.20，Top-1也未更稳定；
- coverage 0.15前5轮渐进启用：没有超过直接0.20；
- 逐样本质量尺度归一化：scene=0和scene=1均退化。

## 当前模型决策

1. **Oracle研究候选**：F16 + K20 + coverage 0.20/temperature 0.10。
2. **稳定Top-1基线**：仍保留原F16 + K20。
3. 新方案推理参数量、FLOPs和延迟与F16完全相同。
4. 下一阶段应固定coverage 0.20，重新训练/校准候选评分器，重点缩小Top-1与Oracle之间的差距。
5. 在宣称ETH/UCY竞争力前，还需要运行标准五数据集/五折正式测试，而不能只依赖Univ训练集内部留场验证。

## 复现入口

- 首轮几何实验：`run_geometry_batch.py`
- coverage强度与warmup：`run_coverage_tuning.py`
- weight 0.20三种子：`EXP_070_COVERAGE_WEIGHT_SWEEP/run_w0p20_replication.py`
- 五场景验证：`EXP_072_COVERAGE_CROSS_SCENE/run.py`
