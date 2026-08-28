# PHYSICAL_ATTENTION_BIAS_SERIES

本系列用于重新验证论文编码阶段的核心假设：

> 在原 F16 稀疏 Top-K 交互结构不变的情况下，使用逐时刻、物理因子化的 attention bias，直接改变 agent × neighbor × time 的注意力分布。

这不是继续增加普通 attention 模块，而是验证“轻量物理先验能否补偿稀疏交互的信息损失”。

## 1. 与旧 F9/F45 的区别

旧 soft-risk bias 把 closing、distance、TTC+DCA 压成单个 scalar risk score，再用每个 head 一个强度系数加入 logits。

本系列不再把不同量纲的物理量直接线性混合，而是分别处理 distance、closing speed、TTC、DCA。每个通道使用 5 个固定 RBF basis，每个 attention head 只学习少量 basis coefficient。

对于 4-head F16，新 bias 仅增加 4×4×5=80 个参数。同时 bias 保留时间维，不再把 8 帧交互历史压成同一个 pair-level bias。

## 2. 三个实验

### PB1 — time bias add

实验名：pb1_time_bias_add

保留原 F16 的 pair-level relation bias、relation edge value 与 Top-K graph，额外加入 time-resolved physical bias。新增 bias 权重全部零初始化，因此初始行为严格接近原 F16。

这是第一优先级实验，用来回答：逐时刻物理偏置本身是否提供额外信息？

### PB2 — time bias replace

实验名：pb2_time_bias_replace

删除旧 pair-level relation bias，保留 relation edge value，改为只使用新的 time-resolved physical bias 控制 attention logits。

它回答：新 bias 能否替代原来较重、较隐式的 learned pair bias？

### PB3 — bias-only interaction

实验名：pb3_bias_only_interaction

进一步删除 relation encoder、relation time pooling、relation bias 与 relation edge value，只保留原 F16 content attention + 新 time-resolved physical bias。

这是最强的轻量化消融：不再用高维 edge token/message，仅靠低成本物理 bias 能否维持预测质量？

## 3. 运行顺序

先同步仓库，然后进入 PHYSICAL_ATTENTION_BIAS_SERIES。

第一步必须先跑：

    C:\coding_toobox\envs\astra2\python.exe smoke_check.py

正常应看到三个 [ok]。smoke test 会检查输出 shape、zero-init bias、bias 梯度是否非零以及梯度是否 finite。

如果 smoke test 不通过，不要开始正式训练。

第二步建议先跑 seed 1：

    C:\coding_toobox\envs\astra2\python.exe run_physical_bias_batch.py --seeds 1 --variants pb1_add pb2_replace pb3_bias_only

这会同时跑 General 和 Challenge 两个开发划分。

出现正信号后再跑三随机种子：

    C:\coding_toobox\envs\astra2\python.exe run_physical_bias_batch.py --seeds 1 2 3

## 4. 判定标准

本系列不能只按 minADE 是否大幅下降判断。

PB1 的核心成功条件：

1. General / Challenge 不出现明显相反趋势；
2. minADE20 / minFDE20 至少基本不劣于 F16；
3. 如果 overall 改善有限，后续必须按 interaction difficulty 分桶分析；
4. 新 bias 必须学出非零且有结构的 head/channel response。

PB2/PB3 的核心是 accuracy-efficiency Pareto：

- 若 PB2 精度接近 PB1，但参数/关系建模更简单，优先 PB2；
- 若 PB3 精度只轻微退化，但明显减少 edge encoder 参数和计算，它仍然可能是论文轻量化主模型；
- 若 PB3 同时保持或改善精度，则是本系列最强结果。

## 5. 这一轮先不要做的事情

不要同时改 decoder、Top-K、新图、relation type bias、新 temporal branch 或 Coverage weight。先把 physical attention bias 的因果作用验证清楚。

后续若 PB1/PB2 有稳定正信号，再单独进入 K={2,4,6,full} 的 accuracy-interaction-cost Pareto 实验。