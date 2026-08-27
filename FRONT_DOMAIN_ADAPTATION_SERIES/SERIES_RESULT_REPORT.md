# 前端域适配系列结果总结

## 核心结论

1. EXP-134 表明 K20 候选没有普遍塌缩；固定锚点和学习偏移均有效。主要困难是高速度、非线性轨迹下的 CV 中心偏移，以及固定尺度候选不能随运动尺度充分扩展。
2. EXP-135 的单种子结果中，Speed Scale 和 Dual Temporal 曾显示约 1% 的局部收益；Density Self Motion 失败。
3. EXP-136 三种子复验证明：Speed Scale 与 Dual Temporal 都没有稳定改善 minADE；minFDE 平均收益不足 1%，且不能在三个种子和两种开发划分中稳定保持。
4. Speed Scale 不增加参数，并在 Challenge Same-path FDE 上 3/3 个种子小幅改善，可保留为“尺度信息有用”的机制线索，但当前注入方式不足以晋级。
5. Dual Temporal 增加 10.17% 参数，只在 General Top-1 指标上稳定改善，在 Challenge Top-1 ADE 上 3/3 变差，当前形式停止。
6. Baseline 从 General 到 Challenge 的 minADE/minFDE 分别恶化约 67.0%/81.6%。当前真正短板是运动尺度与轨迹形状的分布迁移，而不是缺少一个普通的附加分支。

## 当前决策

- 正式模型暂时保持不变。
- 不组合 Speed Scale 与 Dual Temporal。
- 停止 Density Self Motion 和当前 Dual Temporal。
- 不使用官方五折测试继续筛选；只有冻结的新方案通过多种子开发验证后，才进入正式五折确认。

## 下一阶段

按单变量实验依次验证：

1. 运动形状—尺度解耦编码；
2. 高速度、高曲率、低密度困难样本重加权；
3. 保持 Top-K 不变的尺度条件关系偏置。

完整数值与逐种子判断见 `EXP_136_MULTISEED_REPLICATION/RESULT_ANALYSIS.md`。

## EXP-137 后续验证补充

在独立的 `FRONT_SCALE_GENERALIZATION_SERIES` 中，又以三种子、General/Challenge 双划分验证了形状—尺度解耦、运动困难样本重采样和尺度条件关系偏置。三者都没有缩小 Challenge 与 General 的 minADE 差距，因此均未晋级，也不进行组合。

最明确的新信息是：替换式 Shape–Scale 在 General Top-1 ADE/FDE 上 3/3 种子改善约 1.85%/2.23%，但 Challenge 全面反向；尺度关系偏置也只在 General 有轻微 FDE 信号。下一阶段应优先验证保留 Baseline 状态的零初始化尺度残差，以及仅训练期使用未来 GT 困难度的损失重权重，而不是继续增加社会注意力偏置。

## EXP-138 后续验证补充

EXP-138 以三种子和双开发划分验证了零初始化 Shape–Scale 残差、未来困难度整体损失重权重、更宽速度增强和曲率增强。后三者均失败并停止。零初始化残差是唯一保留候选：General minADE/minFDE 改善 0.17%/0.96%，Top-1 ADE/FDE 3/3 种子改善；Challenge minFDE、Same-path FDE 和 Top-1 也改善，但 minADE 平均仍恶化 0.27%，主要由 seed 2 的 1.17% 恶化造成。

因此正式模型仍不变，下一阶段只对该残差的 0.10/0.15 幅度与 residual-norm 约束做校准；不再与失败的损失重权重或增强方案组合。
