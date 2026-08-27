# EXP-137A 结果分析

## 结论

不晋级；停止当前“替换式”实现，但保留尺度残差方向。

## 三种子结果

- General：minADE +0.09%，minFDE -0.53%，Top-1 ADE/FDE -1.85%/-2.23%。
- Challenge：minADE +0.42%，minFDE +0.30%，Top-1 ADE/FDE +1.19%/+0.16%。
- General Top-1 两项 3/3 种子改善，但该收益在 Challenge 中反向。
- 参数量 283,390，比 Baseline 少 0.44%；训练时间却增加约 2.3%–4.9%。

## 解释

分支能改善域内模式排序，但没有提高 oracle 覆盖或困难运动泛化。完全替换原始 robust state 可能丢失绝对物理运动信息。若继续，应使用保留 Baseline 状态、零初始化附加残差的形式。
