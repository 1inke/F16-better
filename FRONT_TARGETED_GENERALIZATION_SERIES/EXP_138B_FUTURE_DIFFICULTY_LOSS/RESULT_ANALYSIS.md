# EXP-138B 结果分析

## 结论

淘汰当前统一未来困难度重权重。

- General minADE/minFDE：+0.61%/+0.26%。
- Challenge minADE/minFDE：+0.70%/+0.14%。
- Challenge Same-path FDE、Top-1 ADE/FDE 也均变差。

将样本权重同时作用于分类、轨迹回归、端点、多样性和 coverage，破坏了原有损失项平衡。当前形式不再继续，也不与残差方案组合。
