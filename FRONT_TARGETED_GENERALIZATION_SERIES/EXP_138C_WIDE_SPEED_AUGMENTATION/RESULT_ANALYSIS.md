# EXP-138C 结果分析

## 结论

淘汰，恢复并保持原 0.9–1.1 速度增强。

- General minADE/minFDE：+0.40%/-0.29%。
- Challenge minADE/minFDE：+0.54%/+0.23%。
- Challenge minADE 在 3/3 个种子变差。
- 虽然部分 Top-1 指标改善，但没有转化为 oracle 候选精度。

0.75–1.25 的范围对当前训练分布过宽，不再继续组合或微调。
