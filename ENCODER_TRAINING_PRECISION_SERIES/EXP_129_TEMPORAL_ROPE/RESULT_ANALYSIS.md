# EXP-129 Temporal RoPE 结果

## 结论

不晋级。scene 1 的minADE改善1.58%，但same-path FDE和minFDE分别退化3.11%和2.84%；scene 3的三个oracle指标也全部退化。

| 场景 | minADE20 | same-path FDE | minFDE20 | Top-1 ADE/FDE |
|---|---:|---:|---:|---:|
| scene 1 | 0.132021 | 0.244835 | 0.217128 | 0.316706 / 0.621172 |
| scene 3 | 0.187732 | 0.433707 | 0.363222 | 0.470423 / 1.028979 |

加权minADE几乎不变（退化0.006%），same-path FDE和minFDE分别退化1.85%和1.08%；Top-1 ADE/FDE改善约2.82%/3.01%。参数量与基线完全相同。

RoPE同样表现为改变时间表示和候选排序，但没有改善终点覆盖。当前仅8帧历史，不值得继续进行频率、base或层数扫参。
