# EXP-115 KDTA Trend + Event + Recent

针对 EXP-114 的 minADE/same-path FDE 短板，在 `[pair-conditioned early event, exact frame 6, exact frame 7]` 前增加一个无参数早期趋势 token：前 6 帧 temporal features 的均值。

最终 4 个 K/V token 为：

1. early trend mean；
2. pair-conditioned deformable early event；
3. exact frame 6；
4. exact frame 7。

这项适配不增加参数，目标是补回整条路径形状信息。相比 F16 的 8 个历史 token，score 元素减少 50%，attention 核心主要计算预计减少约 45%。

