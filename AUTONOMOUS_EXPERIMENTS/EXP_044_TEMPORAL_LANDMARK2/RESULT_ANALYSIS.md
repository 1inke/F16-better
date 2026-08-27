# EXP-044 结果分析

| 参数量 | Original ADE/FDE | Fused ADE/FDE | minADE20/minFDE20 | 30-epoch 时间 |
|---:|---:|---:|---:|---:|
| 284,658 | 0.5034 / 0.8721 | 0.4403 / 0.7763 | 0.1615 / 0.2444 | 199.6 s |

通过首轮并进入多种子。相对 F16 seed1，Original ADE 退化约 1.0%，Fused ADE 退化约 2.7%，但 Original/Fused FDE 分别改善约 0.7%/0.9%。它把每个邻居 8 个历史 K/V 压成两个可学习时间 landmark，temporal-social 的 K/V 投影和 attention 时间轴理论减少 75%。

最初的顺序 CUDA 测试曾显示 batch=64 加速约 17%，但四模型 18 轮交替基准未复现：F16/F29 为 13.20/13.40 ms，差异落在运行波动内。可靠收益是峰值前向显存从 58.44 MiB 降到 54.58 MiB（约 6.6%）以及 attention 核心理论规模下降。当前定位是“计算规模/显存优先候选”，需 EXP-050 多种子确认。
