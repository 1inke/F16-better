# EXP-116 Pair-Conditioned Soft Temporal Prior

EXP-113–115 表明硬采样能改善 Top-1，却会损害 Oracle 覆盖。本实验保留 F16 完整 8 帧 K/V，只将 pair-conditioned event position 转为时间高斯先验并加入 attention score。

- 事件位置初始为 3.0，可在 `[0,6]` 内学习；
- 高斯宽度固定为 1.5 帧；
- 每个 head 有一个可学习的有界先验强度，初始为 0.25；
- 原 QK、relation bias/value、稀疏 mask 全部保留。

这是对 KDTA 与完整 F16 的软融合，不再减少 attention token，但仍保持总参数低于 30 万。

