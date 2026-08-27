# EXP-052 Temporal Half Pool

由 F29 三种子学到的 landmark 权重反推：两个 landmarks 始终分别近似平均观测序列前四帧与后四帧。本实验删除可学习 landmark logits 和 einsum，直接执行无参数双段均值，再做标准 F16 QKV attention。目标是保留 8→2 的计算压缩，同时改善单样本常数开销。
