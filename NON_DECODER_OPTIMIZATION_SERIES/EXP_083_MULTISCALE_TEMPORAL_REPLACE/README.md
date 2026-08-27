# EXP-083 Multi-scale Temporal Replacement

用低秩、多膨胀率深度卷积运动混合器替换 F16 的单层时序 Transformer，保留动态时间查询和稀疏社会注意力。目标是在提升运动尺度适应性的同时显著减少参数与时间注意力计算。

训练底座与 EXP-080 相同。
