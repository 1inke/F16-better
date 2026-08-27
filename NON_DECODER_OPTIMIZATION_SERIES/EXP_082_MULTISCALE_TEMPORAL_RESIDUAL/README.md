# EXP-082 Multi-scale Temporal Residual

在 F16 现有时序 Transformer 后增加低秩、多膨胀率深度卷积运动混合器。三条时间尺度由速度、平均速度、加速度和转向特征动态门控；社会稀疏注意力与 K20 解码器不变。

训练底座采用 EXP-080：原始场景分布、EMA、旋转/镜像、0.9–1.1 速度缩放和 Coverage 0.20。
