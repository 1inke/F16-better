# EXP-122 MQ-2 Temporal Mode Query

在 MQ-1 基础上，让每个 mode query 读取对应目标行人未池化的 8 帧 F16 token。使用共享 32 维 Q/K/V/O 投影、零初始化输出层与小残差门控，原时间均值 context 保持主路。

现有 K20 endpoint/path residual、Coverage、scorer 和稀疏交互均保持不变。正式结果未达到晋级条件，因此没有继续扩大到 MQ-3。
