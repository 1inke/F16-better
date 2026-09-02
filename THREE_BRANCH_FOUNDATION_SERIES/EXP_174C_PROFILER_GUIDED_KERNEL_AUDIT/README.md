# EXP-174C：Profiler-Guided Kernel Audit

冻结 EXP-173 F16 + H3 + Joint-V2，使用真实 ETH 测试批次和
`torch.profiler` 定位 CPU、CUDA、显存、FLOPs、调用次数和小算子启动瓶颈。
本实验不训练、不修改模型权重。

默认审计 batch=1 和 batch=64，并输出 top operators、模块范围、算子类别、
Chrome trace 和决策摘要。只有某个模块占端到端设备时间超过 20%，才建议进行
针对性的 EXP-174D 结构优化。

