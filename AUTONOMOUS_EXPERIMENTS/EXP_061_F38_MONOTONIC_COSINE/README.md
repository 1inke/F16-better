# EXP-061：F38 单调余弦训练

模型结构保持 F38 不变，只把 CosineAnnealingWarmRestarts 改成一次性 warmup + 单调余弦衰减。目标是验证 F38 在最佳 checkpoint 上的跨场景优势能否稳定保留到第 30 轮。

