# Mode-Query and Distillation Series

本系列包含两条互不混合的实验链：

1. 精度链：固定 F16 与 Kinematic K20，逐步测试 anchor geometry、mode-specific temporal retrieval，并按门槛决定是否扩大到局部 agent×time retrieval。
2. 轻量链：先在最终协议下复验 F47，再按精度保护门槛决定是否进行 F16→F47 蒸馏。

统一首筛协议为 seed=1、scene 1/3、30 epochs、K20、Coverage 0.20、EMA 0.995、目标航向规范化、50% 镜像与 0.9–1.1 速度增强、oracle-combo 检查点。

最终状态见 `SERIES_RESULT_REPORT.md`：两条链均在条件门槛处停止，未把负收益结构继续扩大。
