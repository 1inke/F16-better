# Official ETH/UCY Final V2 Three-Seed Validation

本系列冻结方法定义为 `F16 + H3 + V2 + original score`，不加入 M2、verifier、原型解码器或 SDD。

默认只补跑 seeds 2/3。每个随机种子都会从 H3/F16 训练开始，再训练对应的 V2 修正器，避免不同种子共享 seed 1 底座。五折测试仅在最终 V2 checkpoint 上执行。

`run_final_v2_seeds23.py` 支持中断续跑：已有完整 `metrics.json` 的阶段会自动跳过。完成后输出原始逐次结果、逐折 mean/std 和逐种子宏平均；seed 1 从已完成的官方 V2 测试结果读取，不重新训练。
