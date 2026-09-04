# EXP-187A：端到端编码器贡献消融

本实验严格复用 EXP-173 的官方 ETH/UCY 五折协议、H3→fresh Joint-V2 两阶段训练和全部超参数，只补齐：

- E0：`f0_baseline`（trajectory-only）
- E1：`f3_factorized`（sparse/factorized social）

E2 与 F16 直接读取 EXP-173 的同协议结果，不重复训练。由于 E0/E1 参数量不同会消耗不同的模型初始化随机流，本实验将 H3 初始化放入独立固定随机上下文，保证两者的 H3 初始权重逐项相同；Joint-V2 同样使用 EXP-173 的独立固定初始化。第一阶段固定 seed 1；只有 E1 相对 E0 的五折 macro J 提升不少于 2%，且至少 4/5 折改善，才补 seeds 2/3。

正式入口：`run_exp187a.py`。脚本可中断续跑，已有 `metrics.json` 的阶段会自动跳过。
