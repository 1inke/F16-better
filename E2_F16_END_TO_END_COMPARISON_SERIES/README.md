# EXP-173：E2 与 F16 端到端匹配比较

本系列仅比较 `f4_relation_tokens`（E2）和 `f16_temporal_social_cross`（F16）。二者使用相同的官方五折数据、inner validation、随机种子、增强、损失和两阶段 H3→V2 训练流程。

Stage A 训练 encoder+H3 proposal+original score；Stage B 从本变体自己的 Stage-A 最优检查点出发，加入相同初始化的 V2 bounded correction，并联合训练 encoder、H3、score 和 V2。E2 不继承 F16 的任何检查点。

第一轮只运行 seed 1，脚本自动输出 `all_runs.csv`、`per_fold_summary.csv`、`paired_comparison.csv`、`stage_attribution.csv` 和 `summary.json`，不会自动启动 seeds 2/3。

seed 1 判定为 Case B 后，使用 `run_exp173_seeds23.py` 只补 seeds 2/3。该入口会先确认 seed 1 的20个阶段结果完整，复用完全相同的训练配置，支持中断续跑，并在训练结束后重新生成 seeds 1/2/3 的统一结果表。不要为本轮单独修改解码器、score或训练超参数。

三种子正式结果：F16的五折macro J为0.397276，E2为0.406319，F16改善2.23%，并在11/15个同折同种子配对中获胜。按预注册规则，F16成为当前ETH/UCY主编码器候选；E2保留用于后续SDD公平验证。完整结论见实验目录中的 `RESULT_ANALYSIS.md`。
