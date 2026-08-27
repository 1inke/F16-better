# ETH/UCY Oracle Interaction Series

本系列固定 ASTRA 的 Q20 解码器、损失与候选评分逻辑，只修改前端社会交互编码，目标是优先降低 `minADE20/minFDE20`。

统一基线：`f16_temporal_social_cross`。首轮口径：Univ、`seed=1`、`val_scene=1`、30 epochs；通过后再做多随机种子和跨场景验证。

- `EXP_062_DUAL_SCALE_SPARSE`：保留原局部稀疏图，额外加入最多两个远场潜在冲突邻居。
- `EXP_063_RELATION_TYPE_BIAS`：加入跟随、对向、横穿、接近风险等显式软关系偏置。
- `EXP_064_SECOND_ORDER_CONTEXT`：在一次主注意力前为源智能体注入稀疏邻域摘要，获得二阶上下文。
- `EXP_065_TEMPORAL_RISK_BIAS`：将复杂关系类型收缩为单一有界运动学冲突偏置。

运行入口：`run_oracle_interaction_batch.py`。
