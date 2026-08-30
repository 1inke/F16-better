# EXP-159B Score-Guided Diversity Compression

- 输入：EXP-158B 五折 V2-host K40 候选缓存
- 最终候选数：严格 K=20
- 配额：`0+20, 2+18, 4+16, 6+14, 8+12`
- 目标：修复 EXP-159 纯几何最远点采样偏爱极端轨迹的问题
- score：在 K20/V2 候选族内部逐样本标准化，禁止直接跨族比较原始 logits
- diversity：候选整轨迹距离与终点距离，仅作为 score 的小权重补充
- GT：只在20条候选固定后计算指标，不参与选择
- 保存规则：只有 PASSED 或 CONDITIONAL 才保存推荐候选，REJECTED 不向 EXP-161 传递缓存

