# EXP-159C Learnable K40-to-K20 Selector

- 冻结模块：F16/V2 encoder、V2 decoder、K20 decoder
- 唯一训练模块：共享候选质量网络
- 输入：encoder context、观测运动特征、候选轨迹几何、终点、位移序列、族内标准化旧score、family ID
- 监督：`ADE + 0.25 * FDE` 构造的 listwise soft target
- 推理：K40 → learned score → Top-20；最终候选数严格为20
- 禁止：test-fold监督、scene/fold ID、GT推理输入、手工fold配额
- 结果：`summary.json`、`per_fold_summary.csv`、`RESULT_ANALYSIS.md`、折内scorer checkpoint

