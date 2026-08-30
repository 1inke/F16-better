# EXP-160 Score–Trajectory Mismatch Audit

- 类型：冻结模型、零训练、seed 1 官方五折审计
- 对象：H3、V2、M2 的候选、原始 logits 与逐候选 ADE/FDE
- 主要问题：候选细化改善 oracle geometry 时，旧 score head 是否逐步失效
- 主要指标：Spearman、winner hit、Top-N recovery、regret、entropy、active modes
- 决策：运行结束后由 `RESULT_ANALYSIS.md` 给出 PASSED、CONDITIONAL 或 REJECTED

只有评分失配证据成立，才进入 EXP-161 frozen candidate listwise scorer。

