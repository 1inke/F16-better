# EXP-158 Candidate Family Complementarity Audit

- 类型：冻结模型、零训练、seed 1 官方五折审计
- 对象：matched K20、H3、V2、M2
- 主要问题：K20 与 anchor family 是否覆盖不同的未来区域，尤其能否修复 Univ 的候选缺失
- 正式候选数：各 family 均为 K=20；K40 union 只作诊断上界
- 决策：运行结束后由 `RESULT_ANALYSIS.md` 给出 PASSED、CONDITIONAL 或 REJECTED

只有 PASSED 才直接进入 EXP-159 fixed-K20 mixture。

