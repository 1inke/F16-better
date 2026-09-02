# Candidate Uncertainty Series

## EXP-179A Candidate Perturbation-Stability Audit

本实验冻结 EXP-173 的 `F16 + H3 + Joint-V2` 五折 seed1 权重，以 mode identity
原位对齐的方式测试候选对轻微、物理合理观测扰动的敏感程度。

主扰动集合为 original、对称坐标噪声 P1+/P1-、最近两步速度 P3+/P3-；删除早期
观测 P2 仅作补充审计。扰动预测全部反变换到共同的全局坐标后再计算稳定性。

主门槛同时要求：

1. `trajectory_U` 对原评分控制后的真实残差质量达到五折 macro `|Spearman| >= 0.15`，且至少四折同号；
2. 高不确定 Q4 的真实质量比低不确定 Q1 至少差 10%；
3. 原评分 Top5 内真实 winner 的不确定性低于 non-winner，至少四折方向一致。

三项同时通过才进入 EXP-179B；否则关闭 uncertainty scorer。
