# EXP-168B：Hybrid Family Attribution Audit

这是 EXP-168A 失败后的无训练诊断，不修改模型，也不使用 outer-test 选择超参数。

目标是把最终 K20 拆为：

- structured：模式 0–9；
- free：模式 10–19；
- union：全部 20 个模式。

分别统计每个 family 的 minADE、minFDE、same-path FDE、J、winner fraction、终点半径、横向/纵向覆盖、候选多样性和重复率，并使用 outer-train 定义的速度四分位与密度三分位分析 ETH 和 Univ 的场景差异。

本实验只用于解释 EXP-168A，不能根据 outer-test 分箱结果手工调整 10/10 配额后再次读取相同测试折。

