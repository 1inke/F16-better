# 对 Conditional Distributional Anchors 建议的评估

## 总体判断

该建议比继续检索式 pseudo future 更合理。它直接建模当前输入下的候选条件分布，不需要假设外部轨迹属于当前样本，也不会固定替换两个 tail mode。

## 与既有方法的关系

CDA1 并非完全未经尝试的新监督。当前 B0 已包含 soft oracle-coverage loss：它按整轨迹距离给多个候选分配 detached responsibility。CDA1 的新增部分是：

1. 在三锚点空间使用固定尺度规范化不同预测时刻；
2. posterior 同时依赖 anchor likelihood 和 mode prior；
3. mixture NLL 对 logits 提供概率校准梯度；
4. 远离 GT 的 mode 自然获得接近零的 responsibility。

因此它更可能改善候选概率与局部覆盖，而不是产生接近 10% 的几何跃升。

## 对原建议的修正

- 增加 matched continuation control，排除额外 12 epoch 带来的混淆；
- source checkpoint 固定为当前更强的 CSRS B0，避免再次出现 IR1/CSRS 对照错位；
- sigma 全部由训练折确定；validation 不用于尺度或 multiplier 选择；
- NLL 按 6 个 anchor coordinate 归一化后再乘 0.1，避免其数值尺度压过原 B0 损失；
- 第一轮不学习 variance、不扫描 lambda、不改变候选结构。

## 风险

- 单一 GT 对 mixture likelihood 的监督仍可能提高局部 mode responsibility，却无法凭空提供未观测的真实未来分支；
- 与现有 oracle-coverage loss存在部分重复，增益可能很小；
- fixed sigma 若过小会退化为 hard winner，过大则接近 MA1 式多 mode 同追 GT，因此必须报告 `N_eff`；
- 即使 NLL 和 calibration 改善，也不代表 minADE/FDE 一定改善。

若严格对照下 CDA1 失败，应按建议停止继续修改现有 20-mode/3-anchor decoder 的训练目标。
