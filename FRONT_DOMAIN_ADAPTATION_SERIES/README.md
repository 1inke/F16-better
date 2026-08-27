# 前端域适配实验系列

本系列固定当前主模型的 Kinematic K20 解码器、K=20、稀疏 Top-K 图、F16 时间—社会交叉注意力、Coverage 0.20、EMA、50% 反射/速度增强与目标航向规范化。

## 第一步：EXP-134 无训练诊断

读取 EXP-133 已有五折最佳 checkpoint，不重新训练。分解并输出：

- CV 终点误差；
- 固定 K20 锚点的 oracle 终点误差；
- 学习后候选的 minFDE/minADE；
- best-ADE 与 best-FDE 模式一致率；
- 候选终点两两距离、塌缩比例和学习偏移幅度；
- 高速度、高非线性、低密度及三者联合分组。

结果写入 `EXP_134_ENDPOINT_REACHABILITY_DIAGNOSTIC/results`。

## 第二步：EXP-135 非解码开发筛选

- `EXP_135_BASELINE_CONTROL`：当前 F16 对照；
- `EXP_135A_SPEED_SCALE_STATE`：逐智能体稳健速度尺度规范化，同时保留绝对 log-speed；
- `EXP_135B_TARGET_DUAL_TEMPORAL`：只为目标行人增加短期局部与全历史趋势双分支；
- `EXP_135C_DENSITY_SELF_MOTION`：低密度场景中增强目标自身运动残差。

每个方法分别在两种冻结开发划分上训练：

- `general`：每个来源场景内随机留出 5%；
- `challenge`：每个来源场景内按近期速度、运动非线性和低密度联合分数留出最高 10%。

本阶段只读取 `eth_data_train.npy`，不会把官方 ETH 测试集传给训练程序。正式五折留到方法冻结后的确认阶段。

