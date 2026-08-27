# EXP-113 KDTA Base

基础 KDTA 只改变 temporal-social attention 的 K/V 时间采样：每个目标—邻居对根据 `[target agent, source agent, relation]` 预测两个有界连续历史位置，并固定保留最后一帧。两个初始位置为 1.5 和 4.5，残差范围均为 ±1.5；初始 offset 输出为零。

其余保持 F16：缩放点积、运动学稀疏图、relation bias/value、Kinematic K20、Coverage 0.20、EMA、航向规范化和 50% 镜像/速度增强。

