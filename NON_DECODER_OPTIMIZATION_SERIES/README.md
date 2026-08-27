# 非解码器优化实验系列

统一锁定：F16 稀疏时序—社会编码主干、Kinematic K20 解码器、Soft Oracle Coverage 0.20。

实验协议：先在 `val_scene=1, seed=1` 做 30 epoch 筛选；有效方案再做三种子复验与跨场景验证。所有新增训练策略均不修改解码器。

## 第一阶段

- EXP_074：统一新基线（Oracle-aware checkpoint）
- EXP_075：平方根逆频率场景采样
- EXP_076：EXP_075 + EMA
- EXP_077：EXP_076 + 旋转、镜像、速度缩放增强
- EXP_078：EXP_076 + 速度/加速度一致性损失
- EXP_079：增强与运动一致性组合

后续阶段根据筛选结果继续建立多尺度历史编码和 TTC/DCA 风险图目录。
