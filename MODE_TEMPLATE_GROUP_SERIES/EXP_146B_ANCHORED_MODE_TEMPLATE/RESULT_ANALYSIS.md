# EXP-146B 结果分析

锚点对齐离散运动模板在六组 split/seed 配对中均降低 minADE：

- General minADE/minFDE 改善约 0.96%/0.35%；
- Challenge minADE/minFDE 改善约 0.89%/0.09%；
- 同路径 FDE 改善约 0.69%/0.32%。

这证明线性插值确实限制了中间轨迹形状，但幅度不足 1%，属于次要瓶颈。由于模板残差在锚点处为零，终点自由度有限，minFDE 改善尤其弱。增加约 10.27% 参数后的收益不足以支撑继续扩大模板容量。

完整对比见上级目录 `RESULT_ANALYSIS.md`。
