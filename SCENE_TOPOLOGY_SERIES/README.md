# Scene Topology Series

本系列在冻结 `F16 + H3 + Joint-V2 + original score + K20` 的前提下，检验 SDD
参考图中的可通行拓扑是否能够补充当前 K20 缺失的未来方向。

## EXP-178A

- 轨迹模型零训练；只训练一个离线 9.8K 参数的 walkability CNN。
- CNN 仅使用标准 SDD **训练视频**的行人轨迹生成弱标签。
- 验证和测试视频的轨迹标签不会参与阈值选择、拓扑图或 proposal 构建。
- 测试 future 只在所有 deterministic proposals 冻结后用于 Oracle 指标计算。
- 新 proposal 沿 skeleton 可达分支生成，不使用 CV 轨迹作为几何模板。

通过门槛：总体 union minADE 或 minFDE 至少提升 5%，17 个测试视频中至少 12
个正向，并且 little 子集至少一个指标提升 8%。未通过则关闭 scene-topology 训练，
不执行 EXP-178B。
