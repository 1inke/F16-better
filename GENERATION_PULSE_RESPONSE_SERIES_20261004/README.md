# 终点一致的候选响应：固定 vs 可学习时间位置/跨度

人类2026-10-04“继续实验”授权。仅新文件夹保存新增code/cache/log/checkpoint/results/失败；旧科学代码/模型/结果不修改。

先读DESIGN.md和protocol.json。zero=fixed Gaussian response center/span，mean_std=observation-conditioned learned center/span；两组同终点漂移+零和响应结构、同39590名义参数。整体原F16/decoder/scorer固定；原nativeK20/损失/数据/guards不变。不是GT事件标签训练或自回归/在线新观测。

流程：GPU preflight -> 双组真实2epoch EXACT resume -> 吞吐ETA -> start_independent.ps1唯一独立OS worker -> 单次50%TIME检查 -> 更新完成截止 -> 完整审计与分析后交还人类。未通过预检不能launch。

运行状态见runtime_state.json；日志和worker锁仅本folder。正式结束要求worker与results exit双SUCCESS、18policy SHA/explicit1516IDs finiteK20、全部保存指标/组/CI重算、6个完整80epochs、全旧冻结+source/oldSHA不变。诊断不增加正式计数；本新study experiment2，此前completed1/3。

