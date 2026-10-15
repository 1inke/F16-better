# M2 执行注册：冻结目标与联合目标对照

注册时间：2026-10-14 香港。当前 **DESIGN_IMPLEMENTATION_PENDING**，不是运行状态；没有代码预检 PASS、生产 worker、PID 或生产 ETA。

M1已完整审计，S1 FAIL。依事前规则，M2三个seed全部使用各自原G1父模型；不挑M1 seed44，不改用M1-A，不按CAL选择。人类已批准按整个目录逐步实施。M1约40min预算例外仅M1有效，M2仍须实测规划≤30min。

依据 [原M2设计](../M2_JOINT_GOAL_PROCESS.md)、[共同协议](../COMMON_PROTOCOL.md)、[本次精确设计](DESIGN.md) 和 [机器协议](protocol.json)。本文件是预注册快照，之后状态以runtime和真实soleworker为准。

下一实施预检节点为2026-10-14 20:00 HK，一次性节点，不是生产预计结束时间。到点实际实现、核验来源/ID/数值/GPU/恢复/资源与吞吐。全部门通过才独立OS启动，然后给出实测ETA、一次半程与完成节点。超预算或实质科学/source/ID/资源阻碍则停止并报告，不自动继承M1例外。

6个新run，150epoch，24个最终科学保存job，另12个全量初始化P0 job；完整FIT保留。旧G1/旧F16/decoder/scorer/旧结果均不可改。
