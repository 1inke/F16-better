# M1 execution — 显式动态历史配对实验

2026-10-14 人类已明确授权按设计包逐步实施。此 execution 是新生产实现，不修改父 DESIGN ONLY 快照；实际状态以 runtime_state.json 和唯一worker为准。

先源码/ID/统计/实际GPU/数值/两轮全optimizer-RNG恢复/保存审核/资源吞吐预检，再独立OS启动。当前尚未预检/启动，无生产ETA。两臂各3seed、固定25epoch，共150epoch/11100更新/24最终科学saved jobs。初始128记录另计。

继承 ../COMMON_PROTOCOL.md 与 ../M1_DYNAMIC_HISTORY.md；M1完成全saved审计分析后，按S1选择M2上游；M1阴性不取消M2独立结构问题。M3/V按原门，不自动凑满矩阵。禁止旧F16训练、旧失败G2桥续训、loss/宽度/epoch/seed搜索和旧PAUSED任务恢复。
