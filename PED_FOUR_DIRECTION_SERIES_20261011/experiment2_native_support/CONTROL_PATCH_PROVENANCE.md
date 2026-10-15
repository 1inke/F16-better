# 控制补丁来源记录
- 启动后仅更新控制runtime/.memory/根账本；锁定科学代码、protocol、模型、旧科学结果均未修改。
- 首次support_memory_ledger.patch遗漏.memory文件目标，匹配前拒绝，根账本与.memory仍旧值。
- 第二次support_memory_ledger_corrected.patch在根账本成功后，.memory的新增行hunk误以替换后文本为原文而拒绝。apply_patch跨文件可部分提交；不能声称这次零修改。
- 后续combined patch因账本已更新而拒绝。只读JSON核验确认真实RUNNING/PID33260/query00:56:11.200866/deadline01:03:41.200866正确。
- 最终support_memory_only.patch单文件合并原文hunk成功；.memory与账本恢复一致。失败及成功patch原样保留在允许暂存目录，不删除。
- 无第二次startup、worker查询或重启；不涉及科学gate放宽或SOURCE改hash。

