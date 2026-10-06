# Temporal subspace — cycle6 experiment2

正式实验2026-10-02 16:49:17香港时间启动。Windows独立按需任务Trajectory-TemporalSubspace-Cycle6，Python PID19396、服务父PID2728；真实进程和Task Running及TRAIN zero seed1已确认，stderr空。关闭Codex不结束该OS任务；机器需保持登录、开机、不睡眠，后续自动分析仍需应用可用。

六个全新80轮模型，三配对seed。zero=完整最后时间层+新head99942可训练参数；mean_std=只训练该层self_attn+同新head62406。容量差异是干预的一部分，重复fullarm同seed不是独立泛化。双方全初始化exact/native初始输出exact；online9agent8step观察重新计算tokens，两组F16eval禁用dropout，两组实际attention梯度正；仅注意力组FFN/LayerNorm无梯度且exact冻结。其它F16、原decoder/scorer固定，nativeK20baseline不漂移，无GT未来group/logitsforward。原loss/protection.1/constrainedcal/fit25912cal2879/LRorder/velocitybound1.5保持。

预检PASS实际8FITsteps；完整组self_attn/FFN/LN梯度均正，注意力组self_attn正、FFN/LN零。128样本训练步 .119/.147秒仅短工作量测量，不保证整轮吞吐。双方真实两轮连续/中断参数、boolbuffer、optimizer、CUDA/loader/numpy/Python RNG、history/calibration EXACT。旧boolean检查器的减法不支持已以torch.equal完整复核，不是训练异常。preflight源锁+9172旧结果SHA（含upstream原结果和补评估）启动前通过；完成再次全核验。原缓存rawtrain/dev按固定ID只读复用，不改父源码/结果。

预计210–270分钟，规划240；单次60%时间检查19:13:18，初估分析20:49:18香港时间。之前保持静默不查询PID/log/results；到检查时按实际arm/seed/epoch及calIO/18policy评估修正剩余，完成立即审计，不高频轮询。估时参考上一轮实测218.6分钟，区别于旧偏乐观微基准。

完成必核验18policySHA、18显式IDs有限K20、全部存储指标独立重算、白名单冻结、report/audit/SUCCESS、9172旧结果不变。5%change双收益、protected1%、三seed、native AND fullcontrolCI、samepath/coverage/dup/hardmotion和全组运动分布1%门槛不放宽。首次重大信号额外场景样本相关性/三seed不确定性审查。仍为重复ETHdev探索，不是意图理解/独立生成/筛选改进证据。第三个新实验完成必须cycle6完整总review。
