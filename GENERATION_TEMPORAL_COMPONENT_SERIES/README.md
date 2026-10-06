# Cycle6 experiment3 — temporal attention versus FFN

实际于2026-10-02 19:36:39香港时间启动。Windows独立按需无周期任务Trajectory-TemporalComponent-Cycle6，Python PID23904/服务父PID2728；启动确认看到TRAIN zero seed1及epoch1/80，stderr空。关闭Codex不结束OS任务，机器须保持登录/开机/不睡眠；自动分析仍需应用可用。DESIGN.md是锁定的历史预注册，README/runtime_state为当前状态。

六NEW80epoch/三配对seed。zero仅训练最后时间层self_attn+新anchorhead62406参数；mean_std仅训练最后时间层linear1/linear2+同head62310，两组LayerNorm固定。差96参数，近名义容量不等有效容量纯因果证明。重复attentioncontrol同架构种子非独立泛化。无旧治疗head warmstart。原其他F16、native decoder/scorer全固定；online原观测9agent8step重算tokens/coreeval无dropout，两组真实autograd只作用白名单。原nativebaselineK20固定/3anchorcorrection/commonvelocityclip1.5/originalfit25912cal2879/lossprotection.1/constrainedcal/orderLR保持，无GT未来group/logitsforward/新loss/系数/epoch扫参；不宣称意图/独立生成/排名改善。

真实GPU8FITsteps preflight PASS/fullpaired所有初始化EXACT/native初始输出EXACT/token-native alignment/finite128K20/bound/split/source。Attention梯度.000438866，FFN梯度linear1 .000128059/linear2 .000123984；各arm非白名单梯度零且权重exact冻结，两组LN梯度零。原rawtrain/dev按明确IDs只读缓存复用，均来自ETHtrain固定子集，不读test或寻找错误ethdev文件。9716旧结果SHAmanifest含之前所有结果与Subspace；预检前后和启动源锁通过，完成必须再全核验。

两组真实2epoch连续/中断参数/boolbuffer/optimizer/torchCUDAloadernumpyPythonRNG/history/calibration EXACT通过；旧bool减法检查器以torch.equal复核，非训练失败。128样本训练步短测 .102/.061秒仅描述，不保证整轮吞吐；估时结合上轮六online107m54。

预计95–140分钟，规划120；单次60%time query20:48:40，初估分析21:36:40香港时间。此前静默不查PID/log/results；一次检查实际arm/seed/epoch/calIO18policy评估后修正剩余及deadline，完成即时审计，不高频轮询。完成必18policySHA/18显式ID1516有限K20/全部存储metrics独立重算/frozen白名单/report/audit/SUCCESS/9716oldhash不变。

所有major标准不变：changeADE/FDE各>=5%，protectedoverallstabletransition<=1%harm，三seed一致，native及freshattentioncontrol CI不跨0，同路径/覆盖/重复/硬运动学及每组speedp95/accelmean,p95/jerkmean,p95<=native1%harm dt.4。首次信号需场景样本相关性和仅三seed不确定性充分性审查，重复ETHdev探索非独立泛化。此为cycle6第三项；完成先产出全三实验总review再决定大方向，不能盲第四局部子模块/系数/epoch变体。
