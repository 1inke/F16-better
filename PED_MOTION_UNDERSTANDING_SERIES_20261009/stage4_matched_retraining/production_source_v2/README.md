# E3 fresh完整生命周期输入消融：后台已启动

唯一任务Trajectory-PedMotionUnderstanding-MatchedRetraining-20261009，根orchestrator实际PID35588/parent2688，启动2026-10-09T13:41:50.604922+08:00。一次startup确认current task Running/零trigger/新stderr0/current trainer14516；未观察生产epoch完成。progress.json当时是预检noNbr保存阶段，旧时间早于生产，不能当新训练结果。

预算五fold×seeds1/2/3×full/last2/noNbr=45条fresh H3_30→自有branch H3验证选定→V2_20，90阶段/2250epoch。原科学AST、训练信息变换、预算/初始化/validation选择固定。GT训练监督确实含原H3内部oracle候选选择，详见TRAINING_GT_USAGE_NOTE；测试forward不读未来GT。旧模型/结果/失败源码保留不改。

全部当前门槛已PASS：472相关来源原digest初末；五fold全raw-ID-cache米/dt.4一致且train/test目标交集0；六actual513FIT+12val两epoch连续/中断ALLparams_boolbuffers_EMAoptimizer_scheduler_ALLRNG_history_selected EXACT；48个原loss/allgradient batch replay EXACT；last2/noNbr实际全路径被删信息不泄露；三个V2保存全部K20质量原SciPy聚簇/pooledmotion CPU EXACT、负source/output/protocol拒绝。非所有fullfold两轮、非全部历史34k SHA、非外审。

实测H3 FIT64每step.06384–.06446s，V2.04533–.04607s，每arm4warm32measure、实际ETH1516验证.68588–.76764s；无保存benchmark模型/结果后选。总训练吞吐名义47275s，验证377s，source/初始化/保存重读及最终审核另计。范围795–1490min，规划1023min≈17h03。一次预计50%节点2026-10-09T22:13:20.604+08:00，初估完整核验2026-10-10T06:44:50.604+08:00；实际速度/阶段将在唯一半程修正，不高频poll或每天旧节点重复。

预检private峰3.79524GiB/benchmark3.77158，physical最低13.36/commit17.15GiB；生产峰未知待实际RESOURCE，过程6GiB及两系统余量6GiB不降低。可用磁盘47.18GiB，输出保守1.287GiB+checkpoint/temp2.197GiB预算PASS。内部batchguard不是对话轮询，不改OS/pagefile/关闭用户程序。

完成必须根worker和exit双SUCCESS+90完整source-bound done/45V2 explicit-ID保存预测/2250history/allsource和outputSHA/完整saved quality fullclusters five-motion配对trackCI独立审核后分析；epoch100%或root exit单独不算。执行成功不是majorPASS/理解运动/新baseline。保留原major/所有five-motion1%阈值、3seed/少scene/相关窗口/重复原test/CAL限制。E5旋转、神经E6剔兄弟及feature输入可辨识性仍待后续，whole_series_complete=false。

v1审计器deepcopy非叶缓存失败来源完整保留；v2仅干净模板+state_dict+training模式副本修复，实际六组全部重预检PASS，不假继承旧PASS。运行中不得锁后改科学源/覆hash/慢就重启/重训旧结果；异常保留新failure/code/log/partialoutputs，实质science/source-ID/resource/scope阻碍STOP人类。
