# Anchor feedback result — cycle3 experiment2

正常结束：2026-10-02 00:24:04 HK，实际45分07秒。六次新80epoch训练，18个policy文件不是18次独立训练。
独立复核18模型SHA、18预测显式样本ID、1516×20×12×2有限值及全部存储逐样本指标一致；内置冻结18项审计PASS，worker及退出SUCCESS。日志旧SET_CONTEXT/NATIVE标签来自复用reporter，不是额外实验。

校准锁定的constrained模型，三seed平均：
| 组别 | 原native ADE/FDE | 无反馈control | 几何反馈 |
|---|---|---|---|
|overall|.219154/.399725|.219154/.400719|.218374/.402120|
|stable|.175177/.313804|.175330/.316292|.173752/.314145|
|transition|.275791/.512714|.275763/.512306|.275106/.513668|
|change|.372175/.696326|.371474/.691589|.374389/.710006|

突变ADE/FDE对native退化0.595%/1.964%；相对无反馈control退化0.785%/2.663%。对native的delta CI ADE[-.004973,.010447]、FDE[-.002320,.031935]跨零。对freshcontrol的FDE delta+.018416，CI[.003348,.035736]不跨零：本设置出现探索性增量伤害信号；不是证明任何几何输入永远无效。ADE对control CI[-.004425,.010857]跨零。仅3seed/重复ETHdev，不能声称独立泛化或因果瓶颈证明。

同路径change joint .752697 vscontrol .743612(+1.22%)，native .746631；不同oracle不可拼成同一轨迹质量。change粗family覆盖.925262 vsnative .910314，但准确度更差。overall加速度/jerk均值约+10.24%，运动分布门槛失败，即使绝对硬阈值通过。选中epoch反馈58/31/72，对照64/45/49；不依据dev把final或legacy升级成主结果。

5%收益、逐seed一致、CI、samepath和运动分布FAIL；保护组、覆盖、重复、硬运动学PASS。无重大结果、无新baseline、无筛选/意图理解进展。停止将该直接3anchor几何反馈作为正向方案。

下一步FIT/CAL固定128/group、3control-constrained检查点，无optimizer更新、不访问dev/test：stable-change梯度余弦fit [.1000,-.1641,-.1219]；cal[-.4577,-.0929,.0160]。支持共享头存在局部干扰的可能，不能证明是主要失败原因。组别混合与候选mode不是同一概念；winner槽位同时跨多个运动组，因此按mode解耦也可能无法解决。诊断JSON含sample IDs/源SHA，非总体统计或独立重复。

第三项设计保留原context/mode特征、放弃额外geometry输入，只把新head最后6维共享映射改为20个native-slot独立映射，fresh共享对照；原F16/scorer/decoder全部冻结。参数25158 vs32568(+29.46%)，不能归因成等容量纯架构证据。当前仅设计，尚未预检或启动；预检/EXACT恢复通过后才允许启动，第三完成必须cycle3总review。
