# Slot-head result — cycle3 experiment3

Worker SUCCESS2026-10-02 01:35:12.886HK;实际43分29秒。六NEW80epoch训练，18策略产物。独立18SHA/18显式ID/1516×20×12×2finiteK20/全部保存逐样本指标重算PASS，内置frozen18PASS，正常SUCCESS退出。校准epoch可行性失败不是训练崩溃。

共享对照changeADE/FDE .371383/.691852；独立slot输出 .375808/.705654；native .372175/.696326。对native退化.976%/1.339%，对对照退化1.191%/1.995%；对nativechange delta CI ADE[-.007551,.015296],FDE[-.018320,.036482]，对freshcontrol[-.007183,.016307]/[-.014530,.041483]都跨零。没有5%收益，也不能说已经统计证明普遍有害。

独立组overall/stable/transition ADE/FDE .218866/.396436、.172828/.306873、.281366/.514170。Stable对native名义改善1.34%/2.21%，对应ADE/FDEdelta CI[-.004525,-.000247]/[-.012175,-.002056]不跨零；对freshsharedCI也不跨零。但transitionADE+2.02%超过保护门槛，change无收益，不换baseline。重复ETHdev/3seed、多组比较未提供独立泛化确认。

Change同路径joint .752577 vscontrol .743681(+1.20%)、native .746631。粗family覆盖.926756 vsnative.910314，未转化精度；overallaccel/jerkmean+13.99%，运动分布FAIL，即使硬运动学PASS。独立组校准epoch26/23/24，对照37/61/50；独立finalchange .388912/.733070，延长训练没有解决变化预测，不能dev择优替换checkpoint。不能仅凭final恶化认定唯一原因过拟合。

5%/protectedgroup/seed/CI/samepath/motiondistributionFAIL，feasible/coverage/dup/hardmotionPASS。共享vsuntied25158/32568参数差+29.46%，容量与专用化耦合；即使获益也非等容量架构证明。控制实际又fresh跑了一次，但相同seed/协议输出与DirectAnchor相同，不是独立重复。原decoder/F16/scorer全部冻结，仍nativebase+3anchor correction/commonbound，而非独立意图生成或筛选改进。

第三完成后cycle3总复盘已另存，下一方向转向完整冻结观测tokens的读取，而不是继续扩大slot专家/改系数。新方向只设计，未训练。
