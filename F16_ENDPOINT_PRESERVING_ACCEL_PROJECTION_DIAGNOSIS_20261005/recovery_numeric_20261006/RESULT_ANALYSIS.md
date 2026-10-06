# 终点保持加速度投影：完整结果与解释

## 结论

有限输出约束能消除原有大部分额外加速度，而保留绝大多数已形成的完整好候选；但没有修好 CAL 上的预测终点和迁移问题，且部分速度高分位仍越过原1%保护。该结果支持把“额外加速度代价”和“学不准/不能迁移”分开研究，不支持宣称 loss 或 decoder/head 的唯一根因已经证明。

证据：[RESULT](RESULT.json)、[生产保存算术](ARITHMETIC_AUDIT.json)、[全冻结核验](FROZEN_REPLAY_AUDIT.json)、[CPU补充复核](completion_review/REVIEW.json)、[全48组数值](RESULT_ANALYSIS.json)。仅现有六 final 模型冻结 FIT/CAL，不是新训练、DEV收益或第三formal；研究保持2/3。

## 核验与资源

worker和exit双SUCCESS，完成2026-10-06 01:01:45.648515HK，实际399.566826秒（6分40秒）。唯一half检查实际01:03:47.1980679发现全部已完成，taskReady/processnone/stderr0/14done；不再查询50904或任何已完成旧PID，不等旧01:10截止。

14jobs/28输出SHA、524source/31802oldfiles初末SHA不变；六head全weights/bool/scalers逐项exact。生产完整保存质量/原GT-freecomplete-link/pooledmotion/projection重算PASS。CPU补充26snapshot、374283行次全部保存同候选质量/J/组平均/全局motion重算max0、28输出/524source/31802old前后SHA PASS；不含torch、模型前向、optimizer或第二次完整聚簇（生产已全聚簇复算）。端点最大误差1.77636e-15m，nominalreference最大1.52716e-9m，alpha复算最大7.77156e-16。数值alpha余量不是评价epsilon；float64投影输出明确区别原float32。

生产peakRSS3.08047/private3.88969GiB，系统minimumphysical15.56498/commit16.23339GiB；6GiBprocess以及系统两项各6GiBreserve通过。补充CPUprivate2.48776GiB。预检峰值不冒充生产峰值。

## 首要指标：同一条完整候选

严格同一candidate ADE≤.2m AND FDE≤.4m，不能拼接不同slot的minADE/minFDE。FIT变化组6410个窗口、CAL变化组700个；原生分别2603（40.61%）、243（34.71%）。每seed重复同IDs，不把三seed相加当独立样本。

| 对照 | seed | FIT变化组：原→投影 | CAL变化组：原→投影 | CAL救回/丢失 |
|---|---:|---:|---:|---:|
| 均匀双专家 | 1 | 3585→3569 | 222→221 | 1/2 |
| 均匀双专家 | 2 | 3505→3487 | 231→228 | 0/3 |
| 均匀双专家 | 3 | 3519→3515 | 232→231 | 0/1 |
| 固定候选路由 | 1 | 3642→3629 | 227→226 | 0/1 |
| 固定候选路由 | 2 | 3632→3619 | 237→237 | 0/0 |
| 固定候选路由 | 3 | 3616→3607 | 221→220 | 1/2 |


FIT变化组原有good保留99.26–99.66%，投影新增救回7–11/丢失12–26。CAL变化组保留98.70–100%，救回0–1/丢失0–3；仍所有六组低于原生243。不是“投影把所有收益压回原生”，也不是解决CAL泛化。

## 运动学：加速度成功不等全项成功

变化组原加速度均值相对native害1.7766–2.4383%，投影后0.7456–0.7970%。全部48组accmean均≤原生1.01倍，48组总体范围0.6980–0.7970%；逐窗口严格保存物理重算也通过，没有epsilon/floor。

五项pooled ALLK motion联合41/48组PASS。剩7失败均是speedp95，害1.0163–1.4851%：uniformseed2FITchange；routedseed1/2/3FITchange；routedseed2CALchange；routedseed3CALtransition/change。jerkmean/p95、accelp95和accmean在这些组通过。既有终点漂移可能保留速度问题，但这只是解释假设，不等证明速度失败只能由endpoint造成。

## 路径与数量的副作用

全部48组平均minADE与平均Jjoint略微恶化；端点保持使平均minFDE变化最大约3.342e-9m，属于舍入而不是新预测能力。J=minKmax(ADE/.2,FDE/.4)，早中晚采用一个完整Jclosestslot，不拼分段oracle。质量阈值敏感性.3/.5只描述、不改标准。

完整覆盖大体略降，但有反例：CAL整体routedseed1 1475→1476、seed2 1500→1500；不能声称每组都降。GT-free有效good簇多数小降或不变，CAL总体routedseed1 2302→2304；数量不是越多越好，更不能用它抵消覆盖下降。原scorerargmax只是继承原slot，不代表针对投影候选新评分或运动意图概率。

投影alpha中位数约0.503–0.612，5%约0.187–0.254，95%=1；它确实削减时序偏离，不只是数值补丁。数值预约引起的alpha差小（各CAL最大约2.4–2.6e-8），实际endpoint/reference量级已独立审核，不把科学函数收缩与roundoff混淆。

## 当前分支与下一步边界

仍走“FIT能形成更多好候选，但CAL不稳定迁移”的分支，同时保留速度/终点质量问题。并未进入“目标、credit和所有motion完全对齐，只剩结构”的终端分支。新的有用证据是：多数FIT完整候选收益并不需要超过1%加速度预算。

建议下一正式方案再精确设计为：相同合法head、相同既有loss/data/预算，在训练内对照原输出与固定端点保持时序投影；先检查可微连接、量化/float64路径对优化的耦合、真实两armGPU与2epoch完整RNG/optimizer恢复、全部motion和资源。它测试能否训练时适应该可行约束，不意味着扩大scorer/旧decoder解冻或新增质量loss。尚未实施或预检这一训练方案；本次完成后等人类讨论，不自动第三训练、不扫alpha/权重/width/LR/epoch。

## 反方检查2：分析阶段（同助手inline，不是独立外审）

PASS_WITH_LIMITATIONS。Major：41/48不是allmotion通过；CAL原生更好不能被FIT信号覆盖；总体CALseed1+1是反证；零新增救回不能说意图理解增强。保留以上全部证据。最强反驳：投影仅保住已有好轨迹，并没有解决它们为何无法跨split迁移。

## 反方检查3：最终阶段（同助手inline，不是独立外审）

PASS_WITH_LIMITATIONS。Major：冻结后验投影不是可训练性或有限AdamW更新证据；原生运动预算不是现实运动真值；三个场景/重叠窗口/F16已见/CAL单university且旧checkpoint选择存在限制。没有独立泛化、可靠groupCI、SOTA、心理意图、不可预测率或唯一根因结论。NoCritical：按原注册指标完整审计可支持局部几何可行性结论。

本报告由AI协助分析和审查，仅使用锁定本地数据与已保存结果，无外部论文或外部审稿。原major所有条件不放宽，正式计数不变，F16 baseline不变，未声称长期goal完成。

