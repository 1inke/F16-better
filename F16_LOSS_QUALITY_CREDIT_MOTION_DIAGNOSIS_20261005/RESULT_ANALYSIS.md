# Loss→完整覆盖 / credit / 运动学：结果与阶段二决定

## 结论
当前质量目标不是“只会改善均值而完全不推动完整候选”。它在FIT确实推进同路径完整覆盖，已有好候选通常也被hard winner奖励。当前最突出的缺口是**位置改善伴随物理加速度代价，以及FIT改善没有迁移到CAL**。因此不先换soft winner、scorer或扩大head；阶段二固定原head和quality assignment，测试一项直接的physical-acceleration保护反馈。

这只是可证伪的研究优先级，不是loss/head唯一根因结论。损失统计来自checkpoint端点，不等每个训练step的因果效率。原baseline、所有major门槛及运动学1%保护不变。

## 完成与可靠性
真实worker16616于2026-10-05 17:38:48.564450HK完整SUCCESS；启动17:25:06.401691，墙钟822.162759秒（13分42秒）；RESULT内部运行819.125秒。唯一完成核验17:44:01.8486998发现taskReady/processnone/双SUCCESS/stderr0，不再查询该PID。
29jobs、155输出SHA、459source和30071旧文件初末SHA、18模型全部权重/布尔/scaler buffers EXACT、生产ARITHMETIC_AUDIT与FROZEN_AUDIT均PASS。生产审核重构38snapshot完整原GT-free聚簇及保存指标、credit/edge/vector摘要。
新增NumPy completion_review再次独立重算snapshot质量、54条edge的所有注册列/计数、梯度摘要，误差最大4.773959005888173e-15，155输出/459source/30071旧SHA前后PASS。该补充无model forward/optimizer，不声称所有gradient再次backward或全聚簇另跑一遍，更不是外部审稿。
生产peakRSS2.098GiB/private3.622GiB，最小physical16.546/commit15.215GiB。补充private.211GiB。保持约17GB可用内存约束、进程6GiB、系统physical与commit各预留6GiB；未改OS/pagefile或停止用户程序。NONFORMAL，不增加正式计数；此前两研究各3/3 review保持，历史cycle11不重置。

## 1. Loss下降究竟改善什么
主质量标准：同一完整candidate ADE≤.2m AND FDE≤.4m。J=minK max(ADE/.2,FDE/.4)≤1表示合格；Q=minK(ADE+.5FDE)是当前hard winner，**不是20条候选平均误差**。L为原total在全split固定人口分母的贡献；不是训练mixedbatch/dropout/AdamW实际更新。

下表是九个final checkpoint（3表示族×3seed），每个seed在相同ID上，不能把重复窗口当独立人数：

| change组 | 原生 | 九个final范围 |
| --- | --- | --- |
| FIT完整覆盖 | 2603/6410 =40.61% | 3301–3453/6410 =51.50–53.87% |
| CAL完整覆盖 | 243/700 =34.71% | 216–242/700 =30.86–34.57% |
| FIT原total均值 | 原生参考 | 下降19.01–21.56% |
| CAL原total均值 | 原生参考 | **上升1.71–4.73%** |
| FIT pooled ALLK加速度均值 | 原生参考 | 上升15.09–17.89% |
| CAL pooled ALLK加速度均值 | 原生参考 | 上升14.06–16.27% |

所以不能说“CAL loss也整体下降、却没有coverage”；CAL均值本身就恶化。FIT也不是完全合格的学习，因为运动学失败。
逐窗口ΔL与ΔJ的Pearson：FIT .8576–.8695，CAL .8564–.8885。总体同向，但并非无例外。
在loss下降窗口内，J反而恶化的比例FIT8.38–9.62%、CAL12.08–17.52%。这些例外值得研究，不足以抹去总体覆盖增量。ADE/FDE下降却没有新rescue要区分“原已covered”“仍未跨门槛但J改善”“J恶化”，不能把三者合并为objective mismatch。

native→final的change组：FIT各模型rescue964–1121、lost241–304；CALrescue42–59、lost58–81。**仅5.59–8.62%的FIT lost、0–6.25%的CAL lost发生在该窗口loss下降时**；大多数lost同时loss上升。它更符合受训练/迁移伤害，不能概括为“loss通过牺牲好轨迹获益”。

例：raw seed1 FIT救983、损241；loss下降的4388窗口内救962、损19。CAL救49、损62；loss下降333窗口内救45、损3。不能跨FIT/CAL窗口配对，也不能从类型集中直接确定共享干扰还是缺保护。

## 2. Hard winner是否把质量反馈给错候选
在已经存在合格候选的change窗口中，“winner仍不合格”：
- 原生FIT31/2603=1.19%，CAL2/243=.82%。
- final FIT10–26个，占covered不足.8%；CAL0–4个，最多4/240=1.67%。
因此“已有好候选通常被hard winner忽略”不受支持。这个winner是训练质量分配，不是原scorer选择，二者不可混淆。

final J最近slot与Q winner不同：FIT294–336/6410=4.59–5.24%，CAL38–68/700=5.43–9.71%。J最近不等共享head最容易修的slot；非winner坐标primary=0也不等没有共享参数间接学习。

另一方面，旧GT辅助Gaussian witness插入一条轨迹、19条保持不动后，几乎所有可用witness能降低原total，但约68–69% FIT、70–73% CAL这样的slot在当前状态没有直接primary梯度。两件事确实可以同时成立。这里witness不是当前persistent共享head可学性证明，更不是可预测性证明；不能据此直接重复旧softset/energy/mixture失败路线。Credit目前不是第一优先干预。

## 3. 质量与物理运动学的局部参数方向
仅45798现有newhead参数数值求导，值与全部旧权重buffer不动。下表为原primary质量下降方向与实际ALLK物理mean梯度冲突的checkpoint数：

| checkpoint | FIT accelmean | CAL accelmean | FIT jerkmean | CAL jerkmean |
| --- | --- | --- | --- | --- |
| constrained（9个） | 9/9 | 9/9 | 9/9 | 9/9 |
| final（9个） | 8/9 | 2/9 | 8/9 | 0/9 |

Constrained accelmean cosine FIT[-.4641,-.2152]，CAL[-.4135,-.0750]；沿质量下降方向局部会推高加速度均值。原weighted residual-smoothness梯度norm/primary约2e-8–2.3e-6，当前total与primary的物理方向几乎相同。它惩罚的是**修正增量差分平方**，不等绝对轨迹物理加速度，也不等“没用”或简单乘某系数必有效。
native→final loss下降的change窗口，ALLK窗口平均加速度均超过原生1%；这不是断言每条获救候选都不合理，也不是人类运动状态变化本身有错。

反例：final CAL jerkmean局部9/9不冲突，但此前轨迹仍恶化；不能用最后一个gradient sign解释整段训练。Global p95是极稀疏两秩/tie局部导数，结果混合，不能当大多数候选的方向。物理与primary的norm单位不同，不能跨族或跨量纲排行；cos、dot、方向导数不是有限mixedbatch AdamW的因果效果。

## 按人类决策地图选择
1. “loss与coverage同步且motion没恶化”：FIT前半成立、后半不成立；不能直接给loss免责转结构。
2. “loss/误差下降coverage不增”：少数例外存在，但不是FIT主要结论。先看J连续改善与阈值。
3. “loss下降频繁丢好候选”：存在但占lost少数；CAL netlost主要伴loss上升，优先保留迁移解释。
4. “hardwinner忽略已有好candidate”：比例低，不优先softmin/top-k。旧witness证据仍保留。
5. “质量信号到位但修不好”：共享mapping仍有嫌疑，但FIT有实际覆盖学习，CAL非迁移。
6. “quality-motion冲突”：constrained所有族/seed一致，是当前最可辨识干预点；不从cos唯一归因head。
7. “不冲突却motion差”：final CAL反例成立，训练尺度/历史/非线性仍需关注。
8. “FIT好CAL差”：成立，但FIT也motionFAIL，非纯过拟合充分证明。
9. “先改scorer”：生成尚未稳定达到coverage+motion，不进入scorer训练。

## 阶段二单因素
已在全新 [DESIGN](../GENERATION_PHYSICAL_ACCEL_PROTECTION_SERIES_20261005/DESIGN.md) 固定方案：
双方相同**active raw candidate-state32/persistent 45798 head**，原primary/hardwinner、decoder、scorer、原全部loss项/guards/data/budget不动；唯一因素为是否加一项**ALLK物理加速度均值相对native的保护hinge**。固定系数沿用原protection .1，1%容忍沿用原motion门槛，尺度只FIT native组统计，不CAL/DEV扫系数。
这验证直接物理反馈是否能保存覆盖学习并减少运动代价，而不是“加惩罚必解决泛化”。如果只有原样输出/覆盖受压、仍损CAL或其它motion，判失败；不降低门槛，不继续无尽系数扫参。仍不能由失败唯一认定结构根因。
实施预检节点2026-10-05 18:20HK，**不是启动或生产ETA**。当前没有新worker，只有设计；实际GPU/完整冻结审计、两arm真实2epoch EXACT resume、资源及实测benchmark PASS后才独立OS启动和登记正式计数。随后唯一50%时间检查→实际完成节点，不对话轮询。
本次具体分支干预依据人类两阶段持续研究指令，绝不扩大旧F16/原decoder/scorer解冻、事件监督、heldtest/fold/externaldata/cloud/messages/delete范围。

## 反方检查2（同助手内联，非外部审稿）
- Major：总体相关不能证明目标充分对齐；反例J恶化、lost及CAL harm均保留，不把相关当因果。
- Major：已有good winner合理不能排除未covered窗口的credit问题，oldwitness零credit仍有记录；不宣布hardwinner正确。
- Major：加速度惩罚可能限制真实转向、造成保守修正；新增保护仅是一项负向反馈干预，不构造意图标签。
- Major：batch组hinge不等fullpopulation全局gate；不能预设regularizer或结构哪个好。
判定：PASS_WITH_LIMITATIONS，足以选择有限的一项受控研究，不足以主张根因或major。

## 反方检查3（同助手内联，非外部审稿）
- 九checkpoint重复同ID、CAL单university且已选checkpoint、三scene/window相关、原F16已见FIT；不报告独立泛化/SOTA、可靠scene p值或心理意图。
- 只端点不能说明每个loss下降训练step；补充重算不等第二次全梯度反向或外部独审。
- 干预头沿用此前raw探索信号，属探索设计，不DEV独立确认；不能拿后续DEV做新loss参数选择。
- Future GT仅训练既有质量项/事后标签，绝不输入预测forward。原major及全部组五运动学≤native1%保持。
判定：PASS_EXPLORATORY_CONTROLLED_STAGE2；baseline不升级、goal未完成。

## 证据文件
[RESULT](RESULT.json)、[ARITHMETIC_AUDIT](ARITHMETIC_AUDIT.json)、[FROZEN_AUDIT](FROZEN_AUDIT.json)、[RESOURCE_USAGE](RESOURCE_USAGE.json)、[completion_review](completion_review/REVIEW.json)及全部jobs/records。原preflight合成精确cos断言失败来源保留，非本生产失败。
