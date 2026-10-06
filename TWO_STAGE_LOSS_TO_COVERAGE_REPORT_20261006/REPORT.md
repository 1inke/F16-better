# 两阶段研究报告：loss改善是否带来完整好候选
日期：2026-10-06，香港时间。范围从人类标注的“启动第一阶段诊断，独立后台运行，不重训或修改模型”开始，到物理反馈三项正式实验完成与总复盘。更早的实验只作为诊断所读取的既有checkpoint与反证，不计为本阶段新训练。

## 摘要
本阶段最明确的结论不是“新模型已经更好”，而是定位有了进展：
1. 原质量目标在FIT确实能推动同一条完整好候选增加，不是只降低误差而完全不增加覆盖。
2. 已有合格候选时，hard winner通常把质量反馈给合格候选；但尚无合格候选的窗口，反馈分配是否最有利仍未解决。
3. 位置质量改善伴随加速度/jerk代价；直接物理反馈与训练内投影可以明显控制这些副作用。
4. 控制加速度后，FIT仍能形成更多完整候选，但CAL终态仍不如原生模型；选定checkpoint有小幅CAL改善，不能说所有时点都失败。
5. 三项正式实验都未达到原major标准，F16 baseline不变。当前优先级更偏共享映射、候选条件化与迁移验证，而不是先换scorer或重复soft winner/权重/epoch扫参。

以上是证据支持的研究优先级，不是loss、head或decoder唯一根因证明。依据：[第一阶段诊断](C:/Users/linke/Documents/New project/F16_LOSS_QUALITY_CREDIT_MOTION_DIAGNOSIS_20261005/RESULT_ANALYSIS.md)、[三项总复盘](C:/Users/linke/Documents/New project/GENERATION_TRAINED_ACCEL_PROJECTION_SERIES_20261006/recovery_source_20261006/evaluation_recovery_20261006/PHYSICAL_FEEDBACK_STUDY_REVIEW.md)。

## 1. 本阶段做了什么，怎样判定成功
第一阶段是冻结诊断：18个已有checkpoint，分析原loss、完整候选质量、直接credit与参数层物理方向；没有重训或修改模型值。随后阶段二完成三项正式对照，每项两arm×3paired seed×80epoch，共18次全新训练、1440条epoch历史。每项有constrained/legacy/final三种保存政策，因此54个policy条目不是54次独立训练。中间另有冻结端点保持投影诊断，不增加正式训练计数。

整个阶段原F16、原decoder与scorer固定。正式训练仅新head；预测forward不输入未来GT、分组标签或logits。没有读heldtest、其他fold或外部数据。

首要标准是至少出现一条同一完整candidate同时满足ADE≤0.2m、FDE≤0.4m。minADE与minFDE分别好但来自不同candidate不算完整覆盖；也不拼接前中后段oracle。连续指标J=min_k max(ADE_k/.2,FDE_k/.4)，J≤1与这一覆盖标准一致。

其次报告GT无关去重后有效好候选簇：全K20按meanpath .10m、endpoint .20m做complete-link聚簇，簇含至少一条完整good才计有效good。数量不是越多越好，复制轨迹不能凑数量。最后才评估选择；当前沿用native slot，不是对修正路径重新训练的scorer。

“constrained”是原CAL规则选定的checkpoint；“final”是80epoch终态，二者不能混用。FIT与CAL在本报告中分别指newhead适应训练集与校准/选checkpoint集，CAL不是独立未见测试集。原5%变化ADE/FDE收益、protected组≤1%损害、三seed、native/freshcontrol CI及覆盖/重复/硬运动学与五项pooled运动学≤native1%全部不放宽。依据：[诊断定义](C:/Users/linke/Documents/New project/F16_LOSS_QUALITY_CREDIT_MOTION_DIAGNOSIS_20261005/RESULT_ANALYSIS.md)、[三项设计与门槛复盘](C:/Users/linke/Documents/New project/GENERATION_TRAINED_ACCEL_PROJECTION_SERIES_20261006/recovery_source_20261006/evaluation_recovery_20261006/PHYSICAL_FEEDBACK_STUDY_REVIEW.md)。

## 2. 第一阶段：你的三个问题得到什么答案
### 2.1 Loss下降→完整覆盖改善：FIT总体一致，CAL不能迁移
读取三种表示族×3seed的九个final checkpoint，变化组结果如下：

| 指标 | 原生F16 | 九个final checkpoint |
|---|---:|---:|
| FIT完整覆盖 /6410 | 2603，40.61% | 3301–3453，51.50–53.87% |
| CAL完整覆盖 /700 | 243，34.71% | 216–242，30.86–34.57% |
| FIT原total变化 | 参照 | 下降19.01–21.56% |
| CAL原total变化 | 参照 | 上升1.71–4.73% |
| FIT加速度均值损害 | 参照 | +15.09–17.89% |
| CAL加速度均值损害 | 参照 | +14.06–16.27% |

所以，“FIT loss下降，但完整覆盖完全没增长”不是主要结果；“CAL loss也下降，却通过损失好候选获益”也不是整体事实，CAL loss本身上升了。当前Q=min_k(ADE_k+.5FDE_k)是hard-winner质量，不是20条候选的平均位置误差。

逐窗口ΔL与ΔJ的Pearson相关为FIT .8576–.8695、CAL .8564–.8885，方向总体一致。但在loss下降的窗口中，J反而恶化仍有FIT8.38–9.62%、CAL12.08–17.52%。这说明不完全对齐，不能给原目标完全免责。相关性只比较已保存checkpoint端点，不等“每下降一单位loss”的逐步因果收益。

Rescue–lost交换确实存在：FIT每模型救964–1121、丢241–304；CAL救42–59、丢58–81。关键是仅5.59–8.62%的FIT lost、0–6.25%的CAL lost发生在该窗口loss下降时。大多数丢失同时loss上升，更支持训练/迁移伤害，不能一概解释为loss奖励了丢失好候选。

这里还要区分“原已覆盖”“J改善但仍没跨门槛”“J恶化”。仅看有没有新增rescue，会把不同现象混在一起。依据：[第一阶段§1](C:/Users/linke/Documents/New project/F16_LOSS_QUALITY_CREDIT_MOTION_DIAGNOSIS_20261005/RESULT_ANALYSIS.md)。

### 2.2 Hard winner：已有good通常拿到反馈，未covered仍有疑点
在已经存在完整good的变化窗口中，final时quality winner仍不合格的比例，FIT不足0.8%，CAL最多1.67%。J最近slot与Q winner不同的比例则为FIT4.59–5.24%、CAL5.43–9.71%。因此不能说hard winner经常忽略已有合格候选。

但有一条重要反证：旧GT辅助Gaussian witness替换单slot、其余19条不变后，几乎所有可用witness都能降低总loss；约68–69%的FIT、70–73%的CAL这类slot在当前状态没有直接primary质量梯度。这一缺口仍值得保留。

两点不矛盾：当前已经合格的候选通常被奖励，不代表最有修正潜力的尚未合格候选都拿到了最合适反馈。另一方面，GT辅助witness不证明当前persistent共享head能学到它，坐标primary=0不等共享参数完全没有间接反馈，J最近也不等最容易修。因此暂不把soft winner/top-k作为第一干预。依据：[第一阶段§2](C:/Users/linke/Documents/New project/F16_LOSS_QUALITY_CREDIT_MOTION_DIAGNOSIS_20261005/RESULT_ANALYSIS.md)。

### 2.3 质量与运动学冲突：支持有限干预，不支持唯一归因
九个constrained checkpoint在FIT与CAL，质量下降方向与加速度均值、jerk均值均呈局部冲突。原weighted residual-smoothness梯度范数相对primary约2e-8至2.3e-6；其定义是修正增量差分平方，不是完整轨迹的绝对物理加速度。

反例是final CAL：加速度均值仅2/9冲突，jerk均值0/9冲突，却依然有最终运动学恶化。最后一个局部导数不能解释整个训练历史；p95梯度还存在稀疏秩与ties问题。局部cosine不是有限mixedbatch AdamW更新的因果效果，也不足以决定“只改权重”或“必须改head”。

因此阶段二先测试一项直接物理加速度保护，而不是同时改assignment、head容量、scorer。依据：[第一阶段§3及反方检查](C:/Users/linke/Documents/New project/F16_LOSS_QUALITY_CREDIT_MOTION_DIAGNOSIS_20261005/RESULT_ANALYSIS.md)。

## 3. 阶段二：三项正式实验怎样改变判断
以下“整体覆盖”使用固定DEV1516窗口，“变化覆盖”使用其中change223窗口；原生分别733、43。每格三个计数对应三个seed，是同一批窗口重复评估，不能加总成独立人数。

| 正式对照的治疗 | 单因素 | 整体完整覆盖 /1516 | 变化完整覆盖 /223 | 五运动学通过 seed-group /12 |
|---|---|---|---|---:|
| 加物理P保护 | 原total vs 原total+.1P_acc | 727/732/737 | 42/47/52 | 4/12 |
| 固定candidate路由 | 均匀双专家 vs FIT中位数固定路由 | 727/738/729 | 42/47/43 | 9/12 |
| 训练内投影 | 同均匀双专家输出投影 vs 不投影 | 736/735/735 | 47/48/42 | 12/12 |

12组是3seed×overall/stable/transition/change四组，各组必须同时通过speedp95、accelmean/p95、jerkmean/p95；组之间并非12个独立样本。每项都是其自身paired control对照；三项之间容量、表示约束与数值精度不同，不能把顺序表当纯因果链。依据：[完整三项复盘](C:/Users/linke/Documents/New project/GENERATION_TRAINED_ACCEL_PROJECTION_SERIES_20261006/recovery_source_20261006/evaluation_recovery_20261006/PHYSICAL_FEEDBACK_STUDY_REVIEW.md)。

### 实验一：P_acc确实有效，但没有形成稳定质量收益
变化组final加速度代价由原约15%压到FIT1.72–1.88%、CAL1.44–1.55%，仍超1%门槛。正式constrained变化minADE/FDE收益仅0.2133%/0.8769%，远不到各5%；native和freshcontrol主要CI均跨0。整体覆盖有seed净减，有seed小增，无法稳定提升。

解释：原正则与完整轨迹物理反馈存在实际差别；加入直接反馈能减少副作用。但不能据此证明泛化问题由loss单独造成，也不能通过不断加大系数追求过关。依据：[物理保护结果](C:/Users/linke/Documents/New project/GENERATION_PHYSICAL_ACCEL_PROTECTION_SERIES_20261005/RESULT_ANALYSIS.md)。

### 实验二：固定分工没有解决迁移，不能否定所有conditioning
两个专家总参数91596，两arm都保留+.1P；只比较均匀混合与基于native预测相对最后观测速度的固定FIT中位数路由。正式变化ADE/FDE比F16恶化0.3088%/1.0144%。某seed改善，不同seed不一致，CAL final仍低于原生。

所以，这条固定q路由策略失败；不是“所有gating或candidate conditioning都无效”的结论。同名义参数不等同有效容量、梯度样本数和优化几何，路由也可能强化原预测错误。依据：[候选路由结果](C:/Users/linke/Documents/New project/GENERATION_CANDIDATE_ROUTED_MAPPING_SERIES_20261005/RESULT_ANALYSIS.md)。

### 中间冻结诊断：大多数FIT收益并不需要超额加速度
端点保持后验投影只收缩零均值时序扰动，不改同一code的终点修正。FIT变化组原good保留99.26–99.66%；CAL保留98.70–100%。全部48组加速度均值通过，但五运动学联合仅41/48通过，另7组speedp95失败；CAL六组覆盖仍低于原生243。

这支持把“额外加速度”和“学不准/不能迁移”拆开：FIT好候选收益并非全部依赖超额加速度。它没有新增未来信息，也没有修好原终点；冻结后处理不等可训练性。依据：[冻结投影结果](C:/Users/linke/Documents/New project/F16_ENDPOINT_PRESERVING_ACCEL_PROJECTION_DIAGNOSIS_20261005/recovery_numeric_20261006/RESULT_ANALYSIS.md)。

### 实验三：训练内投影守住选定模型的运动学，但没有解决完整预测
正式constrained三个seed四组五运动学首次全通过。整体覆盖由733变736/735/735，只净增3/2/2个窗口；change覆盖43变47/48/42，仍有一个seed下降。变化minADE/FDE却比原生恶化0.2162%/1.6004%，transition ADE+1.0913%也违反保护门槛，主要native/freshcontrol CI跨0。

最新整体rescue/lost分别18/15、27/25、24/22：净变化小，但并不是几乎没有单样例改变，而是救回与丢失互相抵消。有效good簇总数929变939/827/829，不能说多样性全部改善，也不能只凭簇数下降就判整体mode collapse。

final并非全部运动学通过：9/12组通过，三个change speedp95仍超1%。相同codes的投影保持终点，不等不同训练完成模型的最终终点相同。依据：[训练内投影完整结果](C:/Users/linke/Documents/New project/GENERATION_TRAINED_ACCEL_PROJECTION_SERIES_20261006/recovery_source_20261006/evaluation_recovery_20261006/RESULT_ANALYSIS.md)。

## 4. 最重要的共同结果：FIT学到了，CAL终态仍未迁移
仅比较同定义的变化组，不把不同split窗口配对，也不把final当选定部署结果。

| 模型 | FIT完整good /6410，三个seed | CAL完整good /700，三个seed |
|---|---|---|
| F16 | 2603，40.61% | 243，34.71% |
| 加P final | 3237/3254/3265 | 230/242/232 |
| 路由 final | 3642/3632/3616 | 227/237/221 |
| 训练内投影 final | 3621/3534/3560，55.13–56.49% | 226/232/222，31.71–33.14% |

第三项已经把变化组FIT/CAL加速度均值控制在native+1%内，FIT仍有14.52–15.88个百分点的完整覆盖提升，但CAL final仍低于原生。这是当前最强的迁移缺口证据，不宜继续用“只要再改加速度权重”解释。

必须保留反例：三项constrained CAL good分别244/247/246、245/239/244、248/252/248。最后一项三seed都略高于原生243，说明选定早期checkpoint不是完全无效。可这些checkpoint已经由CAL筛选过，不能作为独立未见泛化证明。继续训练后FIT提高、CAL回落，可能涉及共享映射干扰、有限优化、场景分布与objective未充分保护；目前无法一锤定因果。依据：[全三项FIT/CAL复盘](C:/Users/linke/Documents/New project/GENERATION_TRAINED_ACCEL_PROJECTION_SERIES_20261006/recovery_source_20261006/evaluation_recovery_20261006/PHYSICAL_FEEDBACK_STUDY_REVIEW.md)。

## 5. 回到你给出的决策地图：我们现在在哪个分支
- “loss降低但完整覆盖不增，优先改objective”：有局部反例，不能作为整体首要判断，因为FIT完整覆盖明确提高。
- “hard winner经常忽略已有好候选，优先assignment”：已有good时不支持“经常忽略”；未covered窗口仍是未排除的问题。
- “quality-motion冲突，先区分反馈还是参数化”：已执行。直接物理反馈有效，训练内约束更有效，但都没有解决稳定完整质量收益。
- “FIT上完整覆盖改善，CAL仍损失，重点共享映射/条件化/泛化”：目前证据最接近这一分支。
- “全部目标、credit和motion已对齐，只剩结构”：尚未到这里，final速度、未covered credit和有限优化仍有缺口。
- “生成端稳定后再scorer”：尚未满足前提。旧argmax沿用原slot，不是对新轨迹的概率判断，不能据其保守输出证明新head忽略运动变化。

与人眼观察的联系：你看到“局部变好但后半段/终点拉开”的现象仍与结果相容；只是本阶段没有证明终点是唯一原因，也没有证明模型理解了心理意图。完整覆盖与J弥补了分开minADE/minFDE的盲点，多样性、选择和早中晚同路径质量仍需共同看；单一oracle指标确实不等于完整任务能力。

建议当前优先级：先让映射/迁移检验可辨识，再据证据选结构；不先训scorer，不重复权重、epoch、路由分位数或profile局部变体。下一具体建议是在现有FIT内部做按行人/场景分组的head适应验证，与匹配随机窗口划分对照，原CAL保持不动。它改变原固定训练/验证协议，目前仅建议、尚未获本次请求批准、未实施或启动。依据：[三项总复盘的路线决策](C:/Users/linke/Documents/New project/GENERATION_TRAINED_ACCEL_PROJECTION_SERIES_20261006/recovery_source_20261006/evaluation_recovery_20261006/PHYSICAL_FEEDBACK_STUDY_REVIEW.md)。

## 6. 可靠性、局限与本次报告性质
三个正式实验均完成全六80epoch历史、显式ID/有限K20、保存指标/聚簇/运动学/CI复算、冻结参数buffer和科学source/旧文件SHA审核，计算/审核成功不等方法达标。第三项训练完成后读取器误将保存预测当dataset cache，导致评估失败；隔离恢复读取与审核，无重训，原失败代码、模型、日志及预测全部保留。数值/资源/源锁阻碍也有保留记录，没有放宽科学门槛使PASS。详见各结果报告。

本次只读取已保存报告并汇总，没有查询旧进程、重放全部模型/梯度、新训练或重新划分数据。原生产的保存算术复算不是外部独立复现；报告的AI辅助与反方审查由同一助手完成。研究审查技能使本报告保留CAL早期改善、未covered credit、final速度失败及容量/精度不同等反证，不把它们压成单一根因。

重要局限：只有3seed、3scene；轨迹窗口相关；原F16已见过数据；CAL只有university且反复选checkpoint；DEV重复探索。当前CI是探索性结果，不能据此宣称SOTA、独立泛化、心理意图理解、不可预测率或唯一decoder根因。

内存长期按约17GB可用规划，进程6GiB、系统physical与commit各预留6GiB。原第三训练失败时没有保存完整生命周期峰值，不能用恢复峰值替代它。没有修改OS/pagefile或停止用户程序。

### 内联反方核查
1. 范围：第一阶段读取既有模型不是新训练；中间投影诊断不占第三正式名额。只按实际完成记录计数。
2. 分析：整体关联不等loss充分对齐；只看final会删去constrained CAL好结果；只看运动学会删去覆盖与误差失败。以上均保留。
3. 交付：当前优先级是待检验假设，不是根因结论。三项major均失败，baseline与长期目标未升级/完成；下一协议变更不自动执行。

## 一句话结论
这段实验已经证明“完整好候选可以在FIT上学出来，而且多数收益不必依赖额外加速度”；还没有证明“这些完整好候选能跨seed、跨CAL稳定出现”。我们现在更接近共享映射与迁移分支，但没有把loss或credit完全排除。

