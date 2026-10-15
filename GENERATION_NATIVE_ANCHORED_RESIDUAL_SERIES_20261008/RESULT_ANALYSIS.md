# 原生锚定小幅残差对照：完成分析

## 结论

这项实验完整执行并通过来源/保存算术审计，但**没有通过研究成功标准**，baseline不升级。原生锚定比恒速锚定更能保住变化难例的原有精度，却没有显著优于native20；即使位置修正很小，加速度均值仍越过1%保护线。不能把这归结为“模型已理解行人运动”，也不能仅凭本项断定loss或head是唯一根因。

来源：[RESULT.json](RESULT.json)、[全部18组CPU复算摘要](completion_review/SUMMARY.json)、[独立复算审核](completion_review/REVIEW.json)、[全部36条/144标量探索CI](BOOTSTRAP.json)。本文由同一助手进行分析及反方审查，非外部独立评审。

## 执行与审核

唯一worker11028/service2840在2026-10-08 02:27:09.675194HK启动，worker与exit双SUCCESS于02:30:55.870470HK，实际226.195276秒（3分46秒）。唯一计划half02:31:09.675194的实际检查02:31:53.4327331发现已全部完成，迟到43.7575391秒；不宣称按时半程，也没有再查询已完成进程/安排旧02:35完成节点。

两个配对arm×三个seed42/43/44=六次新训练，25epoch各/150行历史；18科学保存jobs、34620 run-window重放行（非34620独立样本）、103受保护输出SHA。实际训练量18909/outer1035/inner1856/原CAL2879，原target-track grouped划分不变；fixed25为primary，inner和CAL仅posthoc。三seed不是三份新数据。

生产的所有保存质量、原SciPy GT-free complete-link聚簇、五项pooled运动学、固定16/残差界及最终来源审核PASS。CPU补充不import torch、不模型forward/backward/optimizer，完整18jobs独立复算质量/聚簇/五motion/assignment、36条144标量CI、150历史/六checkpoint绑定manifest及SHA，最大差0；852相关source/input/prior SHA和103输出初末不变。不是历史34k全审计、所有模型/梯度第二次重放、或旧F16权重的GPU重放。旧F16/decoder/scorer本项未实例化或训练，使用受SHA保护的既有native与观察encoder缓存。[ARITHMETIC_AUDIT.json](ARITHMETIC_AUDIT.json)、[PRESERVATION.json](PRESERVATION.json)。

首预检仅输入维度schema错误，12×2=24误写48，source-lock前修为两arm共同120输入/25304参数；原失败源码设计协议完整保留。CPU补充首次18jobs全部算术一致后，在列表字典CI比较报告器上失败，保留来源，仅修递归比较，完整第二次审核PASS。一次控制台相对diversity摘要除以0也已明确保留为报告表达式错误，native初始及最终该项都是0，不存在合法百分比。无生产训练失败或旧结果覆写。[REPAIR_PROVENANCE.md](REPAIR_PROVENANCE.md)、[补充报告器修复记录](completion_review/REPAIR_PROVENANCE.md)。

## 对照与关键分母

两arm输入、配对初始化、train-only statistics、25304参数共享MLP、canonical hardmin4 ADE+.5FDE + 六pair平方hinge.3×.05、AdamW.001/wd.0001/batch256/25epoch/noclip相同。原bottom4固定槽位：nativearm以对应原生路径为base，CVarm以最后观察速度外推为base；其余16EXACT，最终K20，不是K24。头只读观察96D+冻结原生轨迹24D，未来GT/组别/ID/场景不forward；旧logits仅固定槽位映射，不是refined概率。

残差终点上限.20m，逐时刻.20(t/12)^2，均路径位移上限.0752314815m。保存最大端点范数.20000000000000007（double舍入）；预注册2e-15数值算术容差审核PASS，**运动学1%阈值不加epsilon**。小位置界不能限制二阶差分/jerk；此次实际结果印证“几何小改动不等于动力学安全”，但不能唯一确定机制。

nativezero完整native20；CVzero用恒速替换bottom4，保存zero_CV20。输出anchor改变初始质量与可达函数集合，比较最终native-CV不是纯学习效率/conditioning因果。尤其CV受相同小幅界约束，无法据其不佳表现否定所有恒速残差模型。

## 变化组完整好候选覆盖（同一条ADE≤.2且FDE≤.4）

| 评价集 | 变化组分母 | 原生20 | 未训练CV替换 | 训练CV（42/43/44） | 训练native（42/43/44） |
|---|---:|---:|---:|---:|---:|
| outer primary | 253 | 110 | 108 | 108 / 108 / 108 | 111 / 111 / 111 |
| inner secondary | 497 | 166 | 164 | 166 / 166 / 166 | 165 / 165 / 165 |
| 原CAL posthoc | 700 | 243 | 236 | 241 / 239 / 240 | 243 / 244 / 245 |

Outer native每seed rescue2/lost1，只净增1个窗口（+0.3953pp），coverage探索CI[-0.8890,+1.8315]pp跨0；inner每seed rescue0/lost1；CAL rescue4/lost4、3、2，净0、1、2。没有零lost保证，原生锚定也仍有rescue–lost交换。Outer CV每seed rescue0/lost2，与自身未训练CV变化覆盖完全相同；inner相对自身CV rescue2/lost0，CAL相对自身CV rescue5/3/4且lost0，却仍未补回相对native的七个lost变化窗口。不要把不同baseline的rescue/lost混为一谈。

原生完整轨迹支持大体保留，但本项没有新的显著变化覆盖收益。集合有效good簇也不抵消coverage或安全失败；全216端口×组指标在补充SUMMARY，不挑好数。

## minADE/minFDE/J：native并未实质优于原生

Outer变化组三seed均值，米（J无量纲）：

| 集合 | minADE | minFDE | J=minKmax(ADE/.2,FDE/.4) |
|---|---:|---:|---:|
| native20 | .249146619 | .429378298 | 1.390568401 |
| trained native-anchor | .249167579 | .430941976 | 1.394441346 |
| trained CV-anchor | .278093508 | .523573270 | 1.583379761 |

native相对原生ADE均值略坏.000020960m/FDE略坏.001563678m/J略坏.003872945；ADE/FDE/J探索CI全部跨0，且远不满足每项≥5%收益。native-CV连续指标CI负（更好），但native-CV完整coverage CI下界恰为0，不是严格exclude0；更重要的是native本身的native-referenceCI仍跨0。不能把“比被限制的CV好”当成“显著改善原生”。

同一Jclosest候选的outer早/中/晚ADE：native20 .119287/.271154/.374920；trainednative均值 .119871/.270439/.374542；CV .117481/.289200/.445495。CV早段略低而后段高；不拼不同候选的早中晚oracle，也不把这一结果推成decoder唯一根因/运动意图证据。

## 为什么整体好看仍不成功

Outer整体native625→CV638/637/639，nativearm628/628/628；inner932→CV968/963/967，native931/931/931；CAL1490→CV1525/1526/1529，native1493/1494/1495。CV改善主要在稳定组：outer434→448/447/449，CAL994→1035/1039/1041；CAL过渡253→249/248/248，变化243→241/239/240反而下降。CVzero整体原本outer630/inner943/CAL1494，部分优势甚至在训练前就存在；必须计入自身零基线而非全归功学习。

CV在outer整体覆盖增加的同时，整体minADE .175478→.179942、minFDE .296633→.310276、J .980767→1.010202反而更差。阈值覆盖增加和oracle尾部平均恶化可以共存。这也是不能只用overallcoverage/损失下降宣称更理解变化行为的具体反例。

训练n-weighted pre-update batch总loss，epoch1→25：native下降2.378–2.537%，CV下降5.267–5.716%。这只是四条refined路径的训练目标，不是全20set loss、不等于fullFIT posthoc覆盖；此次没有保存全18909 FIT集合的最终coverage，因此不能补造“当前FIT覆盖成功→CAL失败”的新证据。native loss下降同时outer净增1/inner丢1/运动学FAIL，仅支持“该目标与受限输出形式组合尚未交付完整集合质量”，非单独证明objective mismatch或sharedhead根因。native六pair diversity全训练历史0，报告绝对值，不将0分母百分比当成有效指标。

## 五项运动学：两arm都失败

在每seed×split×overall/stable/transition/change的72个final refined20组单元中，66个FAIL；360个物理标量中66个超过native1.01：

- native36/36组单元全部因**加速度均值**失败，+2.61265%至+3.05441%；change outer+2.99765%至+3.05441%，CAL+2.80376%至+2.86516%。其余四项未越1%不代表全部PASS。
- CV30/36组单元因**speedp95**失败，+1.29321%至+3.69380%；剩余6个PASS是outer/CAL change各3seed。CV inner change speedp95+1.29321%至+1.42380%。恒速替换也改变候选池速度组成，不能一概归因残差加速/生理不合理。[全部失败单元](completion_review/SUMMARY.json)。

原major整体FAIL；未进行“全major首次成功”才要求的进一步独立scene/sample充分性升级审核，不假称目标完成或baseline提升。没有放宽1%运动学/.2-.4完整覆盖/5%ADE AND FDE/3seed/CI/同路径标准；.3/.5只固定敏感性。

生产private峰值2.529190GiB/RSS1.859482/minphysical15.963673/commit24.497189；第二CPU补充private.272324/minphysical18.305073/commit27.594070GiB，process6GiB和两项系统各reserve6GiB通过。[RESOURCE_USAGE.json](RESOURCE_USAGE.json)、[补充REVIEW](completion_review/REVIEW.json)。

## 当前研究分支与下一步

本轮抵达的是：**原生锚点保住了大部分原生误差优势，但小幅逐点残差仍不能同时增加变化覆盖并守住运动学。** 不是scorer分支、不是“hardwinner分配已被证明不合理”分支，也不是“候选已大幅解决恒速倾向”分支。

下一项必须先精确区分：逐点位置修正的时间结构是否引入了不必要加速度，以及hardmin4对已有native20支持的训练反馈是否足够；本项尚不能在两者间作唯一归因。优先考虑可辨识、native-anchored的平滑/物理可行输出形式设计，而不是换scorer、盲试softwinner、扫bound/LR/epoch或以加大变化来求覆盖。当前仅本窄对照完成，**不自动新增loss/assignment/物理项或扩大训练/数据范围，不启动未经精确预注册的下一实验**。长期目标尚未达成；当前正式实验完成1项，后续每3项仍需总review，历史cycle11不重置。

## ARS反方检查2：分析后的挑战

Verdict PASS_WITH_LIMITATIONS（对报告诚实性，不是major结果PASS）。

1. Major：native-CV差距可能主要来自未训练anchor的原有质量与受限可达集。最强反驳：“native更好不证明模型新学到了运动规律。” 已同时保留zero_CV/own-zero及native-referenceCI，不宣称理解/新信息。
2. Major：三个seed变化组同计数不是独立人群复制，CAL和轨迹窗口相关。仅target-track×seed探索CI，不宣称独立wholemodel未见泛化；F16已见原数据。
3. Major：tiny position bound并不控制二阶差分，本项没有真正motion梯度对照。只报告观察到的acceleration harm，不判loss/head唯一根因。
4. Minor：边界CI恰触0不可称exclude0；已明示CV负coverage和native-CV边界为0。
5. 反例保留：CV稳定/整体覆盖较好、native CAL seed43/44覆盖净增、CV outer/CAL变化motion PASS；均不能抵消全项major失败。删除最强native-CV对照后，native相对原生仍未成功，结论成立。

## ARS反方检查3：最终报告挑战

Verdict PASS_WITH_LIMITATIONS（同助手，非外审）。

最强反驳：这不是原先64,032 Bo4生成器的等capacity反事实，新的25,304参数共享head/小幅可达集/zero-init也变了。故只能解释本项两arm的输出anchor对照，不能回溯否定整个generation路线。没有真实新motion观察/概率模型/scorer学会运动意图的证据。

另一关键风险是把训练四路径loss下降和全20质量对应混淆；本报告将分母/目标/未保存FIT集合限制写明，不补造FIT成功，也不使用旧结果当另一fresh对照。全部分组和失败已公开，运行/审核PASS与majorFAIL分开。本文所称独立只指保存数据的算术重构，不指外部人类或独立统计复制。

无需新数据/外部文献来解释本地结果；证据链接均指实际保存文件。未来需新loss/assignment/scorer/解冻/数据的具体设计必须另获范围许可，不能用长期自主许可绕过。停止本已完成任务的PID/task/log/resource查询，不安排旧节点每日复跑。
