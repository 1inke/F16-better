# M2结果分析：联合目标未带来可靠覆盖增量，平均加速度门仍失败

日期：2026-10-14，香港。状态：**计算执行及完整保存审计PASS；S2 FAIL；Q2 FAIL；两臂major FAIL。M2已结束，不启动M3。**

## 1. 最重要的结论

在相同预测目标初始化、相同过程解码器和训练预算下，解冻新goal输出head的B臂，没有稳定优于只训练过程的A臂。

- outer变化253窗口/68个target tracks：A覆盖93/91/110，B为101/90/101；B−A平均**−0.263505pp**，26-family校正CI **[−7.268459,+6.111781]pp**，不能声称可靠增益，也不能证明两者等价。
- 变化minADE/minFDE平均点估计分别改善约1.036%/1.235%，但校正CI都跨0；覆盖三seed仅seed42为正，稳定组保护限制也失败。
- B变化覆盖全部低于native110；B−native平均−5.006588pp，校正CI跨0，覆盖劣势尚有不确定性；但变化ADE/FDE/J的校正区间全部为正，误差相对native更差的证据明确。
- 两臂全部96个seed×split×group均失败于**accel_mean**；此外A有23组、B有21组speed_p95失败。其余accel_p95/jerk_mean/jerk_p95均通过，不能以这些好项替代全部五项。
- B确实降低了A的平均加速度代价，但outer变化组仍比native高20.85%–26.93%，远未到1%门。不能把“比A更平缓”写成“物理门已解决”。

因此不续开M3候选预算分配，不扫加速度系数、loss、degree、width/LR/epoch；baseline/historycycle11/未完成goal保持不变。[原决策](RESULT.json)、[独立复核](completion_review/REVIEW.json)。

## 2. 实验实际测试了什么

依据[共同协议](../COMMON_PROTOCOL.md)、[M2原设计](../M2_JOINT_GOAL_PROCESS.md)、[执行协议](protocol.json)。M1 S1 FAIL，故三个seed全部用对应原G1，不挑M1-A/seed44或CAL，不重训G1。

| 项目 | 固定内容 |
|---|---|
| A frozen_goal | 原G1全部参数/buffers冻结；只train7000参数process |
| B joint_goal | 同一父；仅额外train goal MLP80→64→2的5314参数＋相同process |
| 精确参数 | 两臂总33306；A可训练7000，B12314，不能称同可训练容量 |
| process | 84→64→24 GELU，zero-last；h64+query16+FIT标准化预测goal2+观测v0_2 |
| 输入边界 | P2及观测末速度；无未来GT/group/ID/scene/native/logits；P2可能已有历史social信息 |
| 数据 | FIT18909/outer1035（change253）/inner1856/CAL2879；既有target精确IDs |
| 预算 | seeds42/43/44×双臂25epoch，150轮11100更新；固定最后一轮主结果 |
| 训练 | AdamW .001/wd.0001/betas.9,.999/eps1e-8/foreachFalse；batch256/shuffle/drop_lastFalse；无clip/scheduler/早停 |
| loss | hardmin20(ADE+.5FDE)＋.05×190对平均平方hinge(.3) |
| 统计 | 每seed初始FIT全部378180目标；v0每窗口一次；std<1e-6→1；训练/CAL不重拟 |
| 旧模块 | 旧F16/原decoder/scorer未实例化或训练；immutable cache和原来源SHA保护 |

H12、dt.4、w12..1、S=650；d=(g−4.8v0)/.16，e=d−Σwb，a=b+(w/650)e，v=v0+.4cumsum(a)，p=.4cumsum(v)。g始终为预测目标，不是GT teacher。NNfloat32，几何/保存float64。B目标通过输入标准化和endpoint投影两条链均反传；context/query/centres/scalers不解冻。

投影有效过程维数22，不是24满秩。它只保证离散状态与预测终点代数，不保证a1=0、v1=v0、五motion或覆盖。无后clip、额外幅值、native修复域或新物理loss。A全部目标与初始化EXACT；端点/积分算术误差在原数值容差内，没有借此放宽科学1%门。

初始化b=0为满足预测目标的最小平方加速度P0，**不是G1直线/nativeidentity**。P0完整保存3seed×4split=12job，另24最终job。四split共24679个ID；36job合计222111行是重复测量，不是222111个独立行人。inner/CAL仅posthoc，不选checkpoint。

## 3. outer变化主结果

完整good要求同一candidate同时ADE≤.2且FDE≤.4/J≤1；不能拼两个独立最小值。以下n均253，native为同一缓存，不是三个独立native模型。

| arm | seed | complete_good | endpoint_good | minADE(m) | minFDE(m) | J | native→new rescue/lost |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 42 | 93 | 147 | 0.295649 | 0.561385 | 1.694294 | 22/39 |
| B | 42 | 101 | 148 | 0.291981 | 0.557402 | 1.678510 | 25/34 |
| A | 43 | 91 | 139 | 0.303818 | 0.577800 | 1.733682 | 16/35 |
| B | 43 | 90 | 138 | 0.297775 | 0.563675 | 1.694493 | 17/37 |
| A | 44 | 110 | 152 | 0.290743 | 0.542386 | 1.635247 | 28/28 |
| B | 44 | 101 | 150 | 0.291235 | 0.539726 | 1.643923 | 24/33 |
| native | 固定缓存 | 110 | 161 | 0.249147 | 0.429378 | 1.390568 | — |

B−A三个seed完整覆盖差为+3.162055、−.395257、−3.557312pp。B变化FDE三个seed均略好于A，ADE两个好一个差，均值均不劣条件通过；**只有这一个S2组成条件通过，其余四项失败**。

配对事件揭示B不是“保留A支撑再补洞”：

| seed | A→B rescued | lost | retained | 两臂endpoint均good窗口 | 该交集A/B complete |
|---|---:|---:|---:|---:|---|
| 42 | 12 | 4 | 89 | 134 | 93/96 |
| 43 | 8 | 9 | 82 | 125 | 88/85 |
| 44 | 5 | 14 | 96 | 137 | 106/97 |

共同endpointgood交集是训练后筛出的post-treatment人口，只作描述，不替代253主分母或因果归因。seed42该交集有3个净增益，但seed43/44为负，不能从交集中挑出成功机制。

### 3.1 预定区间与多重性

每个split/group做20000次whole(sequence,target_track)与3个model seed独立crossed重采样；A/B/native共享draw；按抽到窗口权重汇总，保存每次索引与SHA。PCG64/default_rng seed20261014。16个人口各20000draw，共16个索引文件；不是把320000个draw当独立科学样本。

M1/M2/M3预留26项family不因M3未运行缩小，尾分位.05/(2×26)，标量CI为99.8076923%；95%只描述。M2只有outer change的B−A/B−native C/ADE/FDE/J共8项primary；endpoint虽在保存数组中，**M2中仍是次级**。32个记录×5指标的160区间不是160项确认性结论，也未校正全部历史适应性实验。

| 比较 | 指标 | 差值 | 26-family CI | 普通95% CI |
|---|---|---:|---|---|
| B−A | coverage (pp) | -0.263505 | [-7.268459, 6.111781] | [-4.347826, 3.669758] |
| B−A | ADE | -0.003073 | [-0.010892, 0.005199] | [-0.007528, 0.001680] |
| B−A | FDE | -0.006923 | [-0.030305, 0.016819] | [-0.020320, 0.006363] |
| B−A | J | -0.015432 | [-0.074455, 0.043584] | [-0.049125, 0.019610] |
| B−A | endpoint_coverage (pp)，次级 | -0.263505 | [-6.445135, 5.422015] | [-3.912395, 3.114214] |
| B−native | coverage (pp) | -5.006588 | [-16.160145, 4.115204] | [-11.517741, 0.887199] |
| B−native | ADE | 0.044517 | [0.016439, 0.084064] | [0.025656, 0.067035] |
| B−native | FDE | 0.124223 | [0.043117, 0.235859] | [0.068974, 0.190034] |
| B−native | J | 0.281740 | [0.111317, 0.510360] | [0.165685, 0.419027] |
| B−native | endpoint_coverage (pp)，次级 | -6.192358 | [-17.523823, 3.388210] | [-12.878983, 0.000000] |

B−A全部预定主区间跨0，不叫显著改善/有害/等价。B−native变化误差校正CI为：ADE[+.016439,+.084064]m、FDE[+.043117,+.235859]m、J[+.111317,+.510360]；这些正区间不支持新候选优于native。

### 3.2 保护组不能省略

B相对A在outer stable超1%的项：

| seed | group | 指标 | harm |
|---|---|---|---:|
| 42 | stable | ADE | +1.057374% |
| 42 | stable | FDE | +2.288672% |
| 43 | stable | ADE | +2.737546% |
| 43 | stable | FDE | +5.292439% |
| 44 | stable | FDE | +2.500288% |

B相对native的outer overall ADE每seed约+2.23%–4.83%，FDE约+6.14%–9.51%；但stable比native好，是必须保留的正反例。不能只选变化组的一个seed，忽略保护组或只报告overall覆盖。

## 4. P0、FIT与迁移：学习收益没有稳定转到outer

变化完整覆盖，seed顺序42/43/44：

| split | n | native | 未训练P0 | A final | B final |
|---|---:|---:|---|---|---|
| FIT | 4725 | 1937 | 2074/2099/2105 | 2196/2263/2289 | 2334/2345/2384 |
| outer | 253 | 110 | 100/91/109 | 93/91/110 | 101/90/101 |
| inner | 497 | 166 | 170/173/175 | 169/167/171 | 163/169/182 |
| original_CAL | 700 | 243 | 203/189/188 | 204/184/214 | 212/190/206 |

重要反例：

1. P0 outer变化100/91/109已接近A final93/91/110；训练A相对P0净−7/0/+1。B相对P0仅+1/−1/−8，不支持“过程训练越久就会救回覆盖”。
2. P0在全部12个change seed×split组合五motion均通过；全48组中31通过，17组只有speed_p95失败。P0仍**不是全安全新baseline**，不能事后选择它当通关方案。
3. A的P0→final outer变化rescue/lost分别4/11、6/6、8/7；B为12/11、10/11、10/18。训练既救回也丢失已有支撑。[全事件](completion_review/ATTRIBUTION.json)。
4. B在FIT变化2334/2345/2384，全部高于A和native1937；FIT终点也增至3236/3268/3254。输入、梯度和训练确实可被利用，不是“完全学不动”。但outer B仍90–101<C110、CAL B190–212<C243。
5. inner B变化163/169/182，其中后两seed超过native166；CAL42/43 B比A好、seed44变差。不能把所有人口概括成同一个方向，更不能独证CAL分布差异是唯一根因。

六run训练batch加权目标均下降：A约.31096/.30903/.30838→.30430/.30253/.30149，B约.30838/.30567/.30569→.29315/.29136/.29050。这是训练期preupdate batch量，不是全FIT覆盖，更不是泛化证明。

A相对历史G1直线的outer变化77/72/82提高到93/91/110，但P0已经提高到100/91/109；不能把全部历史改善归功于训练。旧G2桥和本M2还改变了过程参数化、容量和初始化，不能当严格单因素配对来解释哪一个积分方程有因果优势。

## 5. 运动学：降低代价不等于满足门

原科学口径速度12段、加速度11差分、jerk10差分，ALLK20/全部窗口时间的pooled量，不先求每窗口p95再平均。literal new≤native1.01，无评价epsilon/floor。v0→首未来速度跳变另报，不混进旧门。

| arm | 预定seed/split/group | 五motion全PASS | accel_mean FAIL | speed_p95 FAIL | 其余三项FAIL |
|---|---:|---:|---:|---:|---:|
| A | 48 | 0 | 48 | 23 | 0 |
| B | 48 | 0 | 48 | 21 | 0 |

A全部组accel_mean相对native约+25.20%–60.23%，B约+3.26%–27.26%；B改善明显但最低仍超过1%门。没有用较低jerk或accelp95掩盖均值问题。这是相对预注册native的运动学门失败，不等于已经证明所有这些路径生理上不可能。

outer变化的绝对值：

| arm | seed | accel_mean(m/s²) | 对native harm | jerk_mean(m/s³) | 首future-v0跳变均值(m/s) |
|---|---:|---:|---:|---:|---:|
| A | 42 | 0.132615 | 58.485494% | 0.150646 | 0.033466 |
| B | 42 | 0.106209 | 26.928184% | 0.088174 | 0.033442 |
| A | 43 | 0.132645 | 58.520837% | 0.151114 | 0.030869 |
| B | 43 | 0.101120 | 20.845694% | 0.083343 | 0.033206 |
| A | 44 | 0.127867 | 52.810770% | 0.139589 | 0.031985 |
| B | 44 | 0.104899 | 25.361920% | 0.083003 | 0.032177 |
| native | 固定缓存 | .083677 | — | .460219 | 不作该额外边界比较 |

B减少了A的过程修正强度，仍未安全通过；边界jump也没有稳定明显下降。不能以首未来速度更接近v0来代替平均加速度门。

### 5.1 目标与零终点过程能量分解

以下对**所有candidate**汇总，不只挑GT winner。分解能量为12步加速度分量的均方模，单位(m/s²)²；它是诊断，不是原motion门或新增loss。

| arm | seed | 目标距初始均值(m) | 目标距CV均值(m) | 最小平方项能量 | 零终点项能量 |
|---|---:|---:|---:|---:|---:|
| A | 42 | 0.000000 | 1.382980 | 0.013366 | 0.018760 |
| B | 42 | 0.142431 | 1.462824 | 0.014953 | 0.005887 |
| A | 43 | 0.000000 | 1.308144 | 0.011975 | 0.020242 |
| B | 43 | 0.133493 | 1.398427 | 0.013748 | 0.005399 |
| A | 44 | 0.000000 | 1.353275 | 0.012520 | 0.016955 |
| B | 44 | 0.139892 | 1.435854 | 0.014200 | 0.005538 |

B并非简单把所有终点缩回CV：outer变化的目标−CV均值比A增加，而零终点项能量显著减小（A约.01696–.02024，B约.00540–.00589）。这支持“联合优化重新分配了目标与过程修正”的描述，不是纯目标学习因果证明；B还有5314额外可训练参数，过程和责任也同时变化。原11加速度与新增包含边界的12加速度/11jerk均单列，绝不混用分母。

## 6. 候选完整性与时段解释

outer变化完整路径有效簇A约19.94–19.99、B约19.94–20；endpoint-only .20m聚簇均约19.94–20，没有明显全20候选塌成复制。B有效good簇均值约.478–.498，不能因为簇多就宣称有用。

原GTfree全K complete-link D=max(meanpath/.10,endpoint/.20)cut1保持；endpoint-only描述统计不改变部署或选slot规则。nearest/median/p95终点距离、全部Q与Jslot使用直方图、raw/projected加速度jerk及同slot早中晚指标见[补充](completion_review/SUPPLEMENT.json)。

独立minADE/minFDE若拼接会在outer变化虚增A的10/7/5和B的6/7/9个good，本文全部使用同一完整candidate。Jclosest与Qwinner不同，不拼分段oracle。例如seed43，A的Jclosest早/中/晚约.12247/.32421/.49180m，B约.12045/.31439/.48210m；这是GT后验选定的同一slot的解释，不是部署选择器或心理意图证据。

## 7. 执行、保存审核、资源与来源

唯一worker10364，start2026-10-14 20:26:20.384351，worker/exit双SUCCESS于20:41:48（worker20:41:48.062001、exit20:41:48.047848）。worker elapsed927.677650秒，RESULT内部923.181254秒，起点不同。唯一半程actual20:52:47.5983068，scheduled20:51:50.068，迟57.530307秒，发现已完成；taskReady/processnone/XML0/stderr0/六25history。**没有补startup/half/completion查询或重训**，旧21:17完成节点作废。

完整24final＋12P0 jobs/6checkpoint/150history/131科学输出SHA/3941相关锁源；ARITHMETIC与PRESERVATION PASS。

- audit_saved.py只执行一次，179.309秒：完整36 saved质量、原SciPy ALLK聚簇、五motion、独立几何、逐ID/native/cache对齐、父冻结合同/统计/六checkpoint完整optimizer及RNG/150history、20000逐draw SHA及独立track聚合/26-family/gates、source/output初末SHA PASS。几何复算最大1.24345e-14。
- 另一个CPU saved-only补充只执行一次，114.545秒：36job/144组、endpoint聚簇和距离、Q/J同路径时段/usage、raw/projected/minimum/null与边界等全量复算，source-output初末SHA PASS；最大endpoint误差4.44089e-15。未import torch或新forward/optimizer。
- CPU加载checkpoint只核对合同，不是全部神经模型/梯度二次重放；同算法确定性CI重构不是外部独立科学评审。两项审核的墙钟阶段有重叠，不能相加冒充整个任务elapsed。

| 阶段 | private峰值GiB | RSS峰值GiB | physical最低GiB | commit最低GiB |
|---|---:|---:|---:|---:|
| production | 4.49080 | 3.63323 | 10.39553 | 15.05056 |
| CPU audit | 2.18435 | 1.78782 | 10.51504 | 15.92822 |
| CPU supplement | .95603 | .97225 | 10.20381 | 15.62833 |

全部process6GiB、physical/commit各reserve6保持；无OS/pagefile/用户程序干预。M2约51min规划经人类单项“批准”，实际较快不自动扩大M3/V预算。

旧3930快照/原PREFLIGHT预算STOP保留；仅final_source_review.py/start_independent.ps1两项授权预算控制改变，其余3928 EXACT。当前3941源SHA eaf8fb1a87771dbd6cd308d1abb4975f5c56983f7f65a4d5656a1c277f13feea未变；包括M1原3746来源和107输出及扩展，不是全部历史34k/旧F16GPU回放。旧科学、模型、结果和失败全部保留。

一次分析读取大JSON返回被截断，读取守卫拒绝解析后改成紧凑字段读取；无项目写入或科学修改，不属于production失败，也没有重跑审核/训练。

## 8. ARS反方核查与结论边界

### DA2（同助手inline，非外审）

最强反驳：B额外解冻5314个参数，可改变终点、responsibility和过程；即使某项变好，也不能纯归因为“目标—过程协调”或更理解运动。因此保留全部P0、目标变化与零终点能量，不能只看一个coverage。这里B−A的主CI跨0，不能把没有通关写成方法等价或所有联合模型不可能。

另一个反例：B在FIT coverage/目标拟合改善、inner后两seed较好、A44与native完整覆盖相同，说明存在局部可学习性；停止是预定整体门未满足，不是宣称神经网络完全没学到信息。

### DA3（同助手inline，非外审）

当前3seed、旧target划分、只有3scene、target不等于context隔离、F16已见原数据、反复看outer/CAL与自适应方案历史限制保留；Bonferroni26-family不覆盖全部历史探索。不能声称独立whole-model泛化/SOTA、真实心理意图、不可预测率或唯一loss/decoder根因。

不把平均加速度门解释成所有人类运动的充分标准；但本轮注册的五项必须AND通过，不能因其他运动学值更低而事后放宽1%。没有科学数值/source/ID失败来解释这一阴性结果；保存审核PASS与科学FAIL同时成立。

本报告由AI助手使用academic-research-suite组织；不是外部三位审稿人或独立实验室验证。

## 9. M1＋M2整体判断与下一步

M1固定动态历史GRU没有可靠增量，M2允许联合调整目标仍没有稳定覆盖收益。两者不支持“再添历史分支或多训goalhead就能解决”的简单路径。M2 FIT成功/outer-CAL不足与训练后平均加速度增加，共同提示当前固定输入、目标和有限训练下，质量/迁移/过程代价的联合约束尚未解决；不能唯一归因为缺数据、scorer、loss或某一个模块。

**按原设计，M3不启动**：S2未过，B每seed变化覆盖均低于native，保护与48组运动学也未过。V验证只给可信幸存分支；没有Q2合格分支，本轮也不启动。M3/V标记“条件未触发”，不是“已完成实验”。M1/M2共两阶段已完成，whole_series_complete仍false，baseline/historycycle11/goal不假完成。

停止这一固定S650 decoder方案，不增加loss penalty/加速度系数/degree/width/LR/epoch或换seed来求PASS；不恢复旧G2桥或R1–R5局部探针。本次m2过期followup在报告后移除，不每日重复旧节点、不查询completed10364。若未来另开方向，须有新的明确研究设计与可辨识对照；本轮不自动启动。

## 附录：最终全部96组结果

五motion一列列出全部失败项；n是窗口数，彼此相关，不是96个独立实验。

| arm | seed | split | group | n | complete | endpoint | minADE(m) | minFDE(m) | J | failed_motion |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| A | 42 | FIT | overall | 18909 | 12516 | 15341 | 0.158003 | 0.272042 | 0.903196 | speed_p95,accel_mean |
| A | 42 | FIT | stable | 10767 | 8439 | 9661 | 0.113583 | 0.185999 | 0.644164 | speed_p95,accel_mean |
| A | 42 | FIT | transition | 3417 | 1881 | 2588 | 0.193120 | 0.335204 | 1.113899 | accel_mean |
| A | 42 | FIT | change | 4725 | 2196 | 3092 | 0.233828 | 0.422434 | 1.341083 | accel_mean |
| A | 42 | outer | overall | 1035 | 634 | 801 | 0.181641 | 0.316614 | 1.035289 | speed_p95,accel_mean |
| A | 42 | outer | stable | 604 | 452 | 529 | 0.126442 | 0.204547 | 0.718438 | speed_p95,accel_mean |
| A | 42 | outer | transition | 178 | 89 | 125 | 0.206901 | 0.348981 | 1.173773 | accel_mean |
| A | 42 | outer | change | 253 | 93 | 147 | 0.295649 | 0.561385 | 1.694294 | accel_mean |
| A | 42 | inner | overall | 1856 | 1013 | 1298 | 0.202710 | 0.369377 | 1.164782 | speed_p95,accel_mean |
| A | 42 | inner | stable | 978 | 664 | 793 | 0.147897 | 0.262014 | 0.843852 | accel_mean |
| A | 42 | inner | transition | 381 | 180 | 257 | 0.214327 | 0.385426 | 1.245773 | speed_p95,accel_mean |
| A | 42 | inner | change | 497 | 169 | 248 | 0.301667 | 0.568342 | 1.734225 | accel_mean |
| A | 42 | original_CAL | overall | 2879 | 1521 | 2055 | 0.189115 | 0.327140 | 1.074963 | accel_mean |
| A | 42 | original_CAL | stable | 1535 | 1049 | 1292 | 0.125333 | 0.205853 | 0.708681 | accel_mean |
| A | 42 | original_CAL | transition | 644 | 268 | 414 | 0.238969 | 0.412202 | 1.358228 | accel_mean |
| A | 42 | original_CAL | change | 700 | 204 | 349 | 0.283115 | 0.514846 | 1.617561 | accel_mean |
| B | 42 | FIT | overall | 18909 | 12795 | 15591 | 0.155029 | 0.256992 | 0.872003 | speed_p95,accel_mean |
| B | 42 | FIT | stable | 10767 | 8527 | 9738 | 0.112811 | 0.180762 | 0.632524 | speed_p95,accel_mean |
| B | 42 | FIT | transition | 3417 | 1934 | 2617 | 0.189746 | 0.316731 | 1.073496 | accel_mean |
| B | 42 | FIT | change | 4725 | 2334 | 3236 | 0.226126 | 0.387499 | 1.271996 | accel_mean |
| B | 42 | outer | overall | 1035 | 632 | 798 | 0.181362 | 0.317089 | 1.033079 | speed_p95,accel_mean |
| B | 42 | outer | stable | 604 | 442 | 523 | 0.127779 | 0.209228 | 0.724215 | speed_p95,accel_mean |
| B | 42 | outer | transition | 178 | 89 | 127 | 0.205956 | 0.341521 | 1.163755 | accel_mean |
| B | 42 | outer | change | 253 | 101 | 148 | 0.291981 | 0.557402 | 1.678510 | accel_mean |
| B | 42 | inner | overall | 1856 | 1020 | 1288 | 0.202275 | 0.371356 | 1.157037 | speed_p95,accel_mean |
| B | 42 | inner | stable | 978 | 673 | 784 | 0.148557 | 0.266838 | 0.842104 | speed_p95,accel_mean |
| B | 42 | inner | transition | 381 | 184 | 256 | 0.213847 | 0.390806 | 1.239183 | speed_p95,accel_mean |
| B | 42 | inner | change | 497 | 163 | 248 | 0.299108 | 0.562117 | 1.713790 | accel_mean |
| B | 42 | original_CAL | overall | 2879 | 1499 | 2013 | 0.189579 | 0.327282 | 1.069389 | accel_mean |
| B | 42 | original_CAL | stable | 1535 | 1034 | 1271 | 0.126934 | 0.210729 | 0.713667 | speed_p95,accel_mean |
| B | 42 | original_CAL | transition | 644 | 253 | 400 | 0.240468 | 0.411938 | 1.353478 | accel_mean |
| B | 42 | original_CAL | change | 700 | 212 | 342 | 0.280132 | 0.504983 | 1.588076 | accel_mean |
| A | 43 | FIT | overall | 18909 | 12638 | 15354 | 0.157610 | 0.269117 | 0.897753 | speed_p95,accel_mean |
| A | 43 | FIT | stable | 10767 | 8496 | 9710 | 0.113102 | 0.183583 | 0.640177 | speed_p95,accel_mean |
| A | 43 | FIT | transition | 3417 | 1879 | 2526 | 0.192825 | 0.334701 | 1.108696 | accel_mean |
| A | 43 | FIT | change | 4725 | 2263 | 3118 | 0.233563 | 0.416599 | 1.332151 | accel_mean |
| A | 43 | outer | overall | 1035 | 631 | 788 | 0.183445 | 0.322873 | 1.042779 | speed_p95,accel_mean |
| A | 43 | outer | stable | 604 | 456 | 533 | 0.124741 | 0.203219 | 0.709035 | speed_p95,accel_mean |
| A | 43 | outer | transition | 178 | 84 | 116 | 0.211553 | 0.366547 | 1.193247 | accel_mean |
| A | 43 | outer | change | 253 | 91 | 139 | 0.303818 | 0.577800 | 1.733682 | accel_mean |
| A | 43 | inner | overall | 1856 | 1021 | 1322 | 0.202105 | 0.367046 | 1.163536 | accel_mean |
| A | 43 | inner | stable | 978 | 665 | 806 | 0.146321 | 0.253225 | 0.833350 | accel_mean |
| A | 43 | inner | transition | 381 | 189 | 268 | 0.213181 | 0.375313 | 1.230631 | speed_p95,accel_mean |
| A | 43 | inner | change | 497 | 167 | 248 | 0.303387 | 0.584688 | 1.761845 | accel_mean |
| A | 43 | original_CAL | overall | 2879 | 1482 | 2038 | 0.189731 | 0.326808 | 1.073684 | accel_mean |
| A | 43 | original_CAL | stable | 1535 | 1043 | 1294 | 0.125266 | 0.203148 | 0.708118 | speed_p95,accel_mean |
| A | 43 | original_CAL | transition | 644 | 255 | 406 | 0.238132 | 0.408185 | 1.335866 | accel_mean |
| A | 43 | original_CAL | change | 700 | 184 | 338 | 0.286565 | 0.523112 | 1.634109 | accel_mean |
| B | 43 | FIT | overall | 18909 | 12833 | 15646 | 0.154650 | 0.254229 | 0.867040 | speed_p95,accel_mean |
| B | 43 | FIT | stable | 10767 | 8544 | 9763 | 0.112477 | 0.179134 | 0.630013 | speed_p95,accel_mean |
| B | 43 | FIT | transition | 3417 | 1944 | 2615 | 0.188901 | 0.313775 | 1.066768 | accel_mean |
| B | 43 | FIT | change | 4725 | 2345 | 3268 | 0.225981 | 0.382288 | 1.262722 | accel_mean |
| B | 43 | outer | overall | 1035 | 619 | 782 | 0.183946 | 0.324837 | 1.045602 | speed_p95,accel_mean |
| B | 43 | outer | stable | 604 | 444 | 524 | 0.128155 | 0.213974 | 0.729572 | accel_mean |
| B | 43 | outer | transition | 178 | 85 | 120 | 0.211466 | 0.361552 | 1.195676 | accel_mean |
| B | 43 | outer | change | 253 | 90 | 138 | 0.297775 | 0.563675 | 1.694493 | accel_mean |
| B | 43 | inner | overall | 1856 | 1008 | 1318 | 0.202630 | 0.363341 | 1.149959 | accel_mean |
| B | 43 | inner | stable | 978 | 651 | 796 | 0.149380 | 0.260724 | 0.845580 | accel_mean |
| B | 43 | inner | transition | 381 | 188 | 267 | 0.213603 | 0.372970 | 1.214534 | accel_mean |
| B | 43 | inner | change | 497 | 169 | 255 | 0.299005 | 0.557890 | 1.699413 | accel_mean |
| B | 43 | original_CAL | overall | 2879 | 1485 | 2009 | 0.190335 | 0.326328 | 1.070348 | accel_mean |
| B | 43 | original_CAL | stable | 1535 | 1038 | 1285 | 0.127460 | 0.207654 | 0.715258 | speed_p95,accel_mean |
| B | 43 | original_CAL | transition | 644 | 257 | 394 | 0.239187 | 0.410248 | 1.338791 | accel_mean |
| B | 43 | original_CAL | change | 700 | 190 | 330 | 0.283266 | 0.509359 | 1.602043 | accel_mean |
| A | 44 | FIT | overall | 18909 | 12679 | 15379 | 0.156899 | 0.269111 | 0.894375 | speed_p95,accel_mean |
| A | 44 | FIT | stable | 10767 | 8522 | 9732 | 0.112527 | 0.184230 | 0.636666 | speed_p95,accel_mean |
| A | 44 | FIT | transition | 3417 | 1868 | 2519 | 0.191622 | 0.331708 | 1.101686 | speed_p95,accel_mean |
| A | 44 | FIT | change | 4725 | 2289 | 3128 | 0.232898 | 0.417262 | 1.331702 | accel_mean |
| A | 44 | outer | overall | 1035 | 645 | 811 | 0.179593 | 0.314611 | 1.016353 | speed_p95,accel_mean |
| A | 44 | outer | stable | 604 | 450 | 538 | 0.124967 | 0.204187 | 0.704537 | speed_p95,accel_mean |
| A | 44 | outer | transition | 178 | 85 | 121 | 0.206970 | 0.365562 | 1.194759 | accel_mean |
| A | 44 | outer | change | 253 | 110 | 152 | 0.290743 | 0.542386 | 1.635247 | accel_mean |
| A | 44 | inner | overall | 1856 | 1017 | 1330 | 0.201282 | 0.361313 | 1.147050 | speed_p95,accel_mean |
| A | 44 | inner | stable | 978 | 655 | 806 | 0.146272 | 0.251609 | 0.828251 | speed_p95,accel_mean |
| A | 44 | inner | transition | 381 | 191 | 266 | 0.213229 | 0.381082 | 1.234242 | speed_p95,accel_mean |
| A | 44 | inner | change | 497 | 171 | 258 | 0.300374 | 0.562035 | 1.707542 | accel_mean |
| A | 44 | original_CAL | overall | 2879 | 1508 | 2069 | 0.188894 | 0.325036 | 1.074030 | speed_p95,accel_mean |
| A | 44 | original_CAL | stable | 1535 | 1020 | 1285 | 0.126282 | 0.206258 | 0.717104 | speed_p95,accel_mean |
| A | 44 | original_CAL | transition | 644 | 274 | 426 | 0.235067 | 0.399419 | 1.329320 | speed_p95,accel_mean |
| A | 44 | original_CAL | change | 700 | 214 | 358 | 0.283716 | 0.517068 | 1.621851 | accel_mean |
| B | 44 | FIT | overall | 18909 | 12886 | 15627 | 0.153578 | 0.253517 | 0.862650 | speed_p95,accel_mean |
| B | 44 | FIT | stable | 10767 | 8559 | 9758 | 0.111355 | 0.178013 | 0.623539 | speed_p95,accel_mean |
| B | 44 | FIT | transition | 3417 | 1943 | 2615 | 0.187901 | 0.315372 | 1.067135 | accel_mean |
| B | 44 | FIT | change | 4725 | 2384 | 3254 | 0.224973 | 0.380835 | 1.259640 | accel_mean |
| B | 44 | outer | overall | 1035 | 643 | 807 | 0.179395 | 0.314851 | 1.020478 | speed_p95,accel_mean |
| B | 44 | outer | stable | 604 | 451 | 530 | 0.124581 | 0.209293 | 0.708807 | speed_p95,accel_mean |
| B | 44 | outer | transition | 178 | 91 | 127 | 0.206431 | 0.353411 | 1.191927 | accel_mean |
| B | 44 | outer | change | 253 | 101 | 150 | 0.291235 | 0.539726 | 1.643923 | accel_mean |
| B | 44 | inner | overall | 1856 | 1035 | 1338 | 0.200848 | 0.359225 | 1.138418 | speed_p95,accel_mean |
| B | 44 | inner | stable | 978 | 663 | 809 | 0.146904 | 0.257708 | 0.835568 | accel_mean |
| B | 44 | inner | transition | 381 | 190 | 265 | 0.212317 | 0.372622 | 1.217294 | speed_p95,accel_mean |
| B | 44 | inner | change | 497 | 182 | 264 | 0.298207 | 0.548721 | 1.673904 | accel_mean |
| B | 44 | original_CAL | overall | 2879 | 1510 | 2033 | 0.189387 | 0.327250 | 1.069755 | speed_p95,accel_mean |
| B | 44 | original_CAL | stable | 1535 | 1033 | 1283 | 0.127916 | 0.210267 | 0.722575 | speed_p95,accel_mean |
| B | 44 | original_CAL | transition | 644 | 271 | 406 | 0.235717 | 0.407908 | 1.330394 | speed_p95,accel_mean |
| B | 44 | original_CAL | change | 700 | 206 | 344 | 0.281560 | 0.509571 | 1.591284 | accel_mean |

P0全部48组、全部运动学绝对值及pool分母见RESULT.initial_records；本文没有只选变化组把P0称为全安全。最终报告证据：[RESULT](RESULT.json)、[BOOTSTRAP](BOOTSTRAP.json)、[保存审核](completion_review/REVIEW.json)、[事件归因](completion_review/ATTRIBUTION.json)、[CPU补充](completion_review/SUPPLEMENT.json)、[单项预算授权](BUDGET_EXCEPTION.json)。
