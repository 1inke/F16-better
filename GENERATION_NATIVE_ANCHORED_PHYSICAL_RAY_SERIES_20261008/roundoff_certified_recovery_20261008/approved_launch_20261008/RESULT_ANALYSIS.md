# 原生锚定全路径物理-ray：第三项完整结果

结论：运行与独立保存结果审核均PASS，研究目标major FAIL。物理ray让全部五项运动学保护通过，却把输出压回几乎原生：严格完整覆盖没有新增救回、也没有丢失。安全性恢复，不等于安全互补生成已经成立。原生baseline、历史cycle11及长期研究目标不变。

证据：[原始结果](C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_PHYSICAL_RAY_SERIES_20261008/roundoff_certified_recovery_20261008/approved_launch_20261008/RESULT.json)、[独立审核](C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_PHYSICAL_RAY_SERIES_20261008/roundoff_certified_recovery_20261008/approved_launch_20261008/completion_review/REVIEW.json)、[全部72组别与36探索CI](C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_PHYSICAL_RAY_SERIES_20261008/roundoff_certified_recovery_20261008/approved_launch_20261008/completion_review/REPORT_DATA.json)、[生产资源](C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_PHYSICAL_RAY_SERIES_20261008/roundoff_certified_recovery_20261008/approved_launch_20261008/RESOURCE_USAGE.json)、[协议](C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_PHYSICAL_RAY_SERIES_20261008/roundoff_certified_recovery_20261008/approved_launch_20261008/protocol.json)。采用academic-research-suite证据核验与反方检查，由同一AI助手完成，不是外部独立审稿。

## 执行、来源与复核

唯一worker42092从09:32:13.080314运行到09:36:45.282905，双SUCCESS，实际4分32.202591秒。唯一半程核验09:37:33.3125873，较计划09:36:13.080314迟80.2322733秒，发现任务Ready/无进程/零trigger/新stderr0。没有冒称按时半程，也没有另查已完成进程。

六个fresh训练各25epoch/150行，六checkpoint及完整source/IDs/statistics/objective/optimizer-RNG绑定、18科学保存输出/103输出SHA完整。生产ARITHMETIC_AUDIT/PRESERVATION PASS；CPU补充重新计算全部18保存质量、原SciPy全K20 complete-link聚簇、五项pooled运动学、assignment、固定16/bounds/Bcontrols/alpha/36bootstrap144标量CI，质量/CI最大误差0。1193相关来源初末SHA不变，继承1166原digest；不是所有历史34k文件或所有模型/梯度再次回放。旧F16/decoder/scorer未实例化训练或访问optimizer，immutable cache及SHA保护不能冒称旧模型GPU完整冻结重放。

实际生产private峰值2.792027GiB/RSS2.101807GiB，physical最低15.033325/commit24.150532GiB。独立CPU审核private .305504GiB，elapsed52.279秒。进程6GiB及physical/commit各6GiB余量均通过，未调整OS/pagefile或关闭用户程序。

## 同候选完整覆盖

主标准始终是一条完整candidate同时ADE≤.2m AND FDE≤.4m，J=minK max(ADE/.2,FDE/.4)。各自minADE和minFDE可能来自不同slot，不能拼成成功；早中晚也只用一个Jclosest槽位。

| split/变化组 | N | native good | 同Bernstein不投影，42/43/44 | 训练内physical-ray，42/43/44 |
|---|---:|---:|---|---|
| outer，主结果固定epoch25 |253|110|111/111/111|110/110/110|
| inner，事后 |497|166|165/165/165|166/166/166|
| original CAL，事后 |700|243|243/244/245|243/243/243|

不投影outer每seed救2/损1，inner救0/损1，CAL救4/损4、3、2。物理组所有split、所有overall/stable/transition/change、所有seed均救0/损0，good计数与native相同。overall物理组outer625、inner932、CAL1490均不变；不投影分别628、931、1493/1494/1495。不能只报告加速度保护而省略消失的救回。

物理outer change三seed均值ADE=.24914662694m、FDE=.42937832126m、J=1.39056846563；native为.24914661896/.42937829771/1.39056840094。差值仅约+7.97e-9m/+2.35e-8m/+6.47e-8，不是5%双误差收益。对应探索CI均跨0，coverage差0及CI[0,0]来自保存二元集合相同，不证明有用方法的总体等效。物理对不投影outer change coverage差−.3953pp、CI[−1.9231,+.8334]pp；连续差CI也跨0。全36CI和受保护组在REPORT_DATA保留，不用change一列替代全组保护。

物理change有效good簇outer .50592885、inner .39637827、CAL .43285714，各seed一致；这没有补出coverage hole。所有详细distinct/Jclosest早中晚及.3/.5固定敏感性列已保存，不最大化簇数，也不改变阈值。

## 五运动学与近原生收缩

不投影36组全部FAIL，唯一失败项accelmean，比native高2.66488–3.07252%；其他四项通过。physical36组全部通过原始literal五项1%保护。控制9份输出11个原有数组与上项Bernstein治疗bitwise EXACT，是fresh重复，不是九次新独立泛化。

但physical的alpha中位数在九输出均为0；零alpha比例53.76–56.82%，alpha均值约3.55e-6–4.04e-5。所有保存路径的最大实际残差4.1840e-5m，即约0.04184毫米；最大alpha约.003085。未投影提议仍可达到厘米级，不能说参数一直停在零初始化；投影后几乎无可用修正是当前实际结果。

训练四路径n加权batch total下降：控制2.377–2.535%，physical仅约.00000320–.00001694%；diversity绝对0，百分比未定义。这是batch-preupdate四候选训练分母，不是全K20人口loss；未保存完整FIT set coverage，不能虚构本实验“FIT成功而CAL失败”的新证据。

alpha是31个逐步native相对速度/加速度/jerk条件的共同射线交点，比原五项pooled gate更强。native接近恒速、尤其零或极小加速度/jerk预算时，整条提议可以被压到零。这支持“本ray可行几何过窄”的工程解释，不证明所有物理可行补充不存在、训练目标唯一错误或共享head唯一根因。

## 数值诚实性

生产forward/alpha保持父版本bitwise EXACT，无alpha0 nativefallback。只有独立pointwise审核采用推导的浮点前向误差certificate。九physical输出累计11个raw严格不等式正超额（含重复window/seed，不是11个独立行人），最大2.63193e-15，最高certificate比例.0022514；全部在证书内。原五项pooled研究门槛仍literal≤1.01native、无epsilon/floor。这不是“所有raw逐点不等式bitwise全PASS”。

原3.8646e-17失败、被拒fallback、较大FD步长1e-4/1e-5 face切换及1e-6/1e-7收敛证据均保留。普通浮点可信度与工程修复由助手负责；不再让人类决定容差。不能把初始FD=0或小步长收敛说成所有较大步长/nonzero两eps通过。

## ARS反方检查2（同助手，非外审）

PASS_FOR_LIMITED_REPORT。最强反驳：严格逐候选逐时间native-relative预算远强于接受标准；“全部运动学通过但覆盖不增”可能只是过强可行域，而不是生成线失败。已明确保留该解释，禁止据此否定所有生成或宣布head/loss根因。另一major风险是把重复freshcontrol、CAL复用和浮点级差值当独立泛化/有效改进，报告不作此声称。

## ARS反方检查3（同助手，非外审）

PASS_FOR_EXPLORATORY_NEGATIVE_RESULT。原门槛、每seed/全部组和反例不删，未用certificate放宽研究motion；执行PASS与majorFAIL分开。target-disjoint非context-disjoint、F16已见、3scene/seed、相关窗口、重复CAL/adaptivebottom4限制保留，不能声称理解心理意图/SOTA/独立wholemodel泛化。安全近identity不是互补突破。

## 下一步边界

第三全审核后native-anchor系列3/3，必须先完成全三项及近期整体复盘。停止第四局部alpha/margin/degree/bound/width/LR/epoch扫描，不进入scorer或盲换softwinner。下一大问题是：怎样让生成器直接产生非退化的物理可行互补方向，而不是先生成大提议再沿原生相对零预算射线全部缩掉。此为下一设计方向，不等于本轮已批准新loss/旧解冻/数据或新第四训练。

原六小时03:30:18–09:30:18窗口已结束；人类最新“开始”仅明确延续本第三项及审计/复盘，不重置六小时。完成本归因闭环后结束本次单次自动跟进，长期研究目标仍未达成。

