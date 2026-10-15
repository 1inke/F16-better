# 方向2结果分析：原生支撑保留的K20整合

状态：ANALYZED。执行与保存数据审核PASS；科学目标MAJOR_FAIL。四方向完成2/4，整套未完成。此项是NONFORMAL冻结保存输出诊断，不是训练。

## 结论

安全政策接受了 **0个新候选**，全部18组最终policy20数组与native20逐元素完全相同。覆盖、ADE/FDE/J、有效好簇、所有五项运动学都等于原生值。没有突破，也不升级baseline。

主要瓶颈不是单一“logit排名差”：固定几何标准下多数原生路径并不冗余；剩余可替换尝试又被实际完整候选集运动学拒绝。不能据此断言所有生成器或整合方法不可能有效。新增4路径的K24有限bank有少量覆盖见证，但候选预算不同，而且全部72组五运动学联合门失败。

## 设计和复核

冻结方向1的position3/acceleration3 × seed42/43/44 × outer1035/inner1856/originalCAL2879，共18保存jobs、34,620重复行。重复跨模型/arm的行不是独立样本。不运行Torch、模型前向、backward或optimizer。

固定GT-free政策只接收native20/refined4。native20原SciPy complete-link距离D=max(meanpath/.10,endpoint/.20)，cut1，每簇最低physical slot代表永久保留。新候选0→3各试一次，只考虑仍在集合里的非代表原生槽，选最小到代表距离，必须满足严格新颖性和完整K20单window全部五motion≤immutable nativewindow×1.01。无另槽重试、强制插入、阈值/系数搜索。保留原生几何证书不等于严格GT覆盖保护；window分位数门也不等于population分位数门。

学习路径来源仍是旧bottom-logit4；政策函数不使用GT/groups/logits/ID/scene。K24=原20+新4仅不等预算参考。Primary outer fixed25；inner/CAL均posthoc，未用于选择。

生产双SUCCESS：00:48:41.200866→00:59:33.716976 HK，10分52.516秒。唯一完成检查01:04:33.1053092确认Ready/无进程/0trigger/新stderr0。既有half count1，不再次查询已完成PID33260。

CPU完整复核18jobs、91输出SHA、1614相关source初末SHA、两独立政策决策循环、全部原SciPy质量聚簇/运动学/证书/分解、48bootstrap记录192scalar CI：**全部差0**，311.811秒。新增保存事件归因9.403秒，初末SHA PASS。不是所有历史34k文件审核、模型/梯度第二次回放或外部评审。

来源：[RESULT](RESULT.json)、[完整复核](completion_review/REVIEW.json)、[全组SUMMARY](completion_review/SUMMARY.json)、[事件归因](completion_review/ATTRIBUTION.json)、[bootstrap](BOOTSTRAP.json)、[机器分析](RESULT_ANALYSIS.json)。

## 接纳为什么归零

4次尝试/window，总138,480次。下表分母是尝试，不是独立window：

| 结果 | 次数 | 占全部尝试 |
|---|---:|---:|
| 无可移除非代表原生槽 | 107,064 | 77.314% |
| 通过新颖性但完整集合运动学拒绝 | 31,382 | 22.662% |
| 新颖性拒绝 | 34 | 0.025% |
| 接受 | 0 | 0% |

由于没有接受，四次尝试中几何资格不变化。outer797/1035(77.00%)、inner1581/1856(85.18%)、CAL2083/2879(72.35%) window无可移除槽，各arm/seed相同。平均原生簇数outer19.699、inner19.843、CAL19.656；原20并非“大量重复候选”。

对31,382次physical拒绝逐项统计（同一次可失败多项，不能相加当总样本）：

| 完整临时K20失败项 | 次数 |
|---|---:|
| acceleration mean | 31,382 |
| acceleration p95 | 29,534 |
| jerk mean | 31,382 |
| jerk p95 | 19,984 |
| speed p95 | 2,975 |

这不是只看新增残差的导数，而是实际临时完整集合相对该window原生集合的比较。方向1position3的原bottom4替换可通过population五motion，但替换几何冗余槽时失败；改变被删路径就会改变完整集合的物理分母与尾部，二者不矛盾。此结果不能单独归因于head或loss。

## 严格同候选覆盖

至少一个完整candidate同时ADE≤.2 AND FDE≤.4，J=minKmax(ADE/.2,FDE/.4)≤1；不是把两条路径的独立minADE/minFDE拼成成功。

| split / group | native与policy（每seed） | 旧bottom position3 | 旧bottom acceleration3 | K24 position3 | K24 acceleration3 |
|---|---:|---|---|---|---|
| outer overall /1035 | 625 | 628/628/628 | 628/628/627 | 629/629/629 | 629/629/628 |
| outer change /253 | 110 | 111/111/111 | 111/111/111 | 112/112/112 | 112/112/112 |
| inner overall /1856 | 932 | 931/931/931 | 931/931/931 | 932/932/932 | 932/932/932 |
| inner change /497 | 166 | 165/165/165 | 165/165/165 | 166/166/166 | 166/166/166 |
| CAL overall /2879 | 1490 | 1494/1494/1495 | 1494/1493/1493 | 1499/1499/1499 | 1500/1499/1499 |
| CAL change /700 | 243 | 244/244/245 | 243/243/243 | 247/247/247 | 247/247/247 |

所有stable/transition、绝对误差、有效好簇、distinct、同Jclosest完整路径早/中/晚指标和反例保存于SUMMARY，不只筛选change好数。Policy native→policy rescue/lost都是0。相对于bottom，它会恢复少量旧替换丢失，但同时放弃原有rescue；不是新的生成收益。

Policy−native所有24个CI记录估计和CI均[0,0]，因为数组恒等，不能解释为对“生成有效”的强统计证据。Outer policy−bottom overall覆盖差：position −0.28986pp，探索CI[−0.76053,+0.10823]；acceleration −0.25765pp，CI[−0.69446,+0.10746]。Change两arm −0.39526pp，CI都跨0。Track+3模型seed自助法是探索性，不能伪造独立scene重复。

## 五运动学与不等预算反例

Policy20全72个arm×seed×split×group联合五motion PASS，仅因为等于native。K24全72联合FAIL；其加速度均值相对native+7.55–13.46%、加速度p95+11.06–20.27%、jerk均值+7.55–11.45%、jerk p95+7.20–14.55%，四项每组均超1%。speed p95没有在这些population组超1%并不能抵消其余失败。

K24覆盖保留native且有少量新增rescue，确实是有限bank中的新质量见证；但不证明等K收益、可安全部署、可学习性或“理解运动”。

## 资源

生产peak private1.70916GiB/RSS.27589GiB，系统最小physical15.62609/commit25.01131GiB。CPU复核private1.63982/RSS.21420GiB，最小physical17.62152/commit24.42968GiB。过程6GiB及physical/commit各reserve6GiB守卫未放宽；未调整OS/pagefile或关闭用户程序。预检峰值不冒充生产峰值。

## ARS inline反方2与3（同助手，不是外部评审）

DA2：执行证据充分，科学结论必须为负。身份政策不支持“学到安全互补”；K24更大预算与运动学失败禁止升baseline。固定.10/.20聚簇、最低代表和最严window门也可能过于保守，此替代解释保留，但不事后搜阈值求PASS。组合删除与单slot效用不可混同。

DA3：保留native baseline。不强删singleton、不用GT leave-one-out效用作部署输入、不松1%门、不扫slot/cluster阈值。不将此结果扩展为所有generation失败。全major（change ADE/FDE各5%、protected1%、三seed一致、联合CI、同路径覆盖和五motion）仍未实现，评分方向4门不触发。

## 下一步唯一因素

方向3固定方向1 **position3** 结构、输入、paired初始化、data/统计算法、bottom4/16原生、优化器、25epochs和三seed，仅比较质量functional：

- control：min4(ADE+.5FDE)；
- joint：min4 max(ADE,.5FDE)=.20×J，保持米单位。该functional同时决定hardwinner和winner loss，不拼不同candidate；
- 六pair平方hinge(.3)×.05原样。此次不额外加protection coefficient、不改head、不训练scorer。

position3依据先前primary物理安全选择，而非CAL最佳结果。此对照测试瓶颈quality目标，不直接证明credit根因、保护有效或覆盖会提升。原生支撑整合门不修改；方向4仍等待稳定非零覆盖收益与全部运动学门。新folder精确预注册，实际GPU/source-ID、双方真实子集2epoch全optimizer/RNG exactresume、资源吞吐全部PASS且规划≤30min后才生产启动与ETA；当前尚未实现或启动。

F16已见原pool、target-disjoint非context-disjoint、只有3scene/3seed、窗口相关、重复CAL及adaptive探索限制保留。没有理解运动/SOTA/独立wholemodel泛化/唯一decoder或loss根因结论。

