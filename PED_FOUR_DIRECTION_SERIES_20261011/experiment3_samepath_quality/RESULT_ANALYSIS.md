# 方向3：同路径质量目标对照——完成分析
日期：2026-10-11。状态：ANALYZED；运行及保存数据审核PASS，科学major FAIL。四方向中三项完成，评分方向未触发。

固定position3结构、训练数据、初始化、统计、25epoch和K20集合，只比较 sum=min4(ADE+.5FDE) 与 joint=min4 max(ADE,.5FDE)。后者=.20J，hardwinner与winnerloss一致。六pair平方hinge(.3)*.05不变。两种目标同时改变责任分配与优化几何，不能当作纯credit因果对照。固定bottom4修正，其他16原生候选完全不变；每时刻修正范数上界 .20(t/12)^2。

## 完成与审核
- 独立worker实际2026-10-11 01:33:21.634747→01:37:56.251634香港时间，274.616887秒；worker与exit双SUCCESS。
- 6×25完整训练，150history，18保存jobs；worker完成后自动分析未及时跟进。用户12:03进度询问的一次只读核验确认完成，不冒称准点半程。
- 本轮完成原先欠缺的CPU completion_review/audit_saved.py：87.408594秒；18全部保存质量、原SciPy GT-free全K聚簇、五运动学、固定16/边界/控制重构、assignment、150history与6manifest、36bootstrap/144scalar区间重算误差均0。
- 1746相关科学源、103输出SHA初末不变；sum所有9个输出的注册数组与上一position3逐数组BITWISE EXACT，是fresh重复，不是独立泛化。
- 保存算术无Torch、模型前向或optimizer，未再次回放所有模型或梯度；审核及反方检查均由同一助手执行，非外审。
- 生产private2.770535/RSS2.052505GiB；复核private.321362/RSS.338787，最小physical12.432201/commit20.449432GiB。原process6/system两项reserve6保持。
- 来源：[REVIEW](completion_review/REVIEW.json)、[SUMMARY](completion_review/SUMMARY.json)、[RESULT](RESULT.json)、[BOOTSTRAP](BOOTSTRAP.json)。

## 严格同路径覆盖
好候选必须同一条完整路径ADE≤.2m且FDE≤.4m。三seed依次42/43/44。

|人口|N|native good|sum good|joint good|
|---|---:|---:|---|---|
|outer overall|1035|625|628/628/628|628/628/628|
|outer change|253|110|111/111/111|111/111/111|
|inner overall|1856|932|931/931/931|931/931/931|
|inner change|497|166|165/165/165|165/165/165|
|CAL overall|2879|1490|1494/1494/1495|1494/1496/1494|
|CAL change|700|243|244/244/245|244/245/244|

Joint outer overall覆盖增益+.289855pp，95%探索CI[-.108231,+.760529]pp；change+.395257pp，CI[-.847548,+1.831670]pp。joint−sum的outer所有组coverage差为0，区间[0,0]，不能据此证明未来总体等价。Change paired joint−sum ADE差+.000063668m，CI[-.000113315,+.000231716]；FDE+.000128275m，CI[-.000484082,+.000784689]，J差+.000013539，CI[-.001356298,+.001390038]。没有稳定优于原目标的证据。

Joint outer change相对native FDE分别恶化.476169/.425384/.334069%；ADE分别恶化.026380/.052962%、改善.017531%，远未同时达到5%收益。
全部72 arm×seed×split×group五运动学通过原literal1%门；overall/stable/transition ADE/FDE保护未超1%。安全通过不能替代收益条件。major FAIL，baseline及cycle11不变。

## 新补充：候选选择与修正边界的必要可达上界
这是本轮事后保存数据诊断，不是训练结果、部署oracle或新模型。令原生候选第t步误差为 e_t，许可修正范数 r_t=.20(t/12)^2，则三角不等式给出最小可能误差下界 max(e_t-r_t,0)。下界的12步均值及末步必须同时≤.2/.4，才有可能产生好路径。

计算故意去掉三节点basis、运动学和可学性限制，因此是宽松必要条件；“可能”不能当作真正可达，“不可能”则在当前边界内有数学依据。用朝GT径向截断构造独立核对，下界误差max1.776357e-15；18保存文件native/GT/ID/slot一致、输入SHA初末不变、所有实际rescue均属于可能集合。

|人口|原生失败|固定bottom4可能救回上限|确定超出bottom4边界|若任意20原生anchor可寻址的宽松上限|
|---|---:|---:|---:|---:|
|outer overall|410|6|404|209|
|outer change|143|3|140|56|
|inner overall|924|4|920|471|
|inner change|331|3|328|122|
|CAL overall|1389|21|1368|659|
|CAL change|457|8|449|175|

即使零lost，当前bottom4局部修正的outer overall好窗口最多631，change最多113；实际为628/111。不能期待在这个可行域内仅换loss获得大幅coverage突破。任意20的数字并未满足当前四槽部署、三节点basis或五运动学，更没有预测哪个anchor可修复；不可写成“改选槽就能救回209个”。

补充脚本和全部IDs：[reachability_envelope.py](completion_review/reachability_envelope.py)、[REACHABILITY_ENVELOPE.json](completion_review/REACHABILITY_ENVELOPE.json)。

## 反方检查与决策
反方2：目标对照在受限bottom4可行域中失败，不能推广成联合质量目标普遍无效。CAL有小增、全部运动学通过，应保留；但它们不满足primary显著改善。
反方3：any20上限依赖GT、忽略动力学和学习，只支持“候选寻址值得优先核实”，没有证明某个选择器有效。原生候选不冗余与小幅修正近乎无收益可以同时成立。
方向4继续NOT_TRIGGERED：没有稳定完整coverage收益，暂不评分训练。下一设计优先验证修正对象的训练集限定选择规则；结构、幅度、quality loss和预算不能同时改变。不是继续扫描joint权重或直接放大修正界。
