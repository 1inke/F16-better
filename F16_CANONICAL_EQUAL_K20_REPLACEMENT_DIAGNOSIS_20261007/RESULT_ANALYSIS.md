# 等K20候选替换诊断：结果分析

结论：新增四条确实有小幅互补性，整体严格覆盖在固定20条后仍提高；但未形成全面更好的候选集合。变化组平均minADE/minFDE/J显著恶化，变化覆盖提升不稳，速度尾部保护大量失败。执行成功，major FAIL，baseline不升级。

## 运行与核验

- 唯一worker37084/service2840，2026-10-07T23:25:47.360467HK开始，23:28:30.774602双SUCCESS，实际2m43.414。唯一到点检查23:29:46.2802569迟到发现完成，taskReady/XPath0/processnone/stderr0/27done；不冒称23:27按时检查，无第二次完成查询。
- NONFORMAL，复用9个canonical固定25epoch模型的27份保存输出，52686行。无torch/model前向、反向、optimizer、重训或正式计数增加。原native20/模型和来源不变。
- 原worker完整两套质量/原SciPyGT-free complete-link/五项pooled运动学/分解算术误差0，72条探索bootstrap记录=288标量CI确定性复算0。补充completion_review/review.py再次对全部27保存数据完整复算，质量/聚簇/运动学/分解及CI误差0，81个job文件、bootstrap、9models及470相关locks（继承385原digest）初末SHA PASS，不是全部历史文件或全部模型再次重放。全部组/种子细节见completion_review/SUMMARY.json及RESULT.json。
- 生产峰值private.236328GiB/RSS.254948GiB，系统最低physical19.155117/commit23.476967GiB。补充private.945560/RSS.243492/minphysical19.396240/commit23.255939，6GiB过程及两项余量通过。
- 首次资源STOP在fixtures/数据前，physical.967GiB<6，原source/RESOURCE/runtime/trace均保留failures/resource_gate_1；此次恢复成功不删除旧失败。

## 固定比较

native20 vs 保留16原生+全部4新增，K均20。移除原生低分4条仅按冻结logits descending stable rank，tie原slot优先；移除slot按原顺序放入learned0..3。不用GT挑选、不训练scorer、不宣称沿用原logits是新增路径概率。common16/removed4/learned4分解固定，不搜索槽位/排名/比例。inner与originalCAL只事后评价，不选择方案。

## Primary outer：整体与变化窗口必须分开看

|划分|组/N|原生好窗口|替换后三seed|平均覆盖差pp [探索95%CI]|ADE/FDE收益%（负数为恶化）|
|---|---|---:|---|---|---|
|场景族|overall/1561|1040|1078/1081/1072 |2.370 [1.595, 3.429]|-1.460 / -1.127|
|场景族|change/211|82|86/85/85 |1.580 [-0.958, 4.510]|-15.515 / -18.819|
|行人轨迹|overall/1035|625|655/658/655 |2.995 [1.937, 4.211]|-1.691 / -3.542|
|行人轨迹|change/253|110|113/116/116 |1.976 [-0.138, 4.183]|-9.560 / -17.747|
|记录时间|overall/566|303|314/315/318 |2.238 [0.844, 3.925]|-0.973 / -1.817|
|记录时间|change/142|60|59/58/61 |-0.469 [-3.185, 2.253]|-8.666 / -9.568|

整体coverage探索CI都正，但整体平均误差点估计不改善，不能只报coverage。变化组scene/target覆盖CI跨0，time略降且跨0；所有三族outer变化ADE/FDE/J差值CI为正（替换有害）。这是对固定预算政策的否定证据，而非小收益被隐藏。

## 变化组 rescue–lost：原生低评分并不等于无用

|划分/split|变化N|救回三seed|丢失三seed|原生→替换good三seed|
|---|---:|---|---|---|
|场景族/outer|211|7/6/5 |3/3/2 |82→86/85/85|
|场景族/original_CAL|700|11/18/15 |7/7/7 |243→247/254/251|
|行人轨迹/outer|253|5/8/8 |2/2/2 |110→113/116/116|
|行人轨迹/original_CAL|700|14/17/18 |7/7/7 |243→250/253/254|
|记录时间/outer|142|2/1/4 |3/3/3 |60→59/58/61|
|记录时间/original_CAL|700|13/14/19 |7/7/7 |243→249/250/255|

CAL变化原生243/700，替换后247–255，只净增4–12个；每个模型都有7个原来成功的窗口丢失。其变化ADE平均恶化约9.6–9.7%，FDE约12.5–12.8%，J也恶化。CAL变化coverage各族CI跨0或触0，不能称可靠突破。CAL整体原生1490/2879→1547–1573，平均coverage+2.44/+2.64/+2.59pp且探索CI正，是必须保留的正面反例；CAL不是整体完全不能迁移。

这里不是‘loss下降却coverage不增加’的因果实验：同一冻结checkpoint只改变输出组合，未做参数更新，不能由此确认objective mismatch或共享head干扰。minADE/FDE的不同oracle槽位与同条J/严格覆盖不混淆。

## 集合互补性和有效好簇

equalK4诊断learned4对removed4在outer变化覆盖平均+22.27/+18.71/+14.32pp，ADE/FDE/J也明显改善（该K4探索CI有利）。因此不能说新增路径纯粹无效。但单独K4较好不保证放回common16后的min质量更好：原生低分4条虽平均弱，仍可对少数特别难窗口提供原生16没有的近似路径；新4条的跨门槛救回多于丢失，却不能完全替代这些尾部路径的误差补偿。此为集合互补性解释，不是已证明根因。

有效good簇在outer变化增加（scene原生.51185→.62085/.62085/.58294，target.50593→0.61660/0.62055/0.62846，time.50000→.56338/.54930/.59155），同时time严格coverage可下降。簇数不是目标最大化；K20有效distinct约18.6–18.9不是简单20复制，也不证明所有路径有用。精确每组指标见SUMMARY和原保存数据，不能用effectivegood替代严格coverage/CI/motion。

## 运动学：不能只看加速度

replacement20共108个seed×split×group检查，29 PASS、79 FAIL；所有失败是speed_p95超native1%，最大+3.209%（time seed42 inner stable）。accelmean/p95、jerkmean/p95均在这108组内通过；部分change组speed也通过，但不能代替所有组保护。移除慢路径与加入较快路径会机械改变ALL-K分布，p95上升并不单独证明每条新增轨迹不合理；仍违反当前固定保护标准，不能事后放宽或改成只审加速度。具体79失败明细在SUMMARY.json。

## 当前分支及下一步判断

小互补性成立，等预算总体覆盖信号成立，但‘可靠改善变化行为+全部运动学安全’尚未成立。major明确FAIL：没有change ADE/FDE各>=5%收益，反而恶化；motion大量FAIL；变化coverageCI/seed不稳。基线和历史cycle11/长研究目标不变。

不进入scorer优先分支，不盲换hardwinner/softmin，不扫删除槽位或增加专家数量。本次不能判断行人心理意图，也没有验证‘恒速外推失败子集’——change标签不等于已核验CV失败。若人类批准下一精确实验，优先定位变化难例中被移除原生路径的不可替代作用及新增路径速度尾部，再考虑能利用过去信息、兼顾已有候选互补性和物理可行性的生成设计。下一项需单独预注册与权限充分；本次窄诊断到此闭环，无自动新增实验。

## ARS反方检查2（同一助手，非外审）

Verdict PASS限定解释 / 拒绝major或根因结论。Major问题：整体覆盖增加容易遮蔽变化组平均误差显著变坏，time变化两个seed丢失多于救回；已分组报告。最强反驳：oldlogit底4选择使本政策同时改变native保留组成，失败不代表所有equalbudget组合或生成器无效。反面证据：learned4单独K4优于removed4、所有整体coverageCI正，故不能归纳‘完全没有可迁移规律’。缺口：未有canonical匹配random-arm、未测特定CV-failure条件子集。

## ARS反方检查3（同一助手，非外审）

Verdict PASS带局限。不可把冻结组合评价当训练loss根因/优化因果，也不可把108检验当108独立场景。不同grouping outer IDs/train量、三seed/三scene/window相关/F16已见原pool/CAL单university且历史复用，CI探索且无多重确认保证。SOTA、独立whole-model泛化、‘真的理解运动’不成立。最强反驳仍是组合策略而非纯生成质量；保留旧K3负结果、此次K4正结果与speed尾部失败，未选择有利分母/阈值。所有原科学gates不放宽，AI-assisted分析与算术核验非外部独立审稿。

## 证据入口

[RESULT](RESULT.json)、[BOOTSTRAP](BOOTSTRAP.json)、[ARITHMETIC_AUDIT](ARITHMETIC_AUDIT.json)、[PRESERVATION](PRESERVATION.json)、[RESOURCE](RESOURCE_USAGE.json)、[CPU_REVIEW](completion_review/REVIEW.json)、[全部组指标](completion_review/SUMMARY.json)、[预注册DESIGN](DESIGN.md)。skill: academic-research-suite；设计前DA1与完成DA2/3约束归因范围、完整反例与科学标准。
