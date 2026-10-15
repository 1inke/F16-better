# 方向1：三节点位置修正 / 加速度积分对照

## 结论

六个25epoch训练、18保存jobs完整完成，但**两组均major FAIL，baseline不升级**。加速度积分未稳定改善完整覆盖，其36个split×seed×group的ALLK加速度均值较native提高2.513–2.896%，全部违反literal1%保护线。position3的36组五项运动学全部PASS，加速度均值下降1.938–2.526%；覆盖净增很小且主要CI跨0，不是大突破/理解运动证明。

停止扫描加速度系数、节点或轮数。下一步按已批准四方向做**方向2冻结输出GT-free集合整合单因素审核**，不训练。方向3另行目标对照，方向4生成稳定coverage+五motion门未触发。

## 实验设计

唯一因素：同24134参数120→128→64→6 GELU、past P2 96D+native path24D、paired初始化下，position3节点4/8/12位置插值 vs acceleration3节点1/6/12加速度.05m/s²插值、dt.4两次积分。双方trajectory-wide GT-free lambda位置界.20(t/12)^2，不保证物理安全；equalparams非equal reachable geometry/有效容量。

固定target train18909、outer1035(primary fixed25)、inner1856、CAL2879(posthoconly)，seeds42/43/44。hardmin4(ADE+.5FDE)+.05六pair平方hinge(.3)，AdamW.001/wd.0001/batch256/shuffle25/noclip。固定bottom4原生logit槽修正、另外16 EXACT/K20；bottom-logit不是冗余证明。旧F16/decoder/scorer不实例化或训练，仅immutable cache；GT/groups/logits不headforward，GT只原loss和posthoc。

## 完成审核

- 30276/parent2688从2026-10-11 00:16:31.594153 HK到00:21:34.489003双SUCCESS，实际302.894850秒=5分2.895秒。
- 唯一half计划00:20:31.594，实际00:22:47.756642执行（迟到约2分16秒）发现完成；不冒称准点，不另查完成PID或重复worker。stderr0。
- 6done/150historyrows/18显式ID科学jobs/103相关outputSHA/1490相关source初末PASS；不是全部历史34k审核。
- ARITHMETIC/PRESERVATION/RESOURCE PASS；全部eval参数/bool/scaler/basis EXACT、fixed16/bounds/NumPy controls重构PASS。
- CPU独立completion_review 55.819秒，全部18质量、原SciPy GT-free ALLK complete-link(.10meanpath/.20endpoint)、五motion、controls、assignment、150history/manifests及36bootstrap/144scalar CI重算max0；34620是跨seed/arm重复saved rows，不是独立样本数。
- saved arithmetic不是所有model/gradient二次回放或外部评审。
- Production private2.72343/RSS2.00808/minphysical13.81197/commit23.91995GiB；CPUprivate.28017/minphysical16.52759/commit27.46666GiB。process6GiB/BOTHreserve6未放宽。

## 覆盖与误差

完整好候选严格同candidate ADE≤.2 AND FDE≤.4；J=min_k max(ADE/.2,FDE/.4)。minADE、minFDE可能不同slot，不能拼成完整路径。

|arm|seed|split|group|N|native→refined good|rescue/lost|minADE/minFDE m|五motion|
|---|---:|---|---|---:|---|---|---|---|
|position3|42|outer|overall|1035|625→628|4/1|0.175542/0.296412|PASS|
|position3|42|outer|change|253|110→111|2/1|0.249203/0.431237|PASS|
|position3|42|inner|overall|1856|932→931|0/1|0.193643/0.338562|PASS|
|position3|42|inner|change|497|166→165|0/1|0.260592/0.461188|PASS|
|position3|42|original_CAL|overall|2879|1490→1494|9/5|0.180391/0.302152|PASS|
|position3|42|original_CAL|change|700|243→244|4/3|0.245917/0.430771|PASS|
|position3|43|outer|overall|1035|625→628|4/1|0.175548/0.296325|PASS|
|position3|43|outer|change|253|110→111|2/1|0.249176/0.430913|PASS|
|position3|43|inner|overall|1856|932→931|0/1|0.193603/0.338475|PASS|
|position3|43|inner|change|497|166→165|0/1|0.260502/0.461086|PASS|
|position3|43|original_CAL|overall|2879|1490→1494|9/5|0.180375/0.302117|PASS|
|position3|43|original_CAL|change|700|243→244|4/3|0.245957/0.431059|PASS|
|position3|44|outer|overall|1035|625→628|4/1|0.175510/0.296326|PASS|
|position3|44|outer|change|253|110→111|2/1|0.249024/0.430906|PASS|
|position3|44|inner|overall|1856|932→931|0/1|0.193603/0.338516|PASS|
|position3|44|inner|change|497|166→165|0/1|0.260531/0.461349|PASS|
|position3|44|original_CAL|overall|2879|1490→1495|9/4|0.180380/0.302042|PASS|
|position3|44|original_CAL|change|700|243→245|4/2|0.245987/0.430912|PASS|
|acceleration3|42|outer|overall|1035|625→628|4/1|0.175566/0.296445|FAIL|
|acceleration3|42|outer|change|253|110→111|2/1|0.249259/0.431680|FAIL|
|acceleration3|42|inner|overall|1856|932→931|0/1|0.193561/0.338241|FAIL|
|acceleration3|42|inner|change|497|166→165|0/1|0.260403/0.460562|FAIL|
|acceleration3|42|original_CAL|overall|2879|1490→1494|10/6|0.180414/0.302113|FAIL|
|acceleration3|42|original_CAL|change|700|243→243|4/4|0.245989/0.430688|FAIL|
|acceleration3|43|outer|overall|1035|625→628|4/1|0.175564/0.296361|FAIL|
|acceleration3|43|outer|change|253|110→111|2/1|0.249266/0.431487|FAIL|
|acceleration3|43|inner|overall|1856|932→931|0/1|0.193606/0.338468|FAIL|
|acceleration3|43|inner|change|497|166→165|0/1|0.260537/0.461262|FAIL|
|acceleration3|43|original_CAL|overall|2879|1490→1493|9/6|0.180435/0.302236|FAIL|
|acceleration3|43|original_CAL|change|700|243→243|4/4|0.246049/0.431068|FAIL|
|acceleration3|44|outer|overall|1035|625→627|3/1|0.175590/0.296530|FAIL|
|acceleration3|44|outer|change|253|110→111|2/1|0.249327/0.431915|FAIL|
|acceleration3|44|inner|overall|1856|932→931|0/1|0.193631/0.338489|FAIL|
|acceleration3|44|inner|change|497|166→165|0/1|0.260553/0.461201|FAIL|
|acceleration3|44|original_CAL|overall|2879|1490→1493|9/6|0.180455/0.302346|FAIL|
|acceleration3|44|original_CAL|change|700|243→243|4/4|0.246110/0.431691|FAIL|

outer change253：两组都110→111/seed，rescue2/lost1，仅+.3953pp。position3 changeFDE恶化.356–.433%，acceleration3恶化.491–.591%，远未达到ADE/FDE各≥5%。inner change497两组166→165、rescue0/lost1。CALchange700 position3 243→244/244/245，acceleration3 243→243/243/243（rescue4/lost4）。CALoverall/transition少量改善是正面反例，不能抹去。

outer change effectivegood .505929→.509881两组相同，微小增加不抵物理FAIL。所有stable/transition/native绝对指标、单Jclosest早中晚及五motion绝对/相对值在completion_review/SUMMARY.json；RESULT_ANALYSIS.json含完整72组紧凑行，不跨split配对。

## 统计与训练loss

Paired targettrack+3modelseeds/2000bootstrap/rng20261011探索CI，原36records144scalar全部保存并重算。
- outer overall position3 coverage+.28986pp CI[-.10471,+.79293]pp；acceleration3+.25765pp CI[-.13838,+.72092]pp，均跨0。
- outer change+.39526pp：position3 CI[-.84388,+1.85202]pp，acceleration3[-.84755,+1.83167]pp，均跨0。
- acceleration3-position3 outerchange FDE+.0006753m CI[+.0000401,+.0014977]m，探索性正；其他多项跨0，不单据此宣称普遍结构劣势。
- CALchange position3+.19048pp CI[-.45354,+.95251]pp；acceleration3为0 CI[-.76932,+.84749]pp。inner小幅负coverage上界0，不证明完全不可迁移。
- 第1到25epoch记录train loss下降position3 2.234–2.376%、acceleration3 1.807–1.948%。第1epoch已有更新不是初始化loss；训练分母非完整FIT populationcoverage，未保存fullFITcoverage不能虚构。

## 推论边界和下一步

只能否定**这个**三节点加速度积分+位置lambda+bottom4映射包的稳定优势；不能否定所有运动过程头/证明loss或shared-head唯一根因。位置对照安全但质量弱，只保留证据，不升级baseline。方向2采用native geometry complete-link支撑证书、固定次序与联合删除保护；不GT选槽、不调阈值、K24只不等预算参考。几何支持保留不保证严格阈值边界无lost，也不保证全局pooled p95，仍需实测。native singleton多时允许零插入，不强删rare支持腾位。

## ARS反方2（同助手非外审）

PASS WITH LIMITATIONS，无Critical算术缺陷。Major：loss下降不能代表fullFIT覆盖，且equalparams非equal有效空间；bottom4支撑因素与结构受限耦合。Minor：少数窗口与重复seed不能当大量独立证据。最强反驳：位置lambda收缩加速度空间可能导致结果，不能断言加速度本身不可学；保留限制而非扫系数求PASS。

## ARS反方3（同助手非外审）

PASS bounded conclusions / major scientific FAIL。最强反驳：安全position3少量救援有局部价值，已报告CAL/transition正面证据但不升级。移除最强正面outer净增，结论仍成立；跨场景未验证。F16seen、target非context-disjoint、3scene/3seed、window相关、重复CAL/adaptive历史，不是wholemodel未见泛化、运动理解、心理意图或SOTA。AI-assisted分析及两次批评同助手非外部评审。全部major原标准及五motion1%不放宽。

## 来源

DESIGN/protocol、RESULT/ARITHMETIC_AUDIT/PRESERVATION/RESOURCE_USAGE/BOOTSTRAP、六run history/assignment、completion_review/REVIEW/SUMMARY。科学锁源/旧模型结果不可覆盖；旧暂停E3及旧automation保持，baseline/cycle11/unfinishedgoal不重置。
