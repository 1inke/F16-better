# 方向2预注册：几何支撑保留的K20整合（不训练）

2026-10-11方向1完整分析后、任何方向2结果前固定。属于已授权四方向的第二项，自适应探索，不是独立确认性。这里只实施一个固定GT-free integration政策，不扫描阈值/槽位组合、不改任何模型。实施节点2026-10-11 00:40HK COUNT1，不是生产ETA。

## 问题与来源

方向1 loss下降但primaryouter覆盖仅净增1–3window，bottom4替换仍lost，acceleration3五motion FAIL。历史tail证明低logit不代表集合冗余。因此只比较同一保存native20+refined4池下，原bottom4替换 vs 原生几何支撑保留整合。两arm全部3seed、outer1035/inner1856/CAL2879共18个sourcejobs都报告，不按CAL或GT选择arm/方案。方向1six25epoch已完成，不训练、不重放模型、不实例化torch、旧模块不访问。

## 唯一固定政策

距离 D(p,q)=max(mean_12 Euclidean/.10,endpoint Euclidean/.20)，继承原GT-free complete-link .10/.20，不是从本次测试收益拟合的距离/阈值。

对每window只输入native20与已保存refined4（按原physical bottom4槽升序对应）；function不得接收GT/group/ID/scene/track或logit。原head训练及原refined4来源依赖已固定bottom4，不能声称整个管线摆脱旧logit。

1. 对immutable native20在D上用原SciPy complete-link、cut1。每cluster预留最低physical slot为原生代表，整个政策绝不删除代表；singleton全部保留。每个native因此始终有一条**未修改原生**代表D≤1。
2. 当前集合初始native20；只遍历refined4固定0,1,2,3，各至多尝试一次。原生非代表、且仍保留的slot才是removable；已插入learned路径不可再删除。
3. 若无removable，本次拒绝并继续，不强删singleton。否则选择D(native_slot,reserved_representative)最小的removable slot，精确ties按physical slot升序；此选择不用GT/logits。
4. 假定删该slot后，计算learned与剩余19条完整path的minD；只有minD严格大于该removed到其原生代表的D才算几何新颖，才继续（strict>无epsilon）。不寻找另一删槽或伙伴使它PASS。
5. 暂插入learned后，以window全部20候选全部time points按dt.4重算speedp95/accelmean,p95/jerkmean,p95；五项每项literal≤该window immutable native20相应值×1.01才接受。native值0时上限0，无floor/epsilon。否则恢复完整当前集合并记录拒绝，继续下一个learned。不裁剪/投影/调整learned值。
6. 最终固定K20。每originalnative到保留原native代表D≤1的certificate必须实际重算；输出不可留zero padding/copies。policy可以插入0、1、2、3、4条，必须报告比例/拒绝原因和removed histogram。

它是GT-free几何支撑政策，**不是严格GT质量支撑保证**：D≤1允许.1meanpath/.2endpoint变化，靠近.2/.4边界仍可能lost。个体相近不表示真正GT可互换；完整cluster代表保护避免只因individualLOO=0就联合删空。perwindow quantile保护不数学保证pooledglobalp95，最终每group全population五motion照常严格判定。

## 对照与评价

Primary只outer固定epoch25；inner/CAL仅posthoc。不跨不同population配对。每source保存native20、原bottomrefined20、policy20及origin map/admission/rejection/certificate。K24=native20+refined4仅finitebank不等预算参考：完整候选coverage/rescue/误差作为available-information ceiling，不冒称公平部署或可学习收益；不会GT挑partner。没有重新生成trajectory或用GT选择。

所有ports同ID/group/GT posthoc计算同candidateADE≤.2AND FDE≤.4/J≤1完整coverage、rescue/lost/retained、minADE/FDE/J、单Jclosest早中晚，固定.3/.5敏感性。每K20完整原SciPy GTfree complete-link .10/.20/effectivegood/distinct，不最大化cluster或GTcopies。五项pooledglobalmotion每seed每groupliteral≤native1.01；原major changeADE/FDE各5%、protectedoverall/stable/transition1%、三seed一致及nativefresh联合CI不降低。K24也报告五motion但不等K20主要成功口径。

必须给出native→policy及bottom→policy成对rescues/lost，不能只报净coverage；原native原生代表保留、geometry支持与严格good支撑分别验证。pairedtargettrack+3modelseed2000bootstrap rng20261011，outer/inner/CAL×overall/stable/transition/change×(policy-native,policy-bottom)注册24records96scalarCI，探索不多重显著confirmatory。源两arm分别算，共48records192scalar，精确计数实现后核验。报告所有种子/arm，不选最好。

## 实施、预检与恢复

全部newcode/protocol/cache/log/output/失败仅experiment2_native_support，旧来源所有digests继承并逐项扩first的完整output/分析/CPUreview/失败和extra2audits，不覆盖旧hash。source1490不是allhistory34k。只新源码允许锁前修，legacy不执行top-level，不用torch/model/optimizer。

CPUfixtures：unique singleton不删除；重复native cluster保留一个代表；20copies归1cluster非20信息；jointremoval不删代表；novelty严格tie拒绝；native0 physical拒绝非零；knownunsafe/all5逐项拒绝；GT无关signature和相同paths不同posthocGT政策输出EXACT；stablephysicalties；fixture同candidate/disjointminima，K24不混公平指标。两套NumPy/独立loop政策、完整原SciPycluster与allfiveMotion至少actualouter128双方source重新保存重读max误差原1e-12/certificate literalD≤1（不放宽）。

source/ID/nativecache/slots与sixmodels不修改SHA；已完成jobsource+outputSHA skip、坏source/output实拒绝只分析恢复，不训练optimizerRNGresume。内部resourceguard process6GiB/physical+commit eachreserve6，batch/chunk128/256有限输出，不OS/pagefile/userkill。8batch真实saved128完整政策+fullqualityclusters双算术/全初末SHA与IO实测后给ETA；planning>30min停止长任务，不改科学规模求PASS。预检全部PASS才唯一0triggerOSbyte-lock启动、一次actualPID/parent/task/workerprogress/stderr startup、一次50%预计elapsed→一次completion定点；当前没有worker/PID/ETA。

## 完成

newworker+newexit双SUCCESS、18科学jobs全SHA、allports explicitIDs finiteK20、完整quality/SciPyclusters/pooledmotion/decomposition/certificates/fullCI同注册定义算术、所有source/旧outputs/models初末SHA/RESOURCE PASS；独立新CPUcompletion_review对全部18保存结果重算。saved arithmetic不是模型/梯度再次回放或外审。RESULT_ANALYSIS MDJSON+inlineARSDA2/3同助手非外审、runtimeANALYZED后只做一个下一因素，不能自动开启scorer；方向3目标对照需结构/政策事前固定，不同时改loss/architecture。

若全部政策近identity，则报告原native冗余不足/physicaladmission瓶颈，不能强删/改cluster阈值或无限新policy变体。若coverage改善但motion/major不通过仍FAIL，不升级baseline/historycycle11/goal。方向4当前未触发生成稳定门。F16seen、target非contextdisjoint、3scene/3seed/相关windows、重复CAL和adaptive过程限制保留；没法辨识根因就明确限制，不宣称理解/意图/SOTA。

## ARS反方1（同助手非外审）

PASS bounded question；Major：几何支撑不等GT strict成功保护，perwindowp95不等pooledp95、旧learned仍底logit来源。Major：native多数singleton，policy可能近identity，必须报告可插入规模，不以无插入推断所有generator无信息。Minor：先前结果后选policy属adaptive探索，不能confirmation。最强反驳：.1/.2代表可能不保阈值附近正确path；直接按严格lost+allpooled验证，不以证书替代科学gates。
