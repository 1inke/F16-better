# Removed-tail attribution / leave-one-out utility audit：完成分析
日期：2026-10-08。状态：ANALYZED；仅离线保存数据诊断，**没有训练模型或改动候选策略**。

## 结论

这轮支持“低 logit 不等于冗余；新增路径有互补价值，但直接替换原生尾部会破坏少数不可替代覆盖”。不支持把它进一步概括为“learned4 普遍强化高速运动，压掉低速/停留模式”，更不能证明模型理解了运动或获得了新的物理信息。下一研究重点应是 **保护原生支撑的互补集成**，不是立即再训练、换 scorer 或搜索删哪四个 slot。

## 执行与可信度

唯一后台 worker2084 从00:17:09.852327到00:17:39.18401，实际29.331683秒；worker和exit双SUCCESS，stderr0。原估计2–6分钟只是保守规划。唯一检查00:21:04.3377099，比预计半程迟144.485秒，发现已经完整完成；没有冒称按时检查，也没有再查完成进程。旧一次性自动跟进已移除。

27个固定epoch25输出、52686个run-window行、108个job文件。全20候选leave-one-out枚举、原生/新增误差、lost/rescue案例、几何/速度统计独立复算最大误差0；CPU补充再次从原始保存24路径独立复算最大误差0。571个相关source/input/result SHA及继承470个原digest不变，**不是全部历史34k文件重核验**，不是模型/梯度再次回放。旧全质量/完整聚簇审计结果以不变SHA引用，当前审计未再次跑全部SciPy聚簇。

生产private峰值0.182789GiB，最低物理余量18.6960、commit余量23.0952GiB；CPU补充private峰值0.861271GiB。这两项已完成工作当时的6GiB进程上限和两项6GiB系统余量均通过，未调整OS/pagefile或关闭用户程序。

报告收尾额外复验在FIRST内部resourceguard中止：物理余量0.763245GiB<6，commit16.180519GiB；额外571-source/108-output/12-figure总复验未完成，不能冒称PASS。此前worker和独立completion_review的完整核验仍成立；没有轮询资源、重试或新启动。该helper覆盖了通用RESOURCE_USAGE.json，最新失败快照已保留post_report_verification；原生产数据仍在不可改的RESULT.json.resource，另保存明确标注的production_RESOURCE_SNAPSHOT_FROM_RESULT.json。后续计算等人类释放内存后明确继续。

## A. 被移除路径是否不可替代？

完整好候选始终定义为同一条路径ADE≤0.2m **且** FDE≤0.4m。J=min-slot max(ADE/0.2,FDE/0.4)，Q=min-slot(ADE+0.5FDE)，分别保存，不拼不同slot。

| 主outer变化组 | 窗口N | 三seed lost | 三seed rescue | bottom4独占好候选窗口数 |
|---|---:|---|---|---:|
| scene |211|3 / 3 / 2|7 / 6 / 5|3|
| target |253|2 / 2 / 2|5 / 8 / 8|2|
| time |142|3 / 3 / 3|2 / 1 / 4|3|

所有27份输出共258个lost行事件，对应51个独立sample ID，其中变化组32个ID；这不是258个独立样例。每个实际lost窗口都只有一条native合格路径，它的J和Q winner均在removed4，且个体coverage utility=1。CAL变化组在九个head中丢失的是**同样7个ID**，不是63个独立案例。

“lost的合格原生路径都被移除”首先是common16不变带来的代数必然，不单独算机制发现。有价值的附加证据是：这些实际案例都是单路径支撑，并且与common16不接近。

## B. 到common16的距离

预注册D=min-common max(meanpath separation/0.10m, endpoint separation/0.20m)。全部lost中D最小2.4234、最大12.3945，D≤1的案例0。主outer变化lost的seed42均值：scene5.668、target5.693、time3.864。

CAL那7例：D均值7.464，最近common路径平均全程分离0.710m、端点分离1.460m。因此不是只差一点精确位置、却被原距离门槛算成冗余；它们承载了common16没有的几何支撑。但这仍不是语义“罕见运动模式”或所有聚簇唯一性的证明。

## C. learned4究竟错在哪里？

CAL这7例原生J路径平均ADE0.150m/FDE0.229m；各模型learned4的单条J最近路径平均ADE **1.096–1.215m**、FDE **2.253–2.562m**。不只是轻微跨越0.2/0.4门槛。所有四条路径的误差及独立minADE/minFDE也完整保存。

以单条J最近learned路径计，七例中：方向误差>30°有1–4例；沿GT位移方向端点overshoot>0.2m有2–3例；弧长多出>0.2m有2–4例；GT停留未保留固定2例；累计绝对转角差>30°有4–7例。这些标签**可以重叠**，方向/转向有效性分母完整保存。累计转角差不是严格曲率诊断；没触发标签也不代表轨迹正确，例如位移不足可以产生很大FDE而不触发overshoot。不能用这些事后几何标签宣布单一训练根因。

## D. 是否压掉低速/停留而救回高速？

按预注册GT全程平均速度≤0.1m/s：所有lost的低速计数为0；51个unique lost ID中，11个末3步平均速度≤0.1m/s。二者是不同定义，**不是所有lost都不停留**。

CAL变化7例全程低速0，末段停留2；该完整变化组700窗口的低速75、末段停留158。lost的GT路径内p95速度均值0.881m/s，而rescue在九个模型中只有0.575–0.708m/s，方向与“只救高速、丢低速”的预测相反。主outer scene/target也是lost的GT速度较高；time是反例，lost约0.567、rescue0.569–0.632m/s。样本少且lost/rescue是条件筛选，不作显著性或普遍机制推断。

**真正稳定的速度证据来自集合组成。** 27份输出×overall/stable/transition共81个单元，替换后超过原native p95阈值的速度值数量全都增加；变化组只有13/27增加。全部108单元中94增加，不把重叠group或重复CAL当独立统计。

scene seed42 outer整体：native20 p95=1.4245、common16=1.4620、removed4=1.1219、learned4=1.4427、replacement20=1.4581m/s。common16本身的尾部比learned4更高；**不能把所有恶化归咎于新增候选是最快的**。删除较慢分量、换入相对更快分量改变集合分布，解释了这项固定策略的尾部偏移。
在原native p95固定阈值上，replacement exceedances−native exceedances=learned exceedances−removed exceedances逐项EXACT（例中+3813）。这是计数恒等式，不是把非线性p95变化分解为可相加因果贡献。旧equal-K审计的79/108运动学FAIL、且仅speedp95失败仍保留，其他四项通过不抵消这一项。

## E. logit rank vs marginal utility

所有native20和replacement20都枚举20次单候选删除；coverage/ADE/FDE/J utility均保存，平局utility0，完整人口包括零值。原生重复seed经验证一致，图只取seed42，避免虚增重复数。

主outer原生独占合格路径总窗口493/351/169，其中bottom4分别6/2/3；对应变化组52/83/46中bottom4是3/2/3。这不是“多数低rank都很重要”，但足以反驳“低rank可以安全当冗余删掉”。
变化组scene的bottom4只承担3个独占coverage窗口，却承担约26.46%的全native leave-one-out ADE utility总量、26.27%的FDE utility总量。CAL变化bottom4独占7窗口，native全slot独占181窗口；bottom4 ADE/FDE utility总量17.127/41.477m（按窗口求和，不是平均误差），仍明显非零。

CAL七例J路径物理slot（从1计数）：6×1、14×3、18×1、20×2；它们的**logit rank**分别17×1、18×2、19×2、20×2。slot编号和排名不能混用。原logit rank整体与utility仍有一定关联；结论是它**不是集合冗余指标**，不是它毫无信息。GT utility只能用于事后oracle审计，不能拿评估标签训练部署排序或挑删除策略。

图：figures/rank_utility_outer、rank_utility_inner、rank_utility_original_CAL，均提供SVG/PDF/600dpi PNG/TIFF和SOURCE_DATA.json。14项静态检查全PASS，三图已视觉核验。CAL三列原生曲线相同是同一人口，非三次独立验证。nature-figure影响的是显式导出/单位/分母与视觉核验，未影响数据或结论。

## inline反方审查2（同助手，不是外审）

最强替代解释是组合策略错误而不是生成器无效。当前结果支持它：rescue存在，失去的是原生唯一支撑，集合速度还受删去慢路径影响。但lost=>removed本身是代数恒等式；D大只表明固定几何阈值外，不证明心理意图/模式因果。所有实际lost只有一个好slot，不代表未来不会出现“多个好removed各自U=0、一起删才有害”的情形，合成fixture已保留这个反例。不能仅优化个体LOO utility就保证组合删除安全。

## inline反方审查3（同助手，不是外审）

用户提出的低速压制假设未得到普遍支持：CAL和scene/target救回样例反而更慢，time不同。方向/弧长/停留/转角失败不是排他分类，也不能区分loss与shared head根因。保留native20并增加learned4只是K24，不能冒称固定K20部署成功；几何相近也不保证同路径合格，已存在门槛两侧反例。三seed、重叠窗口、F16已见数据、重复CAL和事后多指标探索使外推受限。所有原major/five-motion1%阈值不改，baseline和formal计数不升。

## 下一步

暂停盲目bottom-logit替换，不做slot/阈值/权重扫描。本轮足以把优先问题收窄为 **native-support-preserving + complementary novelty integration**，不是“再产生更大变化”。

在任何下一训练前，应预注册一个GT-free的集成/保护策略：只以训练人口定规则；保留native20+learned4作为**不等预算参考**；固定K20主对照须独立检查samecandidate coverage、rescue/lost、ADE/FDE/J及全部五项1%运动学保护，并明确不能保证所有oracle支撑无损。不能把当前评估集的U值当部署选择输入。当前没有自动启动新策略试验、模型训练或scorer改动。

来源：RESULT.json、ARITHMETIC_AUDIT.json、PRESERVATION.json、completion_review/REVIEW.json和全108组SUMMARY.json、jobs/*.cases.json。SUMMARY的GT_prevalence.allwindows字段是整个job人口；本报告逐group分母使用row.n，保留原字段而不改锁定科学源码。初次console误用group3查询change，已在报告前按原编码change=2修正；没有更改任何保存科学数组。
