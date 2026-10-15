# 原生全路径相似变换：结果分析

结论：**执行与审计成功，但主要科学目标 FAIL。安全且非零，不等于补出完整好候选。**

## 1. 完整执行与证据范围

生产10:12:03.882747–10:16:49.552609 HK，耗时4分45.67秒。唯一半程检查10:17:17.8864475发现双SUCCESS，无需另查完成节点。六组25epoch共150行、六checkpoint、18科学保存输出齐全。独立CPU复核全部保存质量、原SciPy GT-free全K20 complete-link聚簇、五项pooled运动学、相似系数/射线alpha、固定16路径、bounds、assignment、36项bootstrap/144标量CI及1344相关source/103输出SHA，误差均0。保存算术不是所有模型/梯度再次回放，不是外部评审。1344是相关来源，不是历史全部34k文件。

射线对照九份输出、每份11数组与上一物理射线处理bitwise EXACT：fresh重复不是新增独立泛化证据。生产private峰值2.858GiB；物理/commit最小14.170/23.802GiB，资源标准未降低。

## 2. 完整覆盖：同candidate ADE≤.2且FDE≤.4

每格三个seed42/43/44；固定epoch25为primary；inner/CAL为事后secondary。

|split/group|N|native good|ray good|similarity good|similarity平均ADE收益%|FDE收益%|
|---|---:|---:|---|---|---:|---:|
|outer/overall|1035|625|625/625/625|625/625/625|0.01462|0.01620|
|outer/stable|604|434|434/434/434|434/434/434|0.00030|0.01177|
|outer/transition|178|81|81/81/81|81/81/81|-0.00375|0.07282|
|outer/change|253|110|110/110/110|110/110/110|0.04396|-0.00990|
|inner/overall|1856|932|932/932/932|932/932/932|-0.01276|-0.00133|
|inner/stable|978|601|601/601/601|601/601/601|0.01448|0.01806|
|inner/transition|381|165|165/165/165|165/165/165|-0.01448|0.00235|
|inner/change|497|166|166/166/166|166/166/166|-0.04344|-0.02567|
|original_CAL/overall|2879|1490|1490/1490/1490|1491/1491/1491|0.02268|0.05721|
|original_CAL/stable|1535|994|994/994/994|994/994/994|0.03877|0.08108|
|original_CAL/transition|644|253|253/253/253|254/254/254|0.01688|0.03414|
|original_CAL/change|700|243|243/243/243|243/243/243|0.00875|0.04981|

外层与inner所有组均无rescue/lost。CAL只多一个transition窗口，ID **28854**，三个seed均为同一个窗口；变化组仍243/700，没有新增变化好候选。不能把它算成三次独立rescue。所有丢失均0。

GT-free有效好簇：外层与inner无增加；CAL仅同一个transition窗口带来一个好簇增量。不是候选越多越好，不能以distinct取代完整覆盖。

## 3. Primary变化误差与CI

外层change native ADE/FDE/J=.249146619/.429378298/1.390568401；similarity三seed平均=.249037083/.429420801/1.390300135。ADE约改善.04396%，FDE反而恶化.00990%。远未达到ADE与FDE各5%同时收益。

similarity−native的paired target-track+3seed探索95%CI：ADE[-.000418119,+.000201203]m；FDE[-.000578653,+.000769712]m；J[-.001906758,+.001413063]，均跨0。所有primary组连续指标CI均跨0。相同二值成功集合的coverage bootstrap[0,0]只表示此保存样本上的差值恒0，不是总体等效证明。

反例保留：CAL overall FDE/J及stable连续误差有很小的探索CI不跨0，但变化组CI仍跨0；这些secondary信号不能替代primary，不能宣布全面无效或突破。全部36CI/144标量在BOOTSTRAP.json，12组similarity-vs-native及完整组表在RESULT_ANALYSIS.json；每seed全部72组指标、temporal Jclosest单slot与motion在completion_review/SUMMARY.json。

## 4. 非零与运动学

similarity九份输出中，全部被修四候选每条最大残差≥1mm；逐点残差中位约7.16–8.18mm，最大73.28–105.29mm。ray所有候选最大残差<1mm，中位0。因此本次不是又一次全零输出失败。

全部72 arm×seed×split×group cells的五项原literal运动学1%保护通过，**包括**speedp95、accelmean/p95、jerkmean/p95。similarity外层change加速度均值约下降.203–.210%；不是所有指标都下降，例如外层overall speedp95有约+.0179%的小增幅，仍在原保护内。逐点近零导数的roundoff证书不是科学评价epsilon；真实pooled五项标准没有放宽。

该映射只能对整条原生路径做共同rotation/scale，不能创造新的停走/转向时间形状，静止anchor仍静止。安全非零说明可行几何不必塌到0，不说明能覆盖缺失变化。

## 5. Loss与科学目标

similarity epoch1→25原四路径batch加权训练total下降.179–.202%；ray下降近0。此分母不是fullK20人口质量，也未保存完整FIT覆盖，不能声称FIT完整成功而CAL不迁移。当前证据仅表明：这个安全非零、时间形状受限的映射，少量训练loss下降没有转化为primary变化完整覆盖提升。不能单独据此锁定objective mismatch、credit错误或共享head唯一根因。

## 6. Inline反方2（同助手，非外审）

最强反驳：global similarity只有受限几何自由度；失败可能是可达集合太窄，不代表candidate generation整体不成立。相同25304名义参数并非相同有效容量；bounds、固定bottom4、conditioning、quality credit与有限优化均是替代解释。ray bitwise重复也不能累计成独立泛化证据。

## 7. Inline反方3（同助手，非外审）

最强正面反例：CAL一个transition获救且部分secondary误差CI不跨0。仍不足以宣布major：primary变化覆盖零增、FDE微坏且CI跨0。F16已见数据、target非context disjoint、三scene/seed、窗口相关、重复CAL和adaptivebottom4限制不变；不推断意图、理解运动、独立wholemodel泛化或decoder根因。

## 8. 决策与下一方向建议（不自动执行）

本次ONE授权闭环完成，新study1/3；不为了凑3自动再训。baseline/historycycle11/未完成goal均不变，旧六小时窗口不续开。本次过期followup移除，不再查任何完成PID/task/log。

下一方向若继续，优先研究**保留原生支撑且能改变时间形状的可行参数化**；先用精确可达性设计判断是否能触及原生coverage hole，再决定训练。不是再扫rotation/scale系数、bound、degree、width、LR或epoch，不提前改scorer/softwinner，不把上述建议当已获准的新训练。

