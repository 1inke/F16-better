# 第二项：Recording-time 隔离迁移结果
日期：2026-10-06。状态：完整审计后 ANALYZED；grouped-transfer 研究完成2/3。原 F16 baseline 不变；原 major 全条件 FAIL，不计成重大改进。

## 结论先行
当前位于“相关窗口/上下文暴露依赖与迁移不稳定”的分支，尚不是“loss、credit、motion 都已对齐，只剩结构”的终局。
去掉同录制时间的训练暴露后，整体 ADE/FDE/J 比匹配随机窗口训练更差，支持继续排查迁移和条件分布依赖。但严格完整覆盖的差异区间仍跨0，变化组主区间也很宽。不能称为完整覆盖崩溃、彻底不能泛化，或共享head干扰的唯一因果证据。随机训练自身也未稳定提升变化窗口完整覆盖。
下一项固定场景族曝光对照，不改 head、loss 或 scorer，第三项完整后必须先做三项总review。

## 完成与证据
- 同一唯一worker48928与exit双SUCCESS：10:34:02.309368→11:05:50.4867HK，31m48.177s。唯一完成检查11:09:25.1357043HK；半程11:04:52.9887485已经完成，不重复查询。
- 六fresh训练各80epoch，18 checkpoint，57科学保存任务114输出，216 arm/seed/policy/split/group行，180主time-block+180target-track敏感性区间。不是57独立训练。
- [RESULT](results/RESULT.json)、[ARITHMETIC](results/ARITHMETIC_AUDIT.json)、[FROZEN](results/FROZEN_AUDIT.json)、[PRESERVATION](results/PRESERVATION_AUDIT.json)全部PASS，588锁定科学源/33551旧文件不变。
- [CPU completion review](completion_review/REVIEW.json)：57全部保存质量列/原SciPy GT-free完整聚簇/216组pooled运动学/360区间 EXACT，误差0；114outputs/18models/588source/33551old初末SHA PASS。无torch、模型前向、反向、optimizer；不是全模型或梯度第二次回放，也不是外部独立审稿。

## 对照是什么
同一566 outer ID，random与grouped训练各16104，sequence×原3lossgroup严格匹配，共同12544训练交集拟合双方统计，inner2134共享。原CAL2879只事后报告，未选checkpoint；primary为固定epoch80 final，constrained/legacy是secondary。
50 SOURCE indices固定块，完整20索引窗口。random outer566同sequence时间重叠/564同target暴露；grouped时间重叠0但仍有227远时间同target暴露。time-disjoint不是target/context完全独立。
未知123全ID保留；已知时间比较的边界4720/unused3548原因记录，不把它们称为困难或不可预测。原FIT/CAL是ordered prefix/suffix，不是随机划分。冻结F16原先见过这些数据，只测newhead适应迁移。

## 主结果：同一条完整candidate ADE≤.2且FDE≤.4
| outer组 | N | native | random seeds1/2/3 | time-grouped seeds1/2/3 |
|---|---:|---:|---:|---:|
| overall |566|303|323 / 320 / 316|286 / 304 / 312|
| stable |306|199|212 / 207 / 202|199 / 202 / 201|
| transition |118|44|52 / 55 / 52|46 / 49 / 50|
| change |142|60|59 / 58 / 62|41 / 53 / 61|

Overall平均coverage：native53.53%，random56.48%，grouped53.12%。grouped-random -3.357pp，主14-time-block×3-seed探索CI[-7.843,+.885]pp，track敏感性[-7.182,+.506]pp。coverage差异不能稳称非零。
Change平均coverage：native42.25%，random42.02%，grouped36.38%。grouped-random -5.634pp，主6-time-block CI[-13.732,+5.556]pp；57-target敏感性[-14.182,+2.800]pp。方向令人担忧，但仅6变化块，seed3仍61>native60，是必须保留的反例。
Change rescue/lost：random14/15、13/15、16/14；grouped5/24、10/17、14/13。groupedseed1的大量lost与seed3小净改善不能合成“所有seed都失败”。

## 连续误差与完整覆盖不能混为一谈
| outer | native ADE/FDE/J | random三seed均值 | grouped三seed均值 |
|---|---|---|---|
| overall |.178334 / .310364 / 1.011506|.176697 / .294188 / .983334|.186103 / .324516 / 1.046245|
| change |.234363 / .424106 / 1.327296|.230464 / .392498 / 1.278261|.244372 / .440236 / 1.381906|

Overall grouped-random：ADE+.009405m CI[+.002544,+.014728]；FDE+.030328m CI[+.013495,+.043065]；J+.062911 CI[+.027064,+.091967]。这几个主探索区间不含0。
Change grouped-random ADE+.013908、FDE+.047737、J+.103645，但主time-block三个区间均跨0；target敏感性FDE/J不含0，ADE仍跨0。不得将敏感性替换主分析，只挑有利区间。
Random change FDE平均下降约7.45%，ADE约1.66%，J也改善，但完整覆盖近乎不动。这是连续逼近而未稳定跨门槛的证据，不足以独断objective mismatch或排除objective问题。Q是同slot hard ADE+.5FDE，不是所有候选平均。独立minADE/minFDE不能拼成一条好路径。

## 分段、有效候选、选择与反例
同一Jclosest完整slot，change早/中/晚平均random约.11251/.25443/.34698，grouped约.11372/.26134/.38235；native约.10945/.25092/.36732。后段差异更明显，但该路径会在不同输出间变slot；不是固定slot因果，也不是拼段oracle/decoder根因。
Overall有效good簇native值见保存native_outer，random458/470/456，grouped444/422/469；rawgood575/598/574 vs546/556/571。distinct约19.70–19.78，不能从少数簇减少断言整体collapse，也不是越多越好。
Inheritednativeargmax选中good outer随机144/147/149 vsgrouped136/145/139；它只是旧slot，不是对refined轨迹重新评分的概率。生成端尚未稳定覆盖，scorer不是下一优先步骤。
Per-scene change outer只有hotel5/university128/zara9：hotelnative5→random5/5/5/grouped4/4/5；university54→52/52/56 vs35/48/54；zara1→2/1/1 vs2/1/2。zara和hotel太薄，不能可靠scene推断或宣称任何场景完全不能迁移。详见[PER_SCENE](completion_review/PER_SCENE.json)。

## Inner / 原CAL以及训练损失
Final inner change604 native236：random244/219/236，grouped237/240/244。Grouped在独立inner并非完全无改善。
Final originalCAL change700 native243：random232/219/221，grouped222/243/235。Groupedseed2等native，不能说三seed全低；CAL总体groupedseed2 1502>native1490也是反例，但变化组没有净提升。
Constrained originalCAL change：random244/244/245，grouped245/246/242；选中epoch random10/6/8 vsgrouped9/8/9，近原生的小改动。Outer constrained change60/60/60 vs60/60/59。内层选择成功/局部CAL小增不能提升为primary成功。
六训练loss降17.88–20.31%（batch legacy分母），grouped降幅更大但outer final更差。不能跨训练集直接以loss排序，也不能把训练loss与population原quality混淆。所有72聚合和完整policy/group/CIs来源均保存于RESULT；CPU全部重算，没有只审final。

## 运动学与major
216 policy-group行中214五项≤native1% PASS；唯二FAIL为grouped seed1 final change：
outer speedp95 +1.182903%，inner speedp95 +1.289705%。不将1.18%约成可接受，也不以accelmean投影保证替代speed/jerk。
原major仍FAIL：覆盖不稳/三seed不一致、ADE/FDE各5%并不同时通过、保护/运动学不全满足、native/freshjoint证据不完备。F16 baseline不变。
生产peakprivate3.901711GiB/RSS2.643093，minphysical15.794323/commit15.403229；补充private.206757GiB。都满足process6GiB及systemphysical+commit各6GiB余量，不替换为预检峰值。

## Inline counter-review2：解释反证（同助手，非外审）
不能说random很好而grouped彻底失败：random change覆盖59/58/62对60、groupedseed3 61、inner变化改善/CALconstrained小增都反驳。Coverage主区间跨0，与continuous overall CI支持不冲突。训练暴露、条件分布、有效样本多样性和有限优化变化耦合，匹配sequence×lossgroup仍不是纯sharedhead因果。只冻结F16未证明wholemodel未见泛化。

## Inline counter-review3：决策审查（同助手，非外审）
不靠训练loss降幅选择结构或loss，不能在薄场景子组上选特征/阈值；不以track敏感性替换time-blockprimary。保留唯二speedFAIL和反例，不放宽原major。后续固定zara family-exposure对照只回答新head family transfer；因CAL是university不能把其结果直接解释成CAL特有shift。第三完成后fullthree review再决定mapping/conditioning或分布分析，禁止第四块宽/专家数/系数/LR/epoch变体和未授权scorer/olddecoder训练。

## 下一步
新[experiment3_scene_family/DESIGN](../experiment3_scene_family/DESIGN.md)固定zara（最大非原CAL family，metadata理由而非性能排序），两arm匹配global3lossgroup预算，scene-unseen无任何zara vsrandom可见zara。不能EXACTsequence匹配，明确distribution/exposure耦合。
Metadata-only规模：train16337各/common12183/outer1561(change211)/inner1773/3outerrecordings309tracks；原source/design小范围SHA前后PASS。这不是完整oldSHA/GPU/ID/resume预检，也尚无正式worker/ETA。11:45HK单次实施预检节点；全gates后才唯一OS启动与新50%→completion节点。研究当前2/3，第三尚未start；无重复旧worker或原结果覆盖。
