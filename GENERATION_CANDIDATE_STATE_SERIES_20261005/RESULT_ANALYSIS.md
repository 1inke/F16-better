# 候选状态输入实验：完整结果分析

2026-10-05，candidate-state研究第一项完成，1/3。结论：严格覆盖小幅增加，但主要均值/三seed/置信区间和运动学未形成可靠改进，major FAIL，不升级baseline。

## 完成与可靠性

唯一半程检查09:05:10发现07:52:55已完整完成；实际23分40.913秒（07:29:14启动），不是原150–240分钟预计。worker和results/exit均SUCCESS，6新训练各80epochs/18policy输出/1516显式unique IDs/有限20×12×2，independent_audit、saved_metrics_recompute PASS。保存sample/nativeALLpolicygroups/72perseedgroups/24aggregates/96bootstrapCIs/temporal/pulse分解/frozen/source364/old27838审计完整。旧worker40032已完成，不重复PID查询/半程/旧截止或重训。

新CPU复核[REVIEW](completion_review/REVIEW.json)重算全部18组minADE/FDE、同路径质量和完整GT-free聚簇；min误差4.613e-7≤1e-6，原27838文件/364源SHA前后和18模型SHA PASS。按保存几何重建同一注册SciPy complete-link算法，不冒称外部独立审稿/另一个聚簇算法/所有模型再次前向。补充不含torch/model/optimizer；参考输入来自原锁定可视化数组，不新数据或阈值。

## 主要均值和不确定性

| change组 | minADE | minFDE |
|---|---:|---:|
| 原F16 | .372174740 | .696326435 |
| fresh maskedstate对照 | .369740963 | .692263663 |
| actualstate | .367063254 | .686794281 |

actual相对native收益1.3734%/1.3689%，相对freshcontrol约.7242%/.7901%，均不足5%。nativechangeADE CI[-.0150672,.0041136]、FDE[-.0377758,.0166328]；freshADE[-.0122258,.0057267]、FDE[-.0318403,.0182975]；native/freshjoint都跨0。seed1/2均值较好，seed3ADE .371964335、FDE .704945803，相对fresh .371766359/.694758713均恶化；nativeFDE也恶化。不得用总体平均遮盖seed3。

总体ADE/FDE收益.580%/.702%；stable .552%/.508%；transition ADE害.393%、FDE收益.369%。原protected均值1%保护通过，不等每项major通过。全部legacy/final组和CI保留在[summary](results/summary.json)，不按DEV重选policy。

## 同一高质量候选与去重：比独立min更重要

严格ADE≤.2m AND FDE≤.4m，必须同一完整candidate。三个seed是同1516IDs的重复预测，不能相加成人数。

| 指标 | native | masked seed1/2/3 | actual seed1/2/3 |
|---|---:|---|---|
| 有至少1好候选的样例 |733|732 / 737 / 735|744 / 742 / 749|
| clear-event469中覆盖 |92|91 / 92 / 90|99 / 93 / 100|
| GT-free有效好簇总数 |929|932 / 937 / 929|938 / 942 / 828|
| inherited argmax好样例 |281|281 / 280 / 281|285 / 283 / 289|

actual相对native救回30/22/41，损失19/13/25，净+11/+9/+16（约+.73/+.59/+1.06个百分点）；相对fresh救回31/22/44、损失19/17/30，净+12/+5/+14。clear native净+7/+1/+8，不能混成原change223分组。只增加覆盖不保证大幅人眼改善。

seed3 raw好候选1177（native1146）而effective总数828（native929）；135个原有多有效好簇窗口变成一个，整体distinct数变化137个窗口。并非所有K20塌缩：seed3仍1314/1516窗口有20个distinct簇，160个有18簇。它表明好候选的局部集中/几何跨过聚簇边界，并不自动意味着坏；必须与覆盖增加、丢失、路径变化和运动学合看，不能因“数量下降”否定合理集中，不能因raw增加宣称更多独立高质量模式。

## 运动学、时序和学习迁移

pooled ALLK20 dt.4全组平均加速度害5.23%–10.66%，全部三个seed失败严格1%保护。seed3speedp95总体+1.186%、change+1.389%；change jerkmean+1.074%也失败。其他p95/硬运动/重复/coverage可以通过，不能替代全项1%标准。数值来自[quality summary](results/quality/summary.json)，不是window p95平均替代global pooled p95。

FIT训练loss first→epoch80改善actual13.20%–13.89%、masked6.11%–6.86%；这是原balanced总loss，不是FIT ADE/FDE收益。三actualfinal CALchangeADE/FDE均比epoch1更差，原CAL选择29/25/35epoch，不改80budget或DEV选checkpoint。DEV actualfinal change .382759690/.729853272也恶化，freshfinal ADE/FDE CI探索性正；总体top3改善不能遮盖change完整路径恶化。支持已有学习/泛化差距，不能从loss曲线证明唯一rootcause。

quality早中晚使用同一normalized-adequacyclosest slot；temporal另用同一ADE+.5FDE jointoracle slot，不分段拼oracle。onset99代表缺事件、direction只turn条件才解释；这次不据旧混合direction平均声称意图识别。clock参数变化/endpoint分解只描述用到了head，不等真实onset/duration被学会。

## 资源与局限

生产peakRSS2.579GiB/private3.363，最小physical15.411/commit13.749GiB，全部资源guard通过。补充peakRSS.118/private1.541GiB，无GPU训练，352秒主要含全SHA检查。17GB可用hostRAM约束继续保持，process6GiB/systemphysical+commit各reserve6GiB，不改OS/用户程序。

所有旧F16/whole original decoder/scorer权重buffers固定、immutable nativeK20与旧loss/guards/data不变。预测state不是新增现实信息；mask导致有效容量/优化差异，不能以名义同参数作纯因果证明。原logitargmax只是继承slot，不是新head最大概率方案。3seed/3scene/轨迹窗口重叠/nativeF16已见/FITCAL/CAL university/重复ETHDEV探索限制保留，无SOTA/心理意图/独立泛化/decoder唯一根因结论。

## 反方2、3与下一步

academic-research-suite反方检查由同助手inline完成，非外部独立评审。反方2：必须同时报告净覆盖增量、seed3有效数量下降、全motion和FITCAL过拟合，避免单指标cherry-picking（已落实）。反方3：最强反驳是active state branch容量/优化变化和重复DEV造成偶然收益；native/freshCI、seed3失败及反证已保留，不升级结论。旧GTwitness、freeincrements/timequery/velocitybasis、mode/conditioning/social和tinyFIT失败均不能忽略。

下一合法单因素已经在[全新DESIGN](../GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/DESIGN.md)预注册：双方state分支都active，原始state32 versus 对30个速度/终点分量减去最后已观测速度reference，共同raw FIT scalers，后两统计不变。原head/profile/loss/数据/冻结保持。先真正实现并通过GPU、双armreal2epoch exactresume、SHA、内存和吞吐，然后独立OS启动；09:45是实施规划节点，不是生产开始或完成ETA。遇到实质科学/资源/ID/source/权限阻碍STOP等human，不无限局部扫参。

来源：本folder results/summary、per_seed_metrics、independent_audit、saved_metrics_recompute、quality/pulse/temporal、completion_review/REVIEW与RESOURCE_USAGE。AI辅助分析和inline反例复核已披露；无外部数据/新增人工判断。

