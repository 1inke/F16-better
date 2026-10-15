# T3：固定机制是否跨时间与场景迁移？

版本1.0；DESIGN ONLY。只在T2的S_T2和Q通过后开始，先time，满足条件再scene。不一次启动两个隔离任务。

## 1. 不是第三次在旧outer上找优势

T1/T2只检验现有target隔离人口。T3检验同一固定机制能否在不同分布继续产生训练增量。必须重新在每个新split的训练集合拟合**词典、标签、统计与两臂head**；不能把target训练的中心/统计/网络直接搬进含有其训练身份的另一个outer，称为无泄漏迁移。

冻结P2来自旧F16，旧F16已见原pool的限制依然存在。这里只检验newhead适应的迁移，不是whole-model独立未见场景泛化。

## 2. 两个已有划分，只复用ID，不复用旧训练协议

| 子阶段 | 已有来源 | 采用的grouped训练与评价人口 |
|---|---|---|
| T3-time | [recording-time设计目录](../GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/DESIGN.md) | grouped train16104，outer566/change142，inner2134 |
| T3-scene | [scene-family设计目录](../GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/DESIGN.md) | non-zara grouped train16337，zara outer1561/change211，inner1773 |

original_CAL2879均只posthoc。按原身份审核和保存manifest定位精确ID并在新manifest绑定SHA；上表数量是既有记录，实施时必须核对，不能只凭数量重新抽出“差不多”的split。旧random arm在本阶段不训练，不拿旧random训练结果与新模型做单因素比较。

time的50单位是source index，不是秒或未经核实的视频frame；grouped排除任何行人与保留时间块相交的完整20索引区间。未知时间资格与边界排除沿用原记录。time-disjoint不等于target/context独立。

scene固定zara，无效果排序择族；旧F16见过zara；只能评价一个held-family，不能推断所有场景。scene训练场景组成与target/time不相同，不跨不同outer ID硬配对或比较绝对覆盖高低来断言grouping因果。

## 3. 每split重新做同一个双臂对照

A=新split的dynamic_winner，B=新split的fixed_motion_responsibility，均提供真实P2，完全沿用T1架构、K20、聚类算法与seed、loss系数、25epochs与42/43/44。

每split的A/B共用**该grouped训练集全部样本**产生的词典/标签/统计。不是沿用旧研究random/grouped的common intersection统计，因为此处两臂训练ID完全相同。不得混用旧80epoch/physical loss/7class权重。

每个split6runs/150epochs/24finaljobs+自己的4份P0。原型在两个split可不同，这是同一拟合算法在不同训练数据的合理结果，不声称数值相同。总计最多12runs/300epochs；预算逐个双臂估算，不隐含批准长任务。

## 4. 继续与停止

time完成后，要求该split上的S_T1规则（B−A≥5pp且三seed正/校正下界正，B−P0正且校正下界正、保护不劣）和共同Q全部通过，才考虑scene。

scene同样报告全部规则，不因为样本小、只有一种族或native难超就改标准。T3主family为两split的B−A、B−P0、B−native C/ADE/FDE/J共24项，已包含在全系列48项中。T1/T2各12项加T3共24项，总48；未触发的阶段不缩小除数。

完成报告同时列target/time/scene每个独立人口的效应和分母，不能只合并总窗口得到一个漂亮平均数。

| 结果模式 | 研究判断与动作 |
|---|---|
| target正、time负 | 时间/上下文分布转移仍未解决；停止scene自动推进 |
| target/time正、scene负 | 存在跨族适应限制，不称普遍运动理解 |
| 三者同向且全门通过 | 支持该完整模式机制的迁移价值，再检查统计/历史自适应局限 |
| 仅FIT越来越好 | 拒绝“再训久一点即可解决”的自动延训 |

无论是否通过，原native baseline保持到全部major与相关性审查完成；候选生成研究不能自动变成scorer训练或旧F16解冻。
