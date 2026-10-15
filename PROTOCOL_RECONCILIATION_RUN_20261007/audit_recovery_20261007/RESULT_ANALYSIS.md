# Protocol Reconciliation Run：完整结果分析

## 结论

九个 canonical 训练及27份保存输出现已完成审核。统一协议后，三个 grouped 设置均显示“小幅可迁移的候选补充”，原 CAL 也有小幅覆盖收益。因此不能继续把先前不同训练协议下的失败直接解释为“行人/场景之间完全不能迁移”。但本轮没有通过原重大收益与全组运动学门槛，baseline 不变，也不证明已经排除了相关窗口依赖。

本报告由 AI 辅助计算与分析，采用 academic-research-suite 的证据核验和反方检查；反方2/3由同一助手完成，不是外部独立审稿。来源：本目录 RESULT.json、BOOTSTRAP.json、ARITHMETIC_AUDIT.json、PRESERVATION.json；独立补充 completion_review/REVIEW.json 与 SUMMARY.json；父目录 DESIGN.md、protocol.json、九个 run_manifest/checkpoint/history/assignment_stats；旧 winner-switch 证据见父 OLD_WINNER_SWITCH_ANALYSIS.md。

## 1. 执行与审核状态

- 原训练九个run各25epoch，共225行，全部已保存。父worker38996在最后保存算术审计因系统物理余量低于6GiB停止，失败来源原样保留，不是九次训练未完成。
- 新恢复worker21492：21:25:49.063989启动，21:27:12.151082双SUCCESS，约83.087秒；21:33:05.0605971人类请求触发唯一迟到完成核验，taskReady、零trigger、无进程、27done、新stderr0。没有声称按时执行21:27半程，也不再重复旧节点。
- 恢复仅保存数据算术，无模型构造/前向/反向/optimizer/RNG恢复/重训。原27份完整质量、SciPy complete-link聚簇、全部五项pooled运动学与assignment复算，独立最大误差0。
- 完成补充核验九个 checkpoint 绑定完整source/split/statistics/objective与225行history；train-only统计从缓存encoder重算，三组同seed初始化SHA一致。72条bootstrap记录、288个标量CI独立复算最大误差0。385个锁定相关文件初末SHA不变，继承273原digest；不是所有历史34k文件核验或旧F16模型回放。
- 恢复生产private峰值1.061825GiB/RSS .667332，physical最低18.453197/commit23.074764GiB；补充private .995651GiB，资源合规。

## 2. 主结果：各自 outer，固定epoch25，三seed均值

完整好候选要求同一条完整路径 ADE<=.2m AND FDE<=.4m。下面比较native20与保留其全部路径再新增4条的full24；不是等候选预算比较，不能称生成器已替代原生候选。

| grouped划分 | outer N/变化N | overall覆盖 native→full24 | 变化覆盖 native→full24 | 变化ADE/FDE收益 |
|---|---:|---:|---:|---:|
| scene_family |1561/211|66.62%→69.36%（+2.73pp）|38.86%→41.71%（+2.84pp）|3.25% / 5.25%|
| target_track |1035/253|60.39%→63.57%（+3.19pp）|43.48%→46.25%（+2.77pp）|1.90% / 3.87%|
| recording_time |566/142|53.53%→56.30%（+2.77pp）|42.25%→43.90%（+1.64pp）|1.49% / 2.20%|

变化组真实计数（三seed42/43/44）：scene native82→89/88/87；target110→115/118/118；time60→62/61/64。救回分别7/6/5、5/8/8、2/1/4。每seed均有补充，但绝对数量小。

同grouping target-track×3modelseed探索bootstrap（2000次）：变化覆盖增量CI依次[+.85,+5.25]、[+1.10,+4.74]、[0,+4.46]pp；时间组触及0，不算严格排除0。三项变化 ADE/FDE/J连续改善CI均为负。CI不控制所有多重比较、不把三seed当独立场景，也不把不同outerIDs跨G强配对。总体覆盖增量CI依次[+2.00,+3.76]、[+2.16,+4.38]、[+1.50,+4.30]pp。

## 3. 原CAL：不是本轮checkpoint selector

相同700个变化窗口native good=243（34.71%）：

| head训练划分 | full24变化good三seed | 平均覆盖 | 增量 |
|---|---:|---:|---:|
| scene |254/261/258|36.81%|+2.10pp|
| target |257/260/261|37.05%|+2.33pp|
| time |256/257/262|36.90%|+2.19pp|

各组CAL变化覆盖探索CI为[+1.19,+3.15]、[+1.27,+3.51]、[+1.17,+3.39]pp。CAL变化ADE/FDE均值收益仅scene2.04/3.85%、target2.00/3.72%、time1.75/3.33%，仍低于5%双门槛。各组overall CAL full24good均高于native1490。不能套用旧persistent架构“CAL每次都丢失”的结论；本轮采用不同且人类明确批准的canonical Bo4算法，未给出这些协议差异各自的因果贡献。

## 4. 覆盖、独特好簇、运动学不能混为一谈

- full24保留native20：lost=0、覆盖非减由集合包含保证；这是有限的新增救回证据，不是学会保护已有成功窗口。需要后续等预算验证才能声称候选效率提高。
- outer full24有效好簇均值overall约1.1576/1.0354/.9405，变化约.6224/.6298/.5892；distinct约22.7–23，未出现20条复制成1簇式塌缩，但簇数量不是越多越好。
- learned4单独outer变化覆盖：51/51/48（211）；49/47/52（253）；23/21/26（142）。它能贡献部分好路径，但K4与nativeK20不公平，不能据此独断head能力不足或优于native。
- full24全部27输出×4groups共108项五运动学联合检查：93PASS、15FAIL，失败指标均为accel_mean，最高+3.2668%。outer34/36PASS：target seed42/43 transition accel_mean分别+1.5982/+1.9723%；scene与time outer全组PASS。其余失败在inner及CAL，多为transition；不能只报告change组全部PASS。learned4另有41/108组联合FAIL，存在速度p95与加速度均值问题。
- major全部FAIL：变化ADE/FDE没有同时>=5%，全seed全group运动学1%保护未通过；且没有新的等预算fresh对照证明。即使覆盖CI为正，也不能升级baseline。

## 5. winner-switch 与训练loss

当前九run保存的训练batch-preupdate switch率第1epoch约3.09–3.29%，第25epoch约1.74–2.05%，全过程范围约1.72–3.56%；total从第1到第25epoch下降30.82–33.55%。这些是batch样本加权训练记录，不是人口评估loss，也没有每epoch完整路径快照，不能构造“每下降一点loss就救回多少窗口”的因果曲线。

最终四条learned路径：outer overall switch约2.43–3.89%，change约1.41–4.23%；其他条件组可更高（target outer transition7.30%），不可只报总体。旧九head同输出离线overall约1.5–5.1%，同样没有20–40%大规模责任交换证据。小switch不证明两个算法等价：FDE权重、diversity、clip、梯度尺度及训练动态仍可不同；J最近不等于最可修，也不以switch一项决定是否改loss。

## 6. 当前走到哪条研究分支

证据支持“受限grouped迁移下存在小幅可迁移互补”，而不是“grouped完全不能泛化”。但它尚不满足人类分支中的“覆盖稳定且运动学无明显恶化”：过渡组物理保护失败，且full24预算扩大。也没有充分证据支持优先更换hardwinner assignment；scorer更不是当前优先，因为候选补充能力和预算效率尚未确认。

统一P成功解决了源码/训练协议归因blocker，未完全回答grouping本身的因果问题。本九run没有同canonical random arm，而且scene/target/time的训练量、outer总体、场景组成、数值统计不同。不能宣称已排除窗口相关性、场景分布差异，或给grouping难度排序。F16已见原数据、CAL单university、相关窗口和仅3seed限制保留。

建议下一步先明确一个窄的等预算控制设计，用保留输出评估新增路径是否优于等量native备选；若要直接回答相关窗口依赖，则必须同canonical下做同outer/训练预算/组成匹配的random-versus-grouped，不能横比当前三项outer百分比。运动学应同时看新增路径及受保护transition，不进入scorer、MoE或loss扫参。这里只提出方向，不启动新实验；人类本阶段明确“不扩张”。

## 7. ARS inline反方检查2：分析后的反证

Verdict PASS_WITH_LIMITED_CLAIMS；没有未处理的证据完整性critical问题，但以下major限制不得删除：

1. 最强反驳：多加4个候选自然提高oracle覆盖，而不是训练真正可迁移。本结果有新增救回但缺等预算对照，因此仅说互补，不说候选效率/重大进步。
2. scene变化FDE>5%而ADE<5%，time覆盖CI触0，transition运动学失败。把变化组正数选出来宣布成功属于选择性报告；结论维持majorFAIL。
3. 相同P但不同IDs/训练量，未跑canonical random，无法证明related-window依赖消失或pureGcause。转向结构/分布假设需后续精确控制。
4. 小switch不排除loss作用，覆盖收益不能证明未来概率/scorer可校准，学习器K4也不能公平与K20比。

## 8. ARS inline反方检查3：最终报告审核

Verdict PASS_FOR_EXPLORATORY_REPORT_NOT_BASELINE；报告含完整分母、每seed与反例、预算及相关性限制、全部组运动学失败和资源恢复来源。删除scene最强来源后target仍有小补充，time更弱；反向假设“多数增益来自增加K”仍可信；外推独立场景不成立；意义是协议可追溯与小互补，而非SOTA/decoder根因。AI披露和源路径齐全，同助手审核不是外审。

报告时发现新completion汇总脚本把outer36项motion全部PASS错误写为常量true；原完整failure列表和科学metrics正确，已保存错误脚本/汇总到 completion_review/failures/outer_motion_summary_flag，按真实失败数修正并重做CPU核验。最终outer34PASS/2FAIL。未更改锁定科学源、旧结果、阈值或任何模型，此为助手报告代码错误。

## 9. 后续执行状态

本九run协议归因闭环ANALYZED，三groupings mandatory fullreview见 FULL_REVIEW.md；原训练FAILURE保留，恢复双SUCCESS使保存结果审计完整。停止本轮重复自动核验，不启动第十个run/新训练，不改变F16baseline、历史cycle11或研究目标完成状态。
