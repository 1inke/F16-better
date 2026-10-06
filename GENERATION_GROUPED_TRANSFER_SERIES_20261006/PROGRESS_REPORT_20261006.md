# Grouped-FIT 迁移研究阶段进展报告

**日期：** 2026-10-06（香港时间）  
**范围：** 从首项正式实验“随机窗口训练 vs 按行人分组训练”开始，至本报告核对时的第三项场景族实验状态。报告汇总已保存的实验和审核记录；不代表新的模型训练、重放或独立外部审查。

## 核心结论

两项已完成实验都显示，new head 的迁移表现会随训练窗口的身份和时间暴露而变化；第二项中，按时间块隔离的模型在相同外层窗口上的整体连续误差高于随机窗口模型。但“随机训练明显提高完整好候选覆盖、分组训练导致确定性覆盖崩溃”并未得到统计上稳定的证明：两项研究的覆盖差异区间跨零，第二项的变化组只有6个时间块，且出现seed反例。

因此，目前最稳妥的研究分支是**继续辨别训练暴露与场景分布对 new head 迁移的影响**。证据还不足以将问题归因于单一的 loss、共享 head、decoder 或 scorer。第三项实验已预注册为固定 zara 场景族暴露对照，但截至报告核对时，运行记录仍显示待实施，尚无完整预检或正式训练结果。

## 研究问题与判定标准

这组实验检验：在原 F16、原 decoder 和 scorer 冻结的情况下，改变 new head 的训练窗口暴露方式，是否影响其在匹配外层窗口和原 CAL 上形成完整正确候选的能力。它检验的是**new head 对既有数据的适应与迁移**；原 F16 已见过这批数据，因此不能据此宣称 whole model 对独立未见场景的泛化能力。

首要质量标准是至少有一条**同一候选**同时满足 ADE≤0.2 m、FDE≤0.4 m。分别取最小 ADE 与最小 FDE、但来自不同候选的情况，不计作完整覆盖。连续联合指标 J=minₖ max(ADEₖ/0.2,FDEₖ/0.4)，J≤1 表示该候选达到同一门槛。随后检查 GT 无关的全 K20 去重有效好候选与 rescue/lost；候选数量本身不作为优化目标。旧 native argmax 只是继承的槽位选择，不是为修正后候选重新训练的 scorer。

两项对照都固定训练预算为每臂 3 个 seed、每个 80 epoch，原 F16/decoder/scorer 权重与缓冲值冻结。无未来 GT、未来分组或 logits 参与预测前向。原 CAL 仅作事后报告，不用于这两项实验的 checkpoint 选择。原有 major 标准和严格运动学保护标准均未放宽。

## 实验一：按行人身份分组

**设计。** 两臂各训练 18,909 个窗口，按 sequence 与原三类 loss group 精确匹配；使用相同 outer 1,035 个窗口与 shared inner 1,856 个窗口。Random 臂让 1,028 个 outer 目标身份出现在训练中，已知时间中有 1,024 个 outer 窗口与同目标训练窗口重叠；grouped 臂排除了 outer 目标身份。原 FIT/CAL 本身是按顺序切分，并非原有随机拆分。123 个起始时间不明的身份保留在登记中，不将其伪装成独立窗口。

**固定 80 epoch 的同路径完整覆盖。**

| 外层指标 | 原生 F16 | 随机窗口训练，seed 1/2/3 | 按行人分组训练，seed 1/2/3 |
|---|---:|---:|---:|
| 全部样例 / 1,035 | 625 | 642 / 649 / 632 | 642 / 622 / 632 |
| 变化组 / 253 | 110 | 113 / 115 / 113 | 107 / 108 / 109 |

整体覆盖率为原生 60.39%、随机 61.93%、分组 61.06%；变化组为 43.48%、44.93%、42.69%。分组相对随机的变化组覆盖差约 −2.24 个百分点，探索性区间跨零。与此同时，分组相对随机的变化组连续 ADE、FDE 和 J 更差，其轨迹簇 bootstrap 区间为正。随机臂变化组相对原生的 ADE 几乎不变、FDE 仅约改善 0.78%，相应 native-reference 区间跨零。因此，结果支持“目标身份相关窗口暴露值得继续检验”，但不支持“随机训练显著成功、行人分组后完全不能泛化”的强结论。

**反例与限制。** Grouped 的 inner 变化组覆盖 seed 2/3 高于原生；CAL 变化组 grouped 三个 seed 分别为 224、239、217，对比原生 243，seed 2 的整体 CAL 也优于 grouped 的随机臂平均。按场景拆分后，hotel 和 zara 的 outer 变化样本分别只有 17 和 25 个，不能作可靠的独立场景推断。Grouped 所有审核运动学组通过严格门槛；random 有 5 个 final 组的 speed p95 超过原生 1%。

**阶段判断。** 目标身份暴露可能影响连续误差与迁移，但覆盖差异不稳定；不能据此认定行人记忆、同轨迹泄漏或 shared head 干扰是唯一原因。

来源：[实验一结果分析](</C:/Users/linke/Documents/New project/GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment1_target_track/RESULT_ANALYSIS.md>)、[实验一完成审核](</C:/Users/linke/Documents/New project/GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment1_target_track/completion_review/REVIEW.json>)。

## 实验二：按录制时间块隔离

**设计。** 在同一组 566 个 outer 窗口（其中变化组 142 个）上，random 与 grouped 两臂各训练 16,104 个窗口；sequence×原三类 loss group 的样本数严格匹配，inner 2,134 个窗口共享。固定 50 个 source indices 的时间块作为隔离单位，完整 20-index 窗口用于重叠判断。Random 外层窗口均有同 sequence 时间暴露，564 个有已知同目标暴露；grouped 没有 outer 时间块重叠，但仍有 227 个 outer 窗口与远时间的同目标训练窗口相连。因此，时间块隔离不等于目标身份、邻近上下文或场景完全独立。

**固定 80 epoch 的同路径完整覆盖。**

| 外层指标 | 原生 F16 | 随机窗口训练，seed 1/2/3 | 时间块分组训练，seed 1/2/3 |
|---|---:|---:|---:|
| 全部样例 / 566 | 303 | 323 / 320 / 316 | 286 / 304 / 312 |
| 变化组 / 142 | 60 | 59 / 58 / 62 | 41 / 53 / 61 |

整体平均覆盖率为原生 53.53%、随机 56.48%、时间分组 53.12%。分组相对随机低 3.36 个百分点，但主时间块区间为 −7.84 至 +0.89 个百分点，跨零。变化组覆盖率为原生 42.25%、随机 42.02%、时间分组 36.38%；组间差约 −5.63 个百分点，主区间 −13.73 至 +5.56，也跨零。变化组只有 6 个时间块，区间很宽；grouped seed 3 的覆盖数为 61，仍高于原生 60。

与覆盖不同，整体连续误差更支持时间暴露依赖：grouped−random 的整体 ADE 差为 +0.009405 m（主探索性区间 +0.002544 至 +0.014728），FDE 为 +0.030328 m（+0.013495 至 +0.043065），J 为 +0.062911（+0.027064 至 +0.091967）。变化组平均方向也不利于 grouped，但主时间块区间均跨零。随机训练的变化组 FDE 相对原生改善约 7.45%，但 ADE 约改善 1.66%，完整覆盖几乎未变，说明连续误差改善没有稳定跨越完整轨迹门槛。

**CAL、分段质量与运动学。** 最终 CAL 变化组完整覆盖为原生 243，random 232/219/221，grouped 222/243/235；grouped seed 2 与原生持平，不能说三个 seed 都失败。按同一个 J 最近候选查看早、中、晚段，grouped 的变化组均值为 0.11372/0.26134/0.38235 m，random 为 0.11251/0.25443/0.34698 m，差距在后段较大；候选槽位可能变化，这不是固定槽位因果分析，也没有拼接不同候选。216 个 policy/group 运动学记录中 214 个通过；唯二失败是 grouped seed 1 final 变化组 outer 与 inner 的 speed p95，分别比原生高 1.182903% 和 1.289705%。严格 1% 门槛未放宽。

两臂训练 loss 都下降约 17.88%–20.31%，但 grouped 的 loss 降幅更大并未对应更好的 outer final 结果。两臂训练窗口不同，loss 分母也不是同一批样本的配对比较，不能仅凭这一点判定 loss 函数错误或 head 是根因。

来源：[实验二结果分析](</C:/Users/linke/Documents/New project/GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/RESULT_ANALYSIS.md>)、[实验二完成审核](</C:/Users/linke/Documents/New project/GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/completion_review/REVIEW.json>)。

## 两项实验合起来说明什么

1. **支持训练暴露与迁移有关的线索。** 按行人身份和按时间块隔离的结果，都显示连续误差或部分分组表现会随训练暴露变化。第二项整体 outer 的连续误差区间不含零，是当前较明确的探索信号。
2. **还没有稳定证明“随机窗口带来更多完整好路径”。** 两项主覆盖比较区间都跨零；第二项变化组 coverage seed 间差异大。连续 ADE/FDE 变好不等于同一个候选同时满足 ADE 与 FDE 门槛。
3. **没有锁定唯一机制。** 训练窗口身份与时间、有效样本多样性、输入条件分布、有限优化过程同时变化。按 sequence 和 loss group 匹配降低了部分差异，但未形成纯粹的 shared-head 干扰因果检验。
4. **暂不支持进入 scorer 优先分支。** 生成端的完整覆盖尚未稳定，旧 argmax 也不是对修正候选的新概率评分。现阶段 scorer 不能补出生成集合中不存在的好轨迹。
5. **也不支持宣称所有目标已对齐、只剩结构。** 第二项仍有两条严格 speed-p95 失败；CAL 终态未可靠优于原生。此前 loss/credit/motion 诊断与物理保护实验可作为背景证据，见[两阶段总报告](</C:/Users/linke/Documents/New project/TWO_STAGE_LOSS_TO_COVERAGE_REPORT_20261006/REPORT.md>)，但不与本系列两个不同 outer cohort 的窗口数直接合并。

因此，当前建议把“new head 的共享映射与迁移条件”作为优先研究问题，并继续保留 loss 反馈和运动学约束为可能的共同因素；还不到直接选择 MoE、改 scorer 或认定 loss 无效的阶段。

## 第三项：固定场景族暴露对照

预注册问题是：在固定 zara outer 窗口上，new head 完全不见 zara 的 grouped 训练，是否能与可见 zara 的 random 训练取得相近迁移表现。zara 是按元数据选出的最大非原始 CAL 场景族，**不是按实验效果挑选**。原 CAL 属于 university，所以该实验单独不能回答 university CAL 失败是否由 CAL 特有分布漂移造成。

当前 metadata scale 记录为：每臂训练 16,337 个窗口，共同统计集 12,183，outer 1,561（变化组 211），inner 1,773，outer 涉及 3 个录制序列。Random 训练含 4,154 个 zara 窗口，grouped 训练含 0 个。训练预算按全局三类 loss group 匹配，不按 sequence/scene 匹配，因为场景族暴露就是这项干预的一部分。

**截至 2026-10-06 13:32 香港时间的核验状态：** `experiment3_scene_family/runtime_state.json` 仍记录 `DESIGN_IMPLEMENTATION_PENDING`、`formal_started=false`、`worker_pid=null`；目录已有实现脚本，但尚无已登记的完整 ID/source/GPU/resume 预检结果、生产 worker 启动记录或科学结果。预定 11:45 的实施节点已过，但运行记录尚未推进。现有 metadata scale 检查明确只是规模检查，不等于完整预检通过。故本报告将第三项列为“尚无可报告的实验结果”。状态记录与代码文件存在情况不一致，下一次执行前需以实际预检和启动记录确认状态，不能仅依据脚本存在推断已经启动。

来源：[第三项预注册设计](</C:/Users/linke/Documents/New project/GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/DESIGN.md>)、[当前运行记录](</C:/Users/linke/Documents/New project/GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/runtime_state.json>)、[metadata scale 检查](</C:/Users/linke/Documents/New project/GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/METADATA_SCALE_CHECK.json>)。

## 可靠性、局限与报告性质

两项实验均完成六组 80 epoch 训练和保存结果的独立算术审核；报告所引完成审核重算了已保存的质量、聚簇、运动学及区间结果，并核验了模型权重、缓冲区和源文件完整性。该过程没有再次运行所有模型或梯度，也不是外部独立审稿。每项只有三个训练 seed；窗口存在轨迹内相关性；场景只有三个且分布不均；原 F16 已见过原数据；CAL 为单一 university 场景族且在其他研究中曾用于 checkpoint 选择。这里的区间应视为探索性证据，不能作为独立场景泛化或 SOTA 证明。

本报告只综合已保存结果、审核和预注册设计，没有启动第三项实验、读取旧 worker、重训模型或改变实验标准。研究技能用于组织证据、保留相反结果并区分观察与推断；文中结论由项目保存记录支持，不能视为独立外部复现。
