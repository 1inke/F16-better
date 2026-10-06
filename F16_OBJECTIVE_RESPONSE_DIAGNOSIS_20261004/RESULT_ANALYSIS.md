# 原目标兼容诊断：好候选有奖励，但反馈与学到之间仍有缺口

2026-10-04，FIT25912 / CAL2879 冻结保存数据简报。由同一AI助手分析与撰写；人类批准研究范围。不是新训练、外部审稿或独立泛化试验。

## 结论

目前不支持把问题概括成“原损失会把真正好的候选赶走”。在我们重点关注的“预兆与第一事件相容、粗响应覆盖但完整路径不达标”组，分别为576个FIT、77个CAL样例，只替换一条原生候选为先前已验证的GT辅助达标曲线，十九条候选保持原样，全部样例的原总损失降低。实际三种子模型却只覆盖FIT27/16/17、CAL3/3/1例。证据更支持先检查正确响应为什么没有被学出来，而不是直接加大曲线自由度或移除保护项。[项目数据，2026](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/RESULT.json>) <!--ref:RESULT--><!--anchor:section:all_summaries.FIT.specific_coarse_path_failure-->

“损失奖励这个结果”不等于“模型仅凭过去信息就能产生这个结果”。GT构造逐样例有未来答案，当前诊断没有实际模型前向、优化器或参数梯度。[既定设计](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/DESIGN.md>) <!--ref:DESIGN--><!--anchor:section:Objective%20and%20allocation-->

## 一、重点分类的完整候选覆盖

标准不变：同一完整候选 ADE≤0.2m 且 FDE≤0.4m；不是分别拼接两条候选的最优指标。表中GT辅助不是预测性能，三种子共享样例也不能合并成独立人数。

| 分类 | 样例数 | 原生有好候选 | 实际学习后三种子 | 单条GT辅助替换后 | 总损失降低的替换数 |
|---|---:|---:|---:|---:|---:|
| FIT第一事件线索相容 | 998 | 207 | 234 / 222 / 219 | 997 | 997 |
| CAL第一事件线索相容 | 137 | 31 | 32 / 34 / 30 | 137 | 137 |
| FIT粗响应对但路径错 | 576 | 0 | 27 / 16 / 17 | 576 | 576 |
| CAL粗响应对但路径错 | 77 | 0 | 3 / 3 / 1 | 77 | 77 |

表1来源：生产结果和独立保存汇总复核；后两行是前两行中的分类子集，不是额外独立样例。[汇总复核](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/completion_review/REVIEW.json>) <!--ref:completion_REVIEW--><!--anchor:section:groups.FIT%2FCAL.specific_first_compatible%2Fspecific_coarse_path_failure-->

CAL粗响应组的平均原总损失为0.701173582；单条替换为0.056932748，其中质量项0.056919872、其他正则项合计0.000012876、保护项为零。FIT对应为0.754526229→0.063668851。这里是固定人口分母下的损失贡献，不能当实际minibatch训练损失或ADE/FDE收益。[原始损失分项](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/RESULT.json>) <!--ref:RESULT--><!--anchor:section:all_summaries.CAL.specific_coarse_path_failure.methods-->

## 二、出现好结果与得到直接反馈是两件事

在原生候选状态下，质量项只对当前 ADE+0.5FDE 最小的候选提供直接坐标梯度。上述CAL77例中，64例（83.12%）的可达好候选位于另一个槽位，因此沿该单条替换方向的初始质量导数恰好为零；余13例为负。FIT对应456/576（79.17%）为零、120例为负。虽然完成这次有限替换后全部总损失更小，出发点不一定给目标槽位直接质量反馈。这是一个具体的反馈分配现象。[方向复核](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/completion_review/REVIEW.json>) <!--ref:completion_REVIEW--><!--anchor:section:groups.CAL.specific_coarse_path_failure.directional_counts-->

不能据此说83%的训练被阻断。真实共享头的参数会同时影响多个候选，未被检查的参数梯度和可表达映射可能改变结论；这里也没有新增事件监督或GT推理输入。[反方检查2](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/completion_review/ADVERSARIAL_REVIEW.md>) <!--ref:ADVERSARIAL_REVIEW--><!--anchor:section:Checkpoint%202%3A%20evidence%20synthesis-->

## 三、标量质量与完整达标并不完全相同，但并非当前主要占比

原生候选已有完整好路径时，训练质量赢家仍不达标：FIT79/14867（0.531%），CAL2/1490（0.134%）。实际学习后的对应占比FIT0.467%–0.534%，CAL0.067%–0.265%；CAL第一事件相容组均为零。这并不表示评分器没问题：本次“赢家”使用GT计算训练质量，并不是模型logit选出的部署轨迹。[计数与分母](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/completion_review/REVIEW.json>) <!--ref:completion_REVIEW--><!--anchor:section:groups.FIT%2FCAL.all.methods-->

因此，当前重点仍是零条好候选到至少一条，而不是复制相近轨迹凑数量或先改评分器。原有GT无关有效好候选去重统计继续有效；本诊断只记录原始好候选数量，没有重新聚类合成替换集合。[既定数量限制](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/DESIGN.md>) <!--ref:DESIGN--><!--anchor:section:Fixed%20scope%20%2F%20comparisons-->

## 四、保留的反证

实际三种子CAL整体质量项稍低于原生0.394452520，分别为0.394080882 / 0.393163332 / 0.393208103；加入原保护等项后的总损失却为0.399162817 / 0.396150203 / 0.395148866，均高于原生。实际路径改善仍会伴随受保护样例损伤，不能笼统说正则完全不阻碍任何变化，也不能据此删掉保护。[CAL分项](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/RESULT.json>) <!--ref:RESULT--><!--anchor:section:all_summaries.CAL.all.methods-->

旧自由增量、时间查询/速度基、集合监督尝试和FIT好CAL差的结果仍是反证；本诊断没有让之前的运动学及严格覆盖失败转为通过。人口坐标梯度不是实际miniBatch或头Jacobian，单条替换也没有证明全组运动学满足1%保护。三场景不均、CAL单university、窗口/轨迹相关、F16已见过原数据及旧CAL选择限制仍在。[反方检查2/3](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/completion_review/ADVERSARIAL_REVIEW.md>) <!--ref:ADVERSARIAL_REVIEW--><!--anchor:section:Checkpoint%202%3A%20evidence%20synthesis-->

## 五、完成校验与下一决定

worker与exit双SUCCESS，2026-10-04 23:34:18HK结束，实际181.15秒。450块、900输出SHA、原319源文件及23614旧文件完整核验通过；独立numpy分项/质量/运动算术最大误差1.776e-15，补充保存汇总复核误差零。运行峰值RSS0.740GiB/private1.125GiB，最低系统physical17.192GiB、commit14.397GiB，满足6GiB预算与余量。补充复核再查全旧文件/源SHA、全部数组拼接和所有组汇总，未重放模型或全部梯度向量。[生产审计](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/ARITHMETIC_AUDIT.json>) <!--ref:ARITHMETIC_AUDIT--><!--anchor:section:status%2Fblocks%2Findependent_numpy_max_error--> [资源记录](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/RESOURCE_USAGE.json>) <!--ref:RESOURCE_USAGE--><!--anchor:section:peak_private_GiB%2Fmin_physical_GiB%2Fmin_commit_GiB--> [完成确认](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/halfway_confirmation.json>) <!--ref:halfway_confirmation--><!--anchor:section:worker.finished_at-->

下一步建议讨论“实际响应头的参数梯度与固定FIT小样本学习检查”：确认信号是否传到漂移、幅度、时钟，以及原目标下能否记住这些固定训练样例，再区分学不会与学会但不泛化。这是新实验建议，不是本轮已执行内容，也不自动批准新损失/监督/解冻。先不追求更多自由度，不重复温度、系数或epoch扫参。完整好候选覆盖、GT无关有效数量、最后评分器的阶段顺序保持。

本轮到此完成并等待人类讨论；正式response study仍2/3，不新增正式训练计数，无major、新baseline、意图或SOTA结论。
