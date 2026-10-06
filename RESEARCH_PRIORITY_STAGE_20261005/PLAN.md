# 当前小阶段研究优先级（2026-10-05）

人类要求结合实际数据和五项问题地图确定暂定方向，并执行第一项。此文件是阶段决策，不是新训练或阈值变更。最新授权撤销先前暂停，恢复现有首要任务；不授权新loss、scorer训练、更多解冻或数据范围扩大。

## 目标与证据边界

优先增加拥有至少一条完整高质量候选的窗口：同一candidate ADE≤.2m AND FDE≤.4m。其次评价GT无关全K20 complete-link(.10m meanpath/.20m endpoint)有效好簇数量与重复，不最大化数量或复制候选。最后解决选择。原全部major、运动学1%保护、样本范围和科学来源锁定不变。

风险/事件线索可读不是完整未来充分信息；GT辅助可达不是past-only可学；FIT损失下降不是同组CAL路径收益；继承F16槽位不是新head路径概率。当前数据是探索性研究，不独立泛化/心理意图/SOTA或唯一根因证明。

## 优先顺序

| 优先级 | 暂定问题 | 实际证据 | 下一步及可回答的问题 |
|---|---|---|---|
| 1 | 输入条件化与有限优化稳定性 | Actualstate覆盖733→744/742/749但CI跨0、seed3/运动学失败；相同active state改为观测速度偏差后覆盖725/730/728、CAL保留5/6/7早期微小修正；FIT-only标准化均值/方差显著变化 | 完成已预注册、已预检、当前挂起的第三标准化对照。只改变同一centered输入前30维FIT affine scalers；检验这一预条件是否带来可重复完整覆盖改善，而非证明centering丢失信息或decoder根因。 |
| 2 | 共享映射学习与迁移，兼查生成结构约束 | CAL粗响应覆盖但路径错77例全部GT辅助可达且替换后原loss降低，实际覆盖3/3/1；FIT学习改善未同步转化为CAL变化组表现 | 第三完成与三项总review后，先对齐FIT/CAL相同分组和相同覆盖/误差/运动学指标，区分训练内也学不好与训练内能学、CAL不能迁移；再精确设计可辨识参数映射检查，不盲加head自由度/epoch。设计尚非执行或权限扩张。 |
| 3 | primary质量反馈分配 | hardwinner对非赢家缺少直接质量坐标梯度；CAL77组64条GT辅助目标方向初始质量导数0，但共享参数与正则效应未因此消失 | 先参数层验证反馈能否到达候选响应分支；历史soft-set/energy/path-mixture未达major，不能重复泛化“均分梯度”试验。新loss需具体新授权。 |
| 4 | 评分器与修正后候选对齐 | 生成已有好candidate却被继承旧native argmax错过；新head从未重新预测槽位概率 | 覆盖形成稳定进展后单独评估选择差距；不把旧slot当新head保守意图。scorer训练需要具体新权限，不在本次执行范围。 |

泛化是以上各项的横向审核要求，不是排到最后才处理的第五个模块。优先级1是当前最小可辨识、已有配对对照、成本较低且可闭合证据链的下一动作，不是断言其解释最多失败样例。

## 第一项如何执行与解释

- 恢复GENERATION_DISCREPANCY_NORMALIZATION_SERIES_20261005同一个已核验挂起worker，不新建worker、不重训已完成seed、不改科学code/protocol/scalers。
- 恢复前检查PID/name/parent/commandline/create_time、挂起状态、402注册source SHA和实时RAM/commit。6GiB进程预算、系统physical+commit各≥6GiB余量；人类约17GB可用RAM为持久规划限制。
- 保留全部原初始化/权重/optimizer/RNG内存状态；不把OS原进程恢复说成重新执行2epoch测试。原双方真实2epoch FIT/CAL-subset EXACT预检仍为此前证据。
- 按剩余训练/calIO/18policy/eval/聚簇/全组CI/402source和29191old SHA估计剩余耗时；在预计剩余时间一半一次检查，修正完成节点后退出静默。人类触发的暂停检查不计自动半程检查。
- 完整双SUCCESS及全部原完成审计后第三记3/3，先fullthree candidate-state review，再讨论下一大方向。标准化若只改善FIT或均值、未改善同路径覆盖或违反运动学，仍不升级baseline；即使改善也不等纯因果或运动语义理解。若失败，终止局部标准化变体，回到优先级2的映射/迁移分解。
- 源锁/ID/资源/科学不可辨识/权限等实质阻碍立即STOP等待人类，不降低阈值、删难例或绕范围。

## 依据

- [第一项结果](../GENERATION_CANDIDATE_STATE_SERIES_20261005/RESULT_ANALYSIS.md)
- [第二项结果](../GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md)
- [第三项预注册](../GENERATION_DISCREPANCY_NORMALIZATION_SERIES_20261005/DESIGN.md)
- [目标反馈诊断](../F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/RESULT_ANALYSIS.md)
- [此前响应研究总review](../GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/RESPONSE_STUDY_REVIEW.md)

分析与决策由同一助手完成，非外部独立评审。当前批准执行的是第1项，不提前批准后面模块变更。
