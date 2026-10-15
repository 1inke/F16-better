# 新结果审查与下一步（2026-10-07）

范围：核对新增 S1L3 transfer、selection、calibrated Top-3 报告及相关实现，不宣称已经完整复查所有新增实验。采用 academic-research-suite 的证据与反方审查；同助手内部审查，不是外部独立评审。

## 已发现的结论风险

1. S1L3 相对 canonical static K24 的变化组覆盖平均增幅约1.49个百分点，8/9协议×seed为正、1中性。这是探索性小幅信号，复用窗口/协议和仅3seed不是9个独立数据集，不满足原major升级标准。
2. 相同输出3条：M2 native Top-3整体覆盖42.062%、变化20.671%，高于推荐C1的34.953%/19.697%，也高于C2的37.119%/20.446%。候选来源多样性不能抵消完整覆盖劣势；仍须检查新候选是否救回原生缺失窗口，不能据此宣布 learned candidate 完全无用。
3. M1训练目标 softmax(-2J+3good) 是人为质量效用，不是真实未来发生概率。报告ECE .24395是约24.4个百分点；soft-target NLL/Brier与binary-good ECE评价对象不同，不能混称概率已校准。
4. M1给R2 scorer使用前5000训练样例统计，而R2训练使用完整训练统计；M3又用outer评估集重算scorer/calibrator输入标准化及entropy阈值。后者是GT-free transductive处理与训练推理契约偏差，不等于直接未来标签泄漏，但使原结论不可靠。
5. M1只使用ordered first5000训练样例，需审核组成偏差。拒绝预测后成功率必须同时报告所有窗口、保留窗口及同保留窗口native1/native3对照，不能据此声称机器人安全。
6. 新候选的旧scorer先验是median(logits)-std构造，未实际调用原scorer对修正路径前向；因此“旧scorer对新路径结构性偏置”不能从该设置唯一得出。past-only不可预测、不可避免Pareto或信息天花板同样超过现有证据。

## 当前任务

已通过实际frozen GPU与保存重算烟雾预检，唯一独立Windows零trigger任务13:47:52HK启动。9个现有协议×seed，全模型参数/buffer冻结，无optimizer、新训练、新loss或新数据；不增加正式计数。

A复现M3评估集统计；B恢复M1训练时first5000输入处理和训练阈值；C恢复R2完整训练统计并将现有M1作为输入压力测试。C不能称已修复且校准的新模型。

同预算native3、C1/C2/C3与full24/native20；单candidate ADE≤.2 AND FDE≤.4、有效好簇及五运动学指标；质量效用对actual uniform及binary selected-good校准；拒绝预测同保留窗口对照。保存每窗口数组、统计来源IDs、权重SHA及frozen exact审查。

预计2–5分钟，planning4；唯一半程13:49:53，初预计完成13:51:53。完成后依据同预算与输入契约结果选单因素后续，不预先改loss/head/scorer。源锁只覆盖相关sources/checkpoints/reports/inputs，不冒称全部历史文件全SHA审核。

旧grouped第三项saved worker/exit已双SUCCESS于2026-10-06 18:38:25HK；不再查询旧PID/重训。其mandatory fullthree review尚未在本轮核实，不能沿用旧.memory的RUNNING或声称全部review完成。
