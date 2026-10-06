# 固定候选预算的方向覆盖实验

针对连续两轮集中损失改善稳定样本却损害变化样本，本轮取消 top3 集中损失，测试候选构造和训练分布两个因素。仍是探索性 ETH seed1，禁止 test 与自动扩大种子/折数。

2×2 同架构训练：native_uniform、native_balanced、replacement_uniform、replacement_balanced。Native 使用原20候选；replacement 按原始分数保留前12，替换8个低分槽为观测确定的停止、减速、加速、轻/强左/右转、回转候选。附加 native、replacement_untrained 非训练参照。所有组 K20，不将K28追加的必然 oracle优势当改进。

所有训练组使用上一轮相同的候选类型条件修正网络，重训练生成模块，原 backbone/筛选器冻结。直接构造的是显式运动锚点，再学习锚点修正；并非从原始社会上下文端到端重新训练整个生成器，也不是证明模型理解意图。原始高分候选选择只用历史和 native logits，不使用 GT。新槽位无训练分数，因此不报告选择指标。

普通组使用 winner ADE+.5FDE；平衡组按 fit 中七种 GT 运动类型的频次给 winner loss 加权：sqrt(N/(7*count))，截断[.5,3]后按 fit 均值归一化。不是未来GT输入或推理门控。都保留相同残差、平滑、跨类型距离约束。匹配 seed、35epochs、batch128、LR3e-4 和样本顺序。

fit25912/cal2879；归一化和类型频次只由fit计算。所有组统一用内部校准 minADE+.5minFDE+.1top3ADE 选 best。8模型锁定后加载dev1516。Final为诊断，最终2×2主效应/交互优先匹配final以避免best epoch差异。反复使用dev且backbone见过内部校准，非独立泛化证据。

主处理预指定 replacement_balanced_best，目标变化组 minADE>=2%改善，整体oracle不恶化、变化FDE不恶化、稳定与过渡ADE不超过1%损害、重复/运动学阈值与final方向一致性。新增锚点有不同加速度形状，运动学关口采用绝对阈值超限比例，不与上一轮相对平均加速度关口混为一谈；平均值仍完整报告。K固定替换可能丢失有用低分候选，必须保留未训练替换对照和损失平衡对照。

结果在results/REPORT.md、summary.json、metrics.csv、paired_bootstrap.csv、factorial_final.csv及显式sample_id预测NPZ。不自动接着训练下一轮。
