# 按运动类型集中候选：生成侧 K20 对照实验

目的：检验上一轮全局 top3 集中带来的转向样本恶化，能否通过类型内集中、类型间保护得到缓解。

三个同架构、同 seed、同批次顺序、同预算的对照：
- mode_oracle：最佳候选损失。
- mode_global：额外全局 top3 质量损失。
- mode_family：仅对原始候选中与训练 GT 运动类型相同的最多 3 个候选施加集中损失。

所有对照都使用七类候选运动类型 one-hot 输入和相同跨类型距离下界约束。运动类型由已观测末步速度和原始候选末点启发式判定，不是学出的意图后验。GT 不输入生成器；仅在损失中给出监督类型。若原始候选没有 GT 类型，不添加额外支持损失，由全局 winner loss 处理。因此这一轮不实现新类型槽位分配，也不能宣称理解意图。

冻结 F16 V2 backbone 和全部筛选分数，仅重训练生成修正模块。K20 不变。ETH seed1，35 epochs，每个 arm 25,912 fit / 2,879 internal calibration。所有 6 个 best/final checkpoint 锁定后才加载 1,516 dev；禁止读取 held-out test。内部校准和重复检查的 dev 不构成独立泛化验证。

best 使用共同校准 minADE+.5minFDE+.1top3ADE，final 仅诊断。报告整体、稳定/过渡/变化、七类动作分组；变化组不得恶化是预定义关口。记录类型覆盖、类型保留、类型数、近 GT 支持、重复和运动学。筛选器未训练，原分数选中的误差仅诊断。

输出：results/REPORT.md、summary.json、metrics.csv、paired_bootstrap.csv、models/、带 sample_id 的预测 NPZ。stdout.log/stderr.log 为后台运行日志。旧实验保留。
