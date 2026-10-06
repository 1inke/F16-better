# 保留K20的类型平衡训练复核

原20候选全部保留；普通训练vs类型平衡训练，两组结构、35epochs、LR3e-4、batch128、损失正则、样本顺序与内部校准选模规则完全沿用方向覆盖实验。只改变既定的winner损失类型权重。筛选器和F16主干冻结，仅生成修正模块训练。

seed1是已看过结果的发现性seed，校验原best/final和history后只读复用，不包装成独立验证。新增seed2/3四次配对训练，共三组配对seed。原主干仍是固定seed1，不能称三个完整模型主干seed。报告all_seeds和new_seeds_only，不平均轨迹，不构造ensemble。

训练fit25912/cal2879，归一化与类型权重fit-only；best按原共同校准key选，final诊断。全部12模型锁定后读取dev1516；无test、无折数扩大。dev被重复研究，仅检验训练seed稳定性，不是新的独立数据集泛化确认。

预定义关口优先新seed2/3：各seed变化组ADE/FDE都优于冻结native；新seed均值变化ADE至少改善1%；balanced-vs-uniform变化ADE/FDE交叉seed/样本配对bootstrap CI均<0；overall/stable/transition两项oracle损害<=1%；final变化ADE/FDE不恶化。3seed（仅2新增）bootstrap证据有限，不自动升级到正式baseline或接着跑test。

旧模型/日志保持不变；新组完成checkpoint+history后可核查跳过，部分已保存文件拒绝覆盖。尚未保存的中途arm重新从固定seed开始，非epoch级断点恢复。

输出results/REPORT.md、summary.json、per_seed_metrics.csv、aggregate_metrics.csv、hierarchical_bootstrap.csv、显式sample_id预测NPZ。新增生成参数训练不是预测后处理，但也不是整主干重新训练。
