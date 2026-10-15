# 冻结复算结果及来源限制

9jobs实际13:47:52→13:48:25HK，双SUCCESS，无optimizer。独立NumPy/SciPy保存数据审核完整聚簇/运动学/标量maxerror8.881784e-16；权重buffer固定，相关source/input/checkpoint SHA初末不变（不是全部历史文件）。

有效部分：同K候选生成和组合指标。macro按3protocol×3seed等权，非9个独立数据集：native3整体42.062%、变化20.671%；C1 34.953%/19.697%；C2 37.119%/20.446%；C3 36.634%/19.467%。Full24 vsnative20整体60.181→62.531%、变化41.531→44.438%，说明追加候选存在小幅补充，不是同预算优势。C1每job平均救回native3缺失窗口21.44却丢102，不能用多样性抵消净损失；不同协议窗口数不等，不把平均count当合并比例。

来源错误：本助手parent helper遗漏分类器softmax，影响calibrator p_change输入。Parent calibration/interface数字不作为忠实原M1/M3重放结论。静态class排序在softmax前后相同，候选部分有效。全部父代码、结果及失败来源保留，不修改原报告掩盖错误。

恢复子目录 calibration_contract_recovery仅恢复原CUDAsoftmax，预检后新sole任务执行9jobs双SUCCESS；其RESULT及completion_review/AGGREGATE为正确概率接口复算来源。源读回审查不等于所有模型第二次前向或外部独立审稿。

反方2（同助手）：native3更高覆盖不证明新候选无用；full24有新增救回，变化组C2连续误差也低于native3。固定预算组合可能才是主要问题。

反方3（同助手）：这三个协议外层被反复使用，模型选择及派生研究不是独立confirmatory；native3与native20运动学比较也改变候选池组成，native3较高speedp95不意味着原路径本身被修改。Original all-motion gate仍须保留，不能将coverage结果直接升级baseline。
