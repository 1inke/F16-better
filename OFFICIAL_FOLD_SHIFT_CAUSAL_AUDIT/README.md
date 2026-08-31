# EXP-165 Official Fold Shift + Candidate Geometry Causal Audit

本实验不训练新模型。它复用 EXP-158 的 seed-1 K20/H3/V2 逐样本候选缓存，并只用对应 outer-fold 训练源数据定义速度四分位和密度三分位。

审计回答两个预注册问题：速度尺度偏移是否系统性损害 ETH 的 H3/V2 候选覆盖，以及低速高密度是否导致 Univ 的 H3 横向覆盖不足。只有自动门槛通过，才允许进入 EXP-166A 或 EXP-167。

运行入口为 `run_exp165_fold_shift_candidate_audit.py`。输出包括逐样本诊断、条件分桶、候选几何、折间偏移、自动判定与结果分析。
