# EXP-169B Official Sparse-Social Counterfactual Audit

零训练复用 `COUNTERFACTUAL_SOCIAL_NECESSITY_AUDIT/csna_mask.py` 的精确 focal receive-row self-loop 干预，在相同 EXP-164 V2 seed-1 checkpoint 上比较 Full 与 Self-only。

LDHA 阈值仅由各 outer-fold checkpoint 实际训练 indices 计算，同时报告 outer-test 诊断与 source-inner proxy；只有 Case B3 才允许 EXP-169C。
