# EXP-158 Candidate Family Complementarity Audit

本系列在官方 ETH/UCY 8/12、2.5 Hz、五折、seed 1 协议下，对 matched K20 与 H3/V2/M2 候选族进行零训练审计。

`run_exp158_exp160_joint_audit.py` 只进行冻结推理，一次性收集四个候选族，随后同时生成：

- 单候选族与 K40 union 的 oracle minADE/minFDE/same-path FDE；
- winner family、候选端点/整轨迹重叠距离；
- K20 与 anchor family 获胜样本的运动分布；
- EXP-160 所需的 score-quality correlation、Top-N recovery、regret 与模式使用情况。

正式候选张量保存在 `per_sample/*.npz`，用于后续 EXP-159 固定 K=20 混合实验。K40 仅作诊断上界，不作为 benchmark 结果。

## 正式运行

```powershell
C:\coding_toobox\envs\astra2\python.exe -u "C:\Users\linke\Documents\New project\CANDIDATE_COMPLEMENTARITY_SERIES\run_exp158_exp160_joint_audit.py" --device cuda:0 --folds all --seed 1 --batch-size 64
```

