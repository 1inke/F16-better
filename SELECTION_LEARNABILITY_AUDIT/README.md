# EXP-150A：冻结 S0 候选集的选择可学习性审计

不训练、不改变 20 条候选。对每个 S0 seed 计算当前 score 与最终轨迹质量的相关性、Top-M recovery、winner hit 和 regret，判断是否值得进入冻结 listwise ranker。

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\SELECTION_LEARNABILITY_AUDIT\run_selection_learnability_audit.py" --device cuda:0 --seeds 1,2,3
```
