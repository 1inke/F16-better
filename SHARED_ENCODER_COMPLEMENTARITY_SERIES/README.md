# Shared-Encoder Candidate Complementarity Series

EXP-158 的 K20 与 V2 来自分别训练的完整模型，因此 K40 union 同时混入了编码器集成与候选族互补。本系列在完全相同的 encoded tokens 上调用两种冻结解码器，隔离候选生成机制本身。

正式顺序：

1. EXP-158B shared-encoder complementarity control；
2. 只有 EXP-158B 通过，才进入 EXP-159 fixed-K20 mixture；
3. 固定最终 20 条候选后，再进入 EXP-161 score synchronization。

## 正式命令

```powershell
C:\coding_toobox\envs\astra2\python.exe -u "C:\Users\linke\Documents\New project\SHARED_ENCODER_COMPLEMENTARITY_SERIES\run_exp158b_shared_encoder_control.py" --device cuda:0 --folds all --hosts v2 k20 --seed 1 --batch-size 64
```

