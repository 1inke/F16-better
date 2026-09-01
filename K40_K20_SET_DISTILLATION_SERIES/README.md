# EXP-172 Complementary Candidate Set Distillation

当前只执行零训练的 EXP-172A。它把同折、同 seed、同样本的 official V2 structured K20 与 EXP-157B matched-free K20 合并为 K40，然后使用完全不读取 GT 的确定性 geometry K-medoids 压缩为 K20。

距离固定为：

```text
candidate ADE + 0.25 × candidate FDE
```

运行：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\K40_K20_SET_DISTILLATION_SERIES\run_exp172a_set_distillability.py" --device cuda:0 --folds all --seed 1 --batch-size 64
```

只有 `summary.json` 中 `decision.passed=true` 才允许实现和训练 EXP-172B。K-medoids 不参与部署，也不是 selector 或 quota。
