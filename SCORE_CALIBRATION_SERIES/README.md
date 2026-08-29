# EXP-150B：冻结 S0 Score Calibration

每个 seed 仅在 SDD 验证集（raw scene id 1）拟合 logits 的 temperature 与 20 个 mode bias；候选、编码器、decoder 都冻结，测试集仅最终评估。

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\SCORE_CALIBRATION_SERIES\run_score_calibration.py" --device cuda:0 --seeds 1,2,3
```
