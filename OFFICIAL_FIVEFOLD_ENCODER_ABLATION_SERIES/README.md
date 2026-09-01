# Official Five-Fold Encoder Ablation

固定每个 official ETH/UCY fold 的最终 H3+V2 decoder 与 original score，只训练并比较 EXP-154 的四个精确 encoder variant：

```text
E0 f0_baseline
E1 f3_factorized
E2 f4_relation_tokens
E3 f16_temporal_social_cross
```

首轮运行五折 seed 1：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\OFFICIAL_FIVEFOLD_ENCODER_ABLATION_SERIES\run_official_encoder_ablation.py" --device cuda:0 --folds all --variants all --seeds 1 --epochs 30 --batch-size 64
```

四个 variant 使用相同 fold、inner validation、增强、损失、scheduler、checkpoint criterion、K=20 和冻结 V2 decoder。脚本自动输出 `all_runs.csv`、`per_fold_summary.csv` 与 `summary.json`。
