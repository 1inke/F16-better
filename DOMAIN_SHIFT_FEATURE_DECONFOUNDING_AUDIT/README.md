# DSA：Domain-Shift & Feature Deconfounding Audit

## 目标

利用TNPA已经生成的冻结B0缓存，在不训练主模型、不修改decoder的前提下回答：

1. General与motion-challenge的训练折到验证折分别发生了哪些条件偏移？
2. social对tail-needed的贡献能否在控制scene、speed、curvature和density后保留？
3. F16 encoder表示是否比原始条件更task-informative且更domain-invariant？

## 四部分实验

- D0–D5：5折OOF train-vs-validation domain classification；D0严格只含scene one-hot。
- 标量feature shift：均值、中位数、SMD、Wasserstein、P10/P90。
- Matched-strata probe：按scene、训练折速度四分位、训练折曲率四分位、density_r4分层共同支持匹配。
- Encoder robustness：比较raw observable与冻结encoder在tail任务和domain任务上的AUROC。

General作为负控制，Challenge作为主要决策集。

## Social晋级门槛

- Challenge匹配后M2相对M1的AUROC增量≥0.03；
- 该增量bootstrap 95% CI下界>0；
- General同方向。

只有通过才建议编写DRS1。否则根据shift来源决定DRM1、scene normalization或停止tail-specific modeling。

## 运行命令

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\DOMAIN_SHIFT_FEATURE_DECONFOUNDING_AUDIT\run_domain_shift_deconfounding_audit.py" --seed 1 --bootstrap 500
```

无需GPU推理，预计约1～5分钟。

## 输出

- `audit/general_summary.json`
- `audit/challenge_summary.json`
- `audit/combined_summary.json`
- `audit/feature_shift.csv`
