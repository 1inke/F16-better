# CDA1：Conditional Distributional Anchors

## 实验目标

将当前 20 个 mode 的三锚点视为固定尺度高斯混合分布的 component mean，使用现有 mode logits 作为 mixture prior，在不引入外部 pseudo future、不增加参数的前提下增加条件锚点似然监督。

## 严格对照

本轮不是简单比较“旧 B0 checkpoint”和“额外训练后的 CDA1”，而是从同一个 CSRS 当前强 baseline checkpoint 同时续训：

- `cda0_matched_continuation`：原损失继续训练；
- `cda1_fixed_scale_anchor_mixture`：完全相同训练流程，加 0.1 权重的每维 anchor mixture NLL。

General/Challenge 各训练两次，共四次。CDA1 的主要因果对照是 matched continuation，不是旧 checkpoint。

## 固定尺度

- 只在训练折上运行 source B0；
- 按原 winner quality 选择每个样本的 winner mode；
- 由 winner 的 step 4/8/12 anchor residual 计算 robust scale；
- 用训练折将全局 multiplier 确定到平均 `N_eff` 最接近 2；
- scale 固定，不学习，不读取 validation GT。

## 运行命令

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\CONDITIONAL_DISTRIBUTIONAL_ANCHORS_SERIES\run_cda1_pipeline.py" --device cuda:0 --seed 1 --epochs 12
```

只校准固定尺度：

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\CONDITIONAL_DISTRIBUTIONAL_ANCHORS_SERIES\run_cda1_pipeline.py" --device cuda:0 --seed 1 --scale-only
```

查看命令、不训练：

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\CONDITIONAL_DISTRIBUTIONAL_ANCHORS_SERIES\run_cda1_pipeline.py" --dry-run
```

## 自动诊断

- minADE/minFDE、same-path、top-1；
- ADE/FDE@1/2/5/10/20；
- overall 与 social-needed；
- posterior top-1/top-3 mass、entropy、`N_eff`、NLL；
- score 与 posterior/anchor winner 一致率、ECE；
- winner usage、三锚点 coverage、候选多样性。

## 预计时间

- train-only scale calibration：本机实测约 20 秒；
- 四次 12 epoch 训练：参考相同 F16 训练记录，约 30～40 分钟；
- 诊断：约 1～3 分钟；
- 总计保守预计：约 32～45 分钟。

## 晋级标准

- CDA1 相对 matched continuation 的 minADE20 或 minFDE20 改善至少 0.5%，另一项不明显退化；
- 或 social-needed minADE/minFDE 改善至少 2%；
- General/Challenge 基本同向；
- `N_eff` 合理、diversity 不明显下降、K=10/20 不反转。
