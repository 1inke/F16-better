# TNPA：Tail-Need Predictability Audit

## 实验问题

在不观察未来轨迹的前提下，当前B0能否利用历史运动、社会关系、场景信息或冻结encoder表示，提前判断一个样本是否需要score rank 11–20候选？

本实验不修改、也不训练主模型，只训练Logistic Regression、32维单隐层MLP和Ridge探针。

## 严格定义

- tail始终按预测分数排序后的第11–20名定义，不按固定mode ID定义。
- ADE tail label：`ADE@10-ADE@20 >= 0.05`。
- FDE tail label：`FDE@10-FDE@20 >= 0.10`。
- Union label：满足上述任意一个条件。
- 训练折只用于训练probe，所有指标在对应锁定validation上计算。
- 所有输入必须在推理时可见，禁止使用GT future构造特征。

## 特征消融

- P0：scene-only负控制。
- P1：individual motion。
- P2：social context。
- P3：individual + social。
- P4：冻结encoder的individual、social delta、fused表示。
- P5：individual + social + scene。
- P6：全部可见特征，用于估计可预测性上限。

## 关键输出

- AUROC、AUPRC及bootstrap 95% CI；
- positive/negative recall、Brier、ECE；
- top 10%/20%/35%风险预算下的tail-needed召回率；
- 连续ADE/FDE gain的Spearman、Pearson和MAE；
- P1→P3的social增量、P3→P5的scene增量；
- 是否允许进入CAT1的预注册门控。

CAT1门槛：General与Challenge均满足Union AUROC≥0.70、AUPRC lift≥1.25、连续gain Spearman≥0.15。

## 正式运行

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\TAIL_NEED_PREDICTABILITY_AUDIT\run_tail_need_predictability_audit.py" --device cuda:0 --seed 1 --batch-size 128 --bootstrap 200
```

预计用时约5～15分钟，取决于GPU推理和CPU上的MLP/Bootstrap速度。

快速验证可使用`--max-train 512 --max-val 256 --bootstrap 10`；带样本上限时默认写入`smoke_output`，不会覆盖正式审计结果。

## 输出

- `audit/general_summary.json`
- `audit/challenge_summary.json`
- `audit/combined_summary.json`
- `audit/classification_summary.csv`
- `audit/cache/*.npz`

`combined_summary.json`中的`cat1_training_allowed`为false时，不编写CAT1训练模型。
