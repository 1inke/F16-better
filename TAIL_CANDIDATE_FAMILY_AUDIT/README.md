# TCA：Tail-Candidate / Candidate-Family Audit

## 目标

在不训练新模型的情况下回答两个问题：

1. score rank 11–20 的候选是为大量样本提供小收益，还是为少数样本提供大收益？
2. tail-needed future 是否形成稳定、可用于重新设计候选基底的结构？

## 关键定义

- Tail 按模型 score 排序后的第11–20名定义，与mode ID无关。
- ADE和FDE使用各自的oracle winner，不能混成同一条轨迹。
- Material tail-needed：`ADE@10-ADE@20 >= 0.05`或`FDE@10-FDE@20 >= 0.10`。

## 比普通聚类更严格的地方

只证明GT future能聚类没有意义，因为所有轨迹都天然能按方向和速度聚类。本审计同时要求：

- tail subset聚类具有足够silhouette和多次初始化稳定性；
- cluster不能由极小簇支撑；
- 训练集tail medoid在锁定validation上替换部分原tail后，K=20 minADE或minFDE至少改善0.5%。

只有结构审计和validation medoid proxy同时通过，才允许编写HCB1训练模型。

## 运行命令

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\TAIL_CANDIDATE_FAMILY_AUDIT\run_tail_candidate_audit.py" --device cuda:0 --seed 1 --batch-size 128
```

预计约1～4分钟，无训练过程。

## 输出

- `audit/general_summary.json`
- `audit/challenge_summary.json`
- `audit/combined_summary.json`
- `RESULT_ANALYSIS.md`

若`hcb1_training_allowed=false`，应停止candidate-family训练路线，而不是绕过门槛直接实现HCB1。
