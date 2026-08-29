# Conditional Future Bank：先 Audit，后 CFB1

## 评估

该建议比纯 repulsion 更有机制上限，因为它给 non-winner modes 提供了“应该去哪里”的正向监督。但 processed ETH 文件没有 pedestrian ID，部分序列 frame 还是 `-1`，无法可靠执行同一行人/重叠窗口排除。

本实现采用更严格的 **train-only cross-scene retrieval**：

- Validation GT 永远不进入 bank；
- Query 只能检索训练折；
- Query 与 bank future 必须来自不同 scene；
- 因此同一行人、相邻窗口和共享 future frames 被结构性排除；
- 所有历史和 future 均转到“最后观测点为原点、最近三步 heading 为 x 轴”的局部坐标。

## 流程

一次命令自动完成：

1. 构建 General/Challenge 各自的 train-only future bank；
2. 执行 B0、Bank20、Selected3 与 B0∪Bank20 oracle audit；
3. 只有两个 split 均达到至少 3% union minADE 或 minFDE 上限时，才训练 CFB1；
4. Audit 不通过则自动停止，不浪费训练时间；
5. 通过后在 General/Challenge 各训练一次 CFB1，并自动诊断。

## CFB1

- GT 仍然只监督一个 winner mode；
- 从 20 个检索 future 中保留最相似历史对应 future，再用 farthest-point 选择两个互异 future；
- 三个 pseudo targets 各自只分配一个不同的 non-winner mode；
- GT winner 永远不接受 pseudo loss；
- pseudo loss 权重固定 0.1；
- 不增加推理分支、参数或测试时检索。

## 运行命令

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\CONDITIONAL_FUTURE_BANK_SERIES\run_cfb1_pipeline.py" --device cuda:0 --seed 1 --epochs 12
```

本机实测 Audit/缓存约 15 秒；考虑磁盘和显卡占用，保守预计 1–3 分钟。若通过，连同两个训练和诊断总计约 13–20 分钟。

## 已完成的 Audit 前置验证

严格跨 scene Audit 已通过：

- General `B0 ∪ Bank20`：minADE/minFDE 上限改善 5.89%/11.94%；
- Challenge：改善 5.95%/13.06%；
- 实际蒸馏的 `B0 ∪ Selected3` 在 General 改善 1.31%/2.75%，Challenge 改善 1.84%/3.23%；
- 因此 Bank20 ≥3%、Selected3 ≥1% 的双重门槛均通过，可以进入 CFB1 训练。

只执行可行性 Audit：

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\CONDITIONAL_FUTURE_BANK_SERIES\run_cfb1_pipeline.py" --device cuda:0 --seed 1 --audit-only
```

## 产物

- `audit/combined_summary.json`：严格 oracle audit 与是否放行训练；
- `cache/`：训练期 selected pseudo futures，不应作为原始数据上传；
- `seed1_summary.json`：若训练通过，B0/CFB1 严格比较；
- `diagnostic/`：ADE/FDE@K、social-needed、diversity 和 pseudo mode usage。
