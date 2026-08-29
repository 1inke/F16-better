# Response-Aware Interaction Series

## 核心问题

现有稀疏运动学图判断“谁可能产生交互”，RB1进一步判断“目标是否已经对某个邻居表现出减速、转向或侧向避让响应”。

## 严格对照

- `baseline`：当前EXP-146B模式模板三锚点模型；
- `rb1_response_aware`：只新增逐时刻Response-Aware Bias；
- 固定F16原relation encoder/bias/value、Top-K=4、20模式、4/8/12三锚点、非线性模式模板与轨迹细化；
- 对应split/seed从同一EXP-146B检查点开始；
- 解码器完全冻结，`cv_reference=False`；
- 两组以相同`2.5e-4`学习率微调F16交互编码器。

## RB1

四维响应向量：

1. 目标速度变化沿目标—邻居方向的纵向投影；
2. 沿垂直方向的侧向投影；
3. 目标速度幅值变化；
4. 目标航向变化。

它先经过固定归一化和零初始化的`Linear(4→4 heads)`，再乘以距离—接近机会门。总新增参数为20。

## 运行顺序

烟雾测试已经由脚本作者验证，也可以手动复核：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\RESPONSE_AWARE_INTERACTION_SERIES\smoke_check.py"
```

第一轮只运行seed 1：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\RESPONSE_AWARE_INTERACTION_SERIES\run_response_aware_batch.py" --device cuda:0 --seeds 1 --variants baseline rb1_response_aware
```

共运行General/Challenge × B0/RB1四个任务。只有seed 1达到预设晋级条件后，才补跑seed 2/3和交互子集评估。

## 停止条件

如果General和Challenge的minADE/minFDE均不改善，或一个划分轻微改善而另一个退化超过1%，直接停止，不搜索隐藏层、bias上限、响应特征组合或学习率。

## Seed 1 离线交互收益诊断

RB1 的总体结果属于边界信号，因此在补跑 seed 2/3 前，先运行：

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\RESPONSE_AWARE_INTERACTION_SERIES\run_interaction_gain_diagnostic.py" --device cuda:0 --batch-size 128 --splits all
```

该脚本不训练模型，只读取 seed 1 的 B0/RB1 检查点，并执行严格逐样本配对分析。输出位于 `interaction_gain_diagnostic/`，包括：

- `paired_samples.csv`：逐样本 B0/RB1 指标及差值；
- response、pair-response、lateral-response、gated-response、potential 的 decile 表；
- nearest distance、closing speed、TTC、neighbor count 分桶；
- 使用训练集 Q75 固定阈值的 Potential × Response 二维表；
- cumulative Top-response 表和 `interaction_gain_curves.png`；
- 自动总体指标复现检查与 `combined_summary.json`。

只有 high-response 或 high-potential + high-response 子集获得约 2%–3% 以上的稳定 oracle 改善，并呈现随响应强度增强的趋势，才补跑 seed 2/3。
