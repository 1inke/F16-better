# IRA1：意图分布辅助表征实验

## 为什么做这个实验

PIF1 已经停止：它把 C00 的 20 条候选替换为条件流样本后，虽然 Recall@3 上升，但 `J`、minADE、minFDE、Selected ADE、候选间距和重复率同时恶化。这说明直接用密度样本替换候选会过度集中，却不能回答更关键的问题：**显式意图分布监督能否让现有多模态网络学到更好的运动表征？**

IRA1 不改 C00 的候选生成、不改打分头，也不把辅助流用于推理。它只比较：

- `IRA0 detached`：训练同一个意图流，但其 NLL 不进入共享上下文；
- `IRA1 joint`：同一 NLL 可以塑造共享上下文。

两臂从同一个 C00 checkpoint 开始，流结构、随机种子、参数量、训练轮数和数据顺序相同。因此主比较必须是 `IRA1 - IRA0`；历史 C00 只用于观察继续训练后的绝对位置。

这与已经失败的简单 motion auxiliary 不同：旧实验回归单一下一步/平均速度，容易把多模态未来压成均值；这里拟合的是三阶段速度比与转向增量组成的六维条件分布。

## 固定协议

- 数据：ETH，seed 1，锁定的 1,516 个 scene-tail validation IDs；
- 不读取 test；
- C00 候选数与推理计算路径保持不变；
- 辅助流：6 层、隐藏维 96、NLL 权重 0.05；
- 训练：10 epochs，学习率 `1e-4`，每 2 epochs 验证；
- 不做 NLL 权重、流深度、候选数或 diversity sweep。

## 一键运行

```powershell
& "C:\Users\linke\Documents\New project\INTENT_REPRESENTATION_AUXILIARY_SERIES\run_experiment.ps1" -Phase Pilot -Device "cuda:0"
```

Pilot 会依次执行快速契约测试、IRA0、IRA1 和配对分析。已有完整结果默认跳过；只有明确要覆盖时才添加 `-Force`。

仅重新分析：

```powershell
& "C:\Users\linke\Documents\New project\INTENT_REPRESENTATION_AUXILIARY_SERIES\run_experiment.ps1" -Phase Analyze -Device "cuda:0"
```

查看完成状态：

```powershell
& "C:\Users\linke\Documents\New project\INTENT_REPRESENTATION_AUXILIARY_SERIES\run_experiment.ps1" -Phase Status
```

先检查将执行的完整命令而不训练：

```powershell
& "C:\Users\linke\Documents\New project\INTENT_REPRESENTATION_AUXILIARY_SERIES\run_experiment.ps1" -Phase Pilot -Device "cuda:0" -DryRun
```

## 预注册停止门槛

Best 与 final 都必须同方向，并且 IRA1 相对 IRA0 同时满足：

- `J` 至少改善 2%；
- Selected ADE 至少改善 1%，或 Recall@1 至少提高 2 个百分点；
- minADE/minFDE 至少一个改善，另一个退化不超过 0.5%；
- same-path FDE 退化不超过 0.5%；
- endpoint spread 收缩不超过 5%；
- 重复候选样本率增加不超过 3 个百分点。

失败即停止，不补 seed、不看 test，也不围绕 NLL 权重继续搜索。

