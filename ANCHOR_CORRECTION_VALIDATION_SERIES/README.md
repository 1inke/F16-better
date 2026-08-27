# EXP-143：锚点校正机制验证

本轮不再堆叠结构，而是围绕 EXP-142A 的有效改进做三个单变量验证：

1. **EXP-143A Endpoint Only**：只允许第 12 步终点移动，上限 0.35；判断中间锚点校正是否必要。
2. **EXP-143B Wide Endpoint**：保留第 4/8 步 0.20/0.25 上限，将终点上限从 0.35 放宽到 0.50；判断 minFDE 是否受终点自由度限制。
3. **EXP-143C Quality Rescoring**：直接加载并冻结 EXP-142A 的完整轨迹生成网络，只训练新的候选质量评分头；判断 Top1 选择机制是否是主要短板。

三组均不使用 CV 外推。A/B 从匹配 split 和 seed 的 EXP-139C 检查点开始；C 从匹配的 EXP-142A 检查点开始。默认运行 General、Challenge 和 seeds 1/2/3，共 18 次训练。

## 运行命令

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\ANCHOR_CORRECTION_VALIDATION_SERIES\run_anchor_correction_validation_batch.py" --device cuda:0 --epochs 20 --batch-size 64 --seeds 1 2 3
```

脚本支持断点续跑：已有 `metrics.json` 的任务会自动跳过。除非需要覆盖已有结果，否则不要添加 `--force`。

结果统一汇总到 `all_runs.csv` 和 `multiseed_summary.json`。新增的校正幅度诊断会记录三个锚点的平均值、P95 和最大值，用于判断上限是否真正成为瓶颈。

注意：EXP-143C 冻结了轨迹，因此理论上不会改变 minADE/minFDE；它验证的是 Top1 ADE/FDE 能否通过更合理的候选排序改善。
