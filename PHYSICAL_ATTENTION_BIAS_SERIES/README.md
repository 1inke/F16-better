# PHYSICAL_ATTENTION_BIAS_SERIES

本系列验证逐时刻物理注意力偏置能否改善当前筛选模型的稀疏交互。

## 固定底座

所有实验从匹配 split/seed 的 `EXP-146B_ANCHORED_MODE_TEMPLATE/best_selected.pth` 开始，并固定：

- 20 个运动模式；
- 第 4、8、12 步三个学习式锚点；
- 锚点对齐的非线性运动模板与完整轨迹细化；
- F16 Top-K=4 稀疏社会交互；
- 无 CV 外推；
- 相同数据划分、增强、损失和检查点选择标准。

解码器在本轮完全冻结。B0/PB1/PB2/PB3 仅以相同学习率微调 F16 时序—社会交互编码器，从而把差异集中在注意力计算机制。

## 四组实验

- `baseline`：EXP-146B 原始 F16 交互继续微调，控制额外训练本身的影响；
- `pb1_add`：保留原 pair relation bias/value，叠加逐时刻物理偏置；
- `pb2_replace`：用物理偏置替换原 pair bias，保留 edge value；
- `pb3_bias_only`：删除 relation encoder/bias/value，仅保留 content attention 与物理偏置。

物理偏置分别编码 distance、closing speed、TTC 和 DCA。每个物理量使用 5 个固定 RBF，4 个 attention head 只新增 `4×4×5=80` 个参数。

## 运行

先执行：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\PHYSICAL_ATTENTION_BIAS_SERIES\smoke_check.py"
```

单种子筛选：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\PHYSICAL_ATTENTION_BIAS_SERIES\run_physical_bias_batch.py" --device cuda:0 --seeds 1 --variants baseline pb1_add pb2_replace pb3_bias_only
```

出现稳定正信号后运行三种子：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\PHYSICAL_ATTENTION_BIAS_SERIES\run_physical_bias_batch.py" --device cuda:0 --seeds 1 2 3 --variants baseline pb1_add pb2_replace pb3_bias_only
```

结果位于 `results_current_mode_anchor/`，汇总文件为 `multiseed_summary_current_mode_anchor.json` 和 `all_runs_current_mode_anchor.csv`。已有 `metrics.json` 的任务自动跳过；只有明确需要覆盖时才添加 `--force`。
