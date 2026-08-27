# EXP-137：运动尺度泛化系列

固定 F16、Top-K=4 稀疏图、K20 解码器、损失、增强和 EMA，仅验证三个解码器之前的单变量：

1. `shape_scale`：运动形状—物理尺度双支路编码；
2. `difficulty_sampling`：高速度、高非线性、低密度训练样本温和重采样；
3. `scale_relation_bias`：不改变 Top-K 边，只给注意力 logits 增加尺度条件偏置。

默认运行 general/challenge 两种冻结开发划分和 seed 1/2/3，共 18 次新训练。原有三种子 Baseline 直接读入汇总，不重复训练，也不访问官方测试折。

结果输出：`multiseed_summary.json`、`all_runs.csv` 和每次训练自己的 `metrics.json`。

## 运行命令

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\FRONT_SCALE_GENERALIZATION_SERIES\run_scale_generalization_batch.py" --device cuda:0 --epochs 30 --batch-size 64
```

脚本支持断点续跑：已经存在 `metrics.json` 的训练会自动跳过。不要加入 `--force`，除非确实需要覆盖并重跑全部结果。

## 前置验证

- Python 语法检查与批量命令 dry-run 已通过；
- 两个模型插件均通过一批数据的前向、反向和梯度有限性检查；
- 困难样本重采样完成 1 轮端到端短训练；实测权重范围为 0.72–1.59；
- 形状—尺度编码参数量为 283,390；尺度条件关系偏置参数量为 284,798；采样方案保持 Baseline 的 284,642 参数。

## 时间估计

正式任务共 18 次新训练。根据上一系列单次约 12.5–14 分钟的实测耗时，预计总时间约 3 小时 50 分至 4 小时 15 分；建议预留约 4.5 小时。
