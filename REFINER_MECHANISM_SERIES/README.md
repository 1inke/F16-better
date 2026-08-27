# EXP-140：冻结三锚点生成器的细化机制实验

## 目的

本实验不改变锚点数量，专门隔离细化生成能力。每个 split/seed 加载对应的 EXP-139C `best_selected.pth`，冻结编码器、交互模块、三锚点生成器和评分器，只训练新细化器。

训练和推理均细化全部 20 条候选，因此不存在 Top-8 硬筛选、Oracle 强制补入或训练—推理选择不一致。终点仍严格保持不变。

如果本实验有效，说明 EXP-139D 的主要问题来自联合训练或 Top-8 机制；如果本实验仍无效，说明细化器结构、输入信息或损失设计本身需要调整。

## 运行命令

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\REFINER_MECHANISM_SERIES\run_frozen_all20_batch.py" --device cuda:0 --epochs 20 --batch-size 64 --seeds 1 2 3
```

脚本支持断点续跑；已有 `metrics.json` 的任务会自动跳过。除非明确需要覆盖结果，否则不要添加 `--force`。

## 下一轮：协议修复＋全历史条件细化

`run_next_mechanism_round.py` 会先用正确的验证种子20260825覆盖重跑EXP-140A的3个General任务，然后运行EXP-140B的General/Challenge共6组实验。EXP-140B仍冻结源模型和评分器、细化全部20条轨迹，但细化器同时读取完整8步历史编码和3个锚点。

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\REFINER_MECHANISM_SERIES\run_next_mechanism_round.py" --device cuda:0 --epochs 20 --batch-size 64 --seeds 1 2 3
```

## 实验输出

- 新结果：`EXP_140A_FROZEN_ALL20_REFINER/results/`
- 逐次汇总：`all_runs.csv`
- 多种子汇总：`multiseed_summary.json`
- 结果分析：`RESULT_ANALYSIS.md`（实验完成后填写）
