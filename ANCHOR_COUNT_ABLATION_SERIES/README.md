# EXP-141：学习式锚点数量消融

固定F16编码器、Top-K=4稀疏交互、20候选、损失、EMA、数据划分和训练协议，只改变每条候选轨迹直接预测的锚点数量。所有方法均为纯候选生成，不使用细化器和CV外推。

| 实验 | 锚点位置 |
|---|---|
| EXP-141A | 第6、12步 |
| EXP-139C参考 | 第4、8、12步 |
| EXP-141B | 第3、6、9、12步 |
| EXP-141C | 第2、4、6、8、10、12步 |

## 运行命令

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\ANCHOR_COUNT_ABLATION_SERIES\run_anchor_count_batch.py" --device cuda:0 --epochs 30 --batch-size 64 --seeds 1 2 3
```

默认运行2/4/6锚点 × General/Challenge × 3 seeds，共18次新训练，并自动合并已有3锚点参考结果。脚本支持断点续跑；不要添加`--force`，除非明确希望覆盖已有结果。

## 输出

- `EXP_141A_TWO_ANCHOR_PROPOSAL/results/`
- `EXP_141B_FOUR_ANCHOR_PROPOSAL/results/`
- `EXP_141C_SIX_ANCHOR_PROPOSAL/results/`
- `all_runs.csv`
- `multiseed_summary.json`
- `RESULT_ANALYSIS.md`（正式实验完成后填写）

