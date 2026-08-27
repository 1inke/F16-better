# EXP-144：结构级候选生成

本轮以 EXP-143B 为冻结来源，不再扫描锚点校正上限。所有方案保持 20 个候选、稀疏交互、运动学输入和无 CV 外推。

- **EXP-144A Set Coordinated Generation**：让 20 个候选通过轻量集合注意力相互感知，再联合生成第二阶段锚点残差，减少候选重复和局部聚集。
- **EXP-144B Dynamic Progress Generation**：保留锚点位置，但学习三个单调运动阶段的持续时间，代替对所有行人固定使用第 4/8/12 步。
- **EXP-144C Set + Progress**：联合验证候选集合协同与动态运动进度是否互补。

新增模块均为零初始化，训练开始时严格复现对应 split/seed 的 EXP-143B 输出；编码器、交互模块、EXP-143B 候选生成器、校正器和评分器全部冻结。因此结果差异可以归因到新增结构。

## 运行命令

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\CANDIDATE_GENERATION_SERIES\run_candidate_generation_batch.py" --device cuda:0 --epochs 25 --batch-size 64 --seeds 1 2 3
```

默认执行 3 个方案 × General/Challenge × 3 seeds，共 18 次训练。支持断点续跑，已有结果会自动跳过；不要添加 `--force`，除非明确需要覆盖结果。

输出包括 `all_runs.csv`、`multiseed_summary.json`，以及动态进度锚点的均值/P05/P95诊断。
