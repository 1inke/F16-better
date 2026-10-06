# H3 Objective Attribution

本实验承接 Diversity Stage Attribution 的 Case B 结论，检验更可能影响候选分配的 `oracle_coverage_weight`。

## 设计

固定 ETH、Seed 1、5% scene-tail validation split、H3 架构和全部训练参数，只形成：

| Condition | Diversity | Oracle coverage | 来源 |
|---|---:|---:|---|
| D1O1 | .05 | .20 | 复用 EXP-173 H3 baseline |
| D0O1 | 0 | .20 | 复用 H3-NoDiv |
| D1O0 | .05 | 0 | 新训练 |
| D0O0 | 0 | 0 | 新训练 |

只新增两次 H3 训练，不进入 V2，不访问 test。主要分析使用各自 `best_selected.pth`，同时强制分析共同 epoch 30 的 `final_model.pth`；机制结论必须在两种 checkpoint 视图中方向一致。

## 运行

```powershell
& "C:\Users\linke\Documents\New project\H3_OBJECTIVE_ATTRIBUTION_SERIES\run_experiment.ps1"
```

不应加入 `-Force`，除非明确需要覆盖已有结果。分步命令为：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\H3_OBJECTIVE_ATTRIBUTION_SERIES\run_h3_objective_attribution.py" --condition preflight
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\H3_OBJECTIVE_ATTRIBUTION_SERIES\run_h3_objective_attribution.py" --condition all
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\H3_OBJECTIVE_ATTRIBUTION_SERIES\analyze_h3_objective_attribution.py" --device cuda:0 --batch-size 128 --bootstrap-repetitions 10000 --bootstrap-seed 20260922
```

## 判定尺度

- Endpoint spread：`|Δ| >= 0.10 m`。
- Support@1m 或 ADE<1m：`|Δ| >= 0.5 candidate/sample`。
- minADE/minFDE：相对变化 `>= 2%`。
- 如果 coverage 增加 GT-local support 或改善 oracle quality，它是有用的候选分配器，而不是应删除的“发散源”。
- 如果只增加全局 spread、却不增加局部支持或 oracle quality，才把它判为无效发散驱动。

最终报告：`results\analysis\H3_OBJECTIVE_ATTRIBUTION_REPORT.md`。
