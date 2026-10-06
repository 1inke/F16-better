# F16 H3→V2 Diversity 阶段归因实验

本目录把实验收敛为 ETH、Seed 1、固定 validation split 的阶段归因，不进行 margin 扫描，不访问 test，不自动扩展五折或多 seed。

## 当前趋势与本轮定位

现有 V2-only 消融中，`DIVERSITY_WEIGHT=0.05 → 0` 几乎不改变 endpoint spread。这只能说明 V2 阶段单独关闭太晚，不能区分 H3 早期塑形与多 mode 自发分工。本轮只新增三次训练：H3-NoDiv、C10、C11，并复用 C00/C01。

脚本对文档示例做了一个有意修正：新训练命令不传 `--test-data-path`/`--test-num-path`，因为实验协议后文要求人工复核前不得访问 test；该限制优先于示例命令。

## 两阶段运行

先只运行 H3，并在报告处停止：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\DIVERSITY_STAGE_ATTRIBUTION_SERIES\run_stage_attribution.py" --stage preflight
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\DIVERSITY_STAGE_ATTRIBUTION_SERIES\run_stage_attribution.py" --stage h3
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\DIVERSITY_STAGE_ATTRIBUTION_SERIES\analyze_stage_attribution.py" --phase h3 --device cuda:0 --batch-size 128
```

也可以用封装命令：

```powershell
& "C:\Users\linke\Documents\New project\DIVERSITY_STAGE_ATTRIBUTION_SERIES\run_stage_attribution.ps1" -Phase H3
```

检查 `results\h3_interim_analysis\H3_INTERIM_REPORT.md`，确认 checkpoint、split identity、候选几何和 activity 指标正常后，再运行 V2：

```powershell
& "C:\Users\linke\Documents\New project\DIVERSITY_STAGE_ATTRIBUTION_SERIES\run_stage_attribution.ps1" -Phase V2
```

展开后的 V2 命令为：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\DIVERSITY_STAGE_ATTRIBUTION_SERIES\run_stage_attribution.py" --stage c10
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\DIVERSITY_STAGE_ATTRIBUTION_SERIES\run_stage_attribution.py" --stage c11
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\DIVERSITY_STAGE_ATTRIBUTION_SERIES\analyze_stage_attribution.py" --phase final --device cuda:0 --batch-size 128 --bootstrap-repetitions 10000 --bootstrap-seed 20260922
```

## 安全与可复现约束

- `div005` 必须与官方 YAML 完全一致；`div000` 只能在 `MODEL.DECODER.DIVERSITY_WEIGHT` 一处不同。
- C10/C11 强制从本轮 H3-NoDiv checkpoint 初始化。
- 所有 checkpoint 必须使用与 EXP-173 H3 baseline 完全相同的 validation sample IDs 和顺序。
- 统一缓存保存的是 validation candidates、ADE/FDE 和 logits；不会缓存或评估 test。
- 默认遇到已完成的 `metrics.json` 会跳过，只有显式传 `--force` 才重跑。
- H3 与 V2 分成两个入口，避免未检查 H3 形成机制就自动继续。

## 输出

- `preflight_audit.json`：配置、输入路径和 SHA256 审计。
- `results/h3_interim_analysis/H3_INTERIM_REPORT.md`：H3-only 阶段检查。
- `results/stage_attribution_analysis/STAGE_ATTRIBUTION_REPORT.md`：最终主报告。
- `condition_summary.csv`：生成、选择、覆盖与 diversity activity 指标。
- `dynamic_bucket_summary.csv`：每个条件自己的 R0/R1/A/B/C。
- `fixed_c00_a_cohort.csv`：固定 C00 A-cohort。
- `paired_bootstrap_vs_c00.csv`：10,000 次 sample-wise paired bootstrap 95% CI。
- `validation_candidate_cache/*.npz`：相同 sample_id 顺序的 validation candidate cache。

这些结果仍属于同一 validation split 上的机制诊断，不能表述为独立泛化确认。
