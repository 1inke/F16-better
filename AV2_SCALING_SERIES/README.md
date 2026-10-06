# AV2 Scaling Series — EXP-196

本目录实现 EXP-196 的公平时间分辨率、长历史、密集邻居和联合压力实验。所有结果均为 **NON-OFFICIAL AV2 DIAGNOSTIC**，不是 AV2 leaderboard 指标。

## 已固化的实验约束

- A1C/A2/A3 都从同一个 EXP-195 A1 checkpoint 开始，再训练相同的 4 epochs。
- Long History 固定 10 Hz、相同 target、相同 future，只改变 `T=16/32/40/50`。
- Dense Scaling 固定 `T=32`，使用同一 target pool，比对 `N=5/9/16/24` 与 `top_k=4/8`。
- Joint Stress 使用完整 50-step 历史且当前 6 m 内至少有 5 个可用邻居的独立稀有样本池。
- 时间位置编码和图历史权重按物理秒对齐；12-step decoder 与 2.5 Hz future 不变。
- 效率测试采用随机交错顺序和 3 个 cycle，降低顺序 profiling 偏差。

## 先做快速验证

```powershell
C:\coding_toobox\envs\astra2\python.exe AV2_SCALING_SERIES\validate_exp196.py --device cuda:0
python -S AV2_SCALING_SERIES\run_exp196.py --dry-run --stages fair --seeds 1 2 3
```

## 推荐执行顺序

### 1. EXP-196A/B：公平对照与 3 seeds

复用 EXP-195 已准备的 A1/A2/A3 数组，不需要重新扫描 AV2：

```powershell
python AV2_SCALING_SERIES\run_exp196.py --stages fair --seeds 1 2 3
```

### 2. EXP-196C：长历史

```powershell
python AV2_SCALING_SERIES\run_exp196.py --prepare --stages long --seeds 1
```

如果先跑最低成本的关键比较：

```powershell
python AV2_SCALING_SERIES\run_exp196.py --prepare --variants l2 l4 --seeds 1
```

### 3. EXP-196D：密集邻居

```powershell
python AV2_SCALING_SERIES\run_exp196.py --prepare --stages dense --seeds 1
```

最低成本关键比较：

```powershell
python AV2_SCALING_SERIES\run_exp196.py --prepare --variants d2 d4 --seeds 1
```

### 4. EXP-196E：联合压力

```powershell
python AV2_SCALING_SERIES\run_exp196.py --prepare --stages joint --seeds 1
```

### 5. 正式效率测试

训练完成后运行：

```powershell
C:\coding_toobox\envs\astra2\python.exe AV2_SCALING_SERIES\profile_exp196.py --variants l1 l2 l3 l4 d1 d2 d3 d4 d5 d6 j1 j2 --seed 1
```

快速形状检查使用 `--smoke`；它不能代替正式效率测试。

## 一次跑关键点

先完成 fair 的 3 seeds，再运行：

```powershell
python AV2_SCALING_SERIES\run_exp196.py --prepare --low-cost --seeds 1 --profile
```

已有的 seed-1 fair 结果会自动跳过。不要把 `--seeds 1 2 3` 与 `--low-cost` 同时使用，否则长历史和 dense 也会被重复三次。

## 默认安全 batch

| 阶段 | batch |
|---|---:|
| fair | 64 |
| long | 32 |
| dense | 16 |
| joint | 8 |

同一阶段内 batch 固定，保证组内公平。若显存不足，可使用 `--batch-size` 覆盖，但必须让该阶段所有配置使用同一个值。

## 数据筛选现实约束

指导书建议每个 dense cohort 至少 500 个目标，但本机 smoke 扫描约 5,000 个训练场景时，完整 5 秒历史条件下只发现 2 个 `>=5` 邻居目标。脚本因此采用：

- Dense 主实验：32-step 历史完整，默认每个合并 cohort 至少 20 个，最终报告真实样本数；
- S3 不足时自动合并为 `S2+ = >=5`；
- Joint：单独扫描完整 50-step 且 `>=5` 邻居的稀有样本，默认 train/val 为 200/20。

如果样本不足，脚本会明确失败并报告数量，不会用重复样本填满 cohort。

## 主要输出

- `EXP_196A_CONTINUED_A1/comparison.csv`
- `EXP_196B_MULTI_SEED_TEMPORAL/all_runs.csv`
- `EXP_196B_MULTI_SEED_TEMPORAL/multiseed_summary.json`
- `EXP_196C_LONG_HISTORY/temporal_scaling.csv`
- `EXP_196C_LONG_HISTORY/cohort_metrics.csv`
- `EXP_196D_DENSE_NEIGHBOR/neighbor_scaling.csv`
- `EXP_196D_DENSE_NEIGHBOR/cohort_metrics.csv`
- `EXP_196E_JOINT_STRESS/joint_scaling.csv`
- `EXP_196_SCALING_PROFILE/efficiency_scaling.csv`
- `bootstrap_comparisons.csv`
- `exp196_summary.json`

## 预计运行时间（RTX 5070 Laptop 8 GB）

| 环节 | 预计时间 |
|---|---:|
| 196A/B：A1C/A2/A3 × 3 seeds（复用数据） | 5–10 min |
| Long 数据准备 | 15–30 min |
| Long 4 组训练与评估 | 4–10 min |
| Dense 数据准备 | 20–40 min |
| Dense 6 组训练与评估 | 15–35 min |
| Joint 稀有样本扫描、训练与评估 | 30–70 min |
| 正式交错效率测试 | 10–25 min |
| **关键低成本路线** | **约 60–120 min** |
| **完整 seed-1 scaling + fair 3 seeds** | **约 1.5–3 h** |

时间估计依据：EXP-195 的 8k/2k 正式训练单组为 18–31 秒；本轮 smoke 扫描 384 个完整历史场景约 6 秒，dense 30/30 小样本约 12 秒，但 joint 稀有样本的命中率约为每 500–1,000 个场景 1 个。冷盘、杀毒实时扫描或 GPU 功耗限制时，完整流程可能达到 3–5 小时。
