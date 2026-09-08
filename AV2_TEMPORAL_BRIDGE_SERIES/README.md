# AV2 Temporal Bridge Series — EXP-195A

这是针对当前 F16 + 20-mode anchor decoder 的低成本 AV2-Pedestrian 桥接实现。脚本不会读取地图、图像或车辆 context。

## 运行环境

- 数据准备：系统 Python（当前已检测到 pandas/pyarrow）。
- 训练/评估：`C:\coding_toobox\envs\astra2\python.exe`。
- 默认 GPU：RTX 5070 Laptop GPU 8 GB。

## 一键运行

```powershell
python AV2_TEMPORAL_BRIDGE_SERIES\run_exp195a.py --prepare
```

若 A3 在 8 GB 显存上训练溢出，应让 A1/A2/A3 全部统一改为 batch 32 后重跑，不要只改 A3：

```powershell
python AV2_TEMPORAL_BRIDGE_SERIES\run_exp195a.py --prepare --batch-size 32
```

分步运行：

```powershell
python AV2_TEMPORAL_BRIDGE_SERIES\prepare_av2_ped_bridge.py
C:\coding_toobox\envs\astra2\python.exe AV2_TEMPORAL_BRIDGE_SERIES\run_exp195a.py
C:\coding_toobox\envs\astra2\python.exe AV2_TEMPORAL_BRIDGE_SERIES\profile_exp195a.py
```

快速检查命令构造与模型形状：

```powershell
python AV2_TEMPORAL_BRIDGE_SERIES\run_exp195a.py --dry-run
C:\coding_toobox\envs\astra2\python.exe AV2_TEMPORAL_BRIDGE_SERIES\profile_exp195a.py --smoke
```

## 默认产物

- `manifests/`：固定 train/val target 清单。
- `prepared/`：A1/A2/A3 的 NumPy 数组与数据诊断。
- `results/a1|a2|a3/`：checkpoint、训练历史与原训练器指标。
- `EXP_195A_SHAPE_PRESERVING_BRIDGE/comparison.csv`：CV、ZS、A1–A3 主指标。
- `EXP_195A_SHAPE_PRESERVING_BRIDGE/timestep_error.csv`：1/2/3/4.8 s 分时刻指标；1 s 与 3 s 使用线性插值并明确标记。
- `EXP_195A_SHAPE_PRESERVING_BRIDGE/efficiency.csv`：参数量、FLOPs/MACs、显存、median/p95 latency 与 throughput。

## 预估耗时（本机）

| 环节 | 预计时间 |
|---|---:|
| 8 线程扫描并构建 8k/2k manifest 与三套数组 | 55–100 min |
| A0-CV + A0-ZS | 0.5–2 min |
| A1，4 epochs | 0.5–1.5 min |
| A2，4 epochs | 1–3 min |
| A3，4 epochs | 2–7 min |
| 统一评估 | 1–4 min |
| 正式效率 profiling | 3–10 min |
| **总计** | **约 63–126 min（约 1–2 h）** |

估算使用了本机实测：8 线程读取并物化 400 个场景、63 个合格 target 用时约 26.8 s；按正式 target 命中率外推，数据准备是主要耗时。机械硬盘、实时杀毒扫描大量 parquet 小文件或 GPU 功耗受限时，总时长可能达到 2–3 h。首次建议用单独的 `--output-root`，再加 `--max-scenarios 200` 和较小 cap 做 smoke，避免覆盖正式 manifest。
