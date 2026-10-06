# Observation-Only Regime Change Detector

本实验只验证两个前提：未来运动状态变化是否能从 observation-only 信息预测，以及预测概率是否与冻结 C00 的 selection regret 相关。它不会训练 generator、修改候选、修改 scorer、加入 Gated Delta、访问 test 或自动进入 Stage C。

首轮包含两个 detector：

| 条件 | 输入 | 说明 |
|---|---|---|
| D0 | focal observed history kinematics | 只使用目标自身历史坐标、速度、加速度、速度和转向特征 |
| D1 | D0 + frozen F16 focal representation | 加入由冻结 C00/F16 从观测历史和邻居交互中提取的 social representation |

当前 baseline 没有地图或场景语义输入，因此不执行虚假的 D2。D-best 只按 C00 training IDs 内部划出的 detector-dev AUROC 选择；锁定的 1,516 条 validation 只做一次正式评估。标签严格复用既有 change score 与阈值：stable `<0.25`、transition `[0.25,0.5)`、change `>=0.5`。Binary BCE 排除 transition，但三组诊断保留全部样本。

完整运行：

```powershell
& "C:\Users\linke\Documents\New project\REGIME_CHANGE_DETECTOR_SERIES\run_experiment.ps1" -Phase All -Device "cuda:0"
```

状态查询：

```powershell
& "C:\Users\linke\Documents\New project\REGIME_CHANGE_DETECTOR_SERIES\run_experiment.ps1" -Phase Status
```

输出包括 detector 指标、三组概率、概率分箱与 baseline failure、逐样本诊断、八张规定图和最终报告。即使 H1/H2 通过，Stage C 也只会标记为待人工审阅，不会自动执行。
