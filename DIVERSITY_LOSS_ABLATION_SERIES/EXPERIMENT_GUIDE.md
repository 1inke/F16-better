# F16 Diversity Loss 受控消融实验

## 目的

验证当前 F16 的稀疏 GT 终点覆盖是否由 endpoint diversity loss 主动放大。

当前锁定基线：

- `DIVERSITY_WEIGHT = 0.05`
- `DIVERSITY_MARGIN = 0.4 m`
- EXP-173 V2 / `f16_temporal_social_cross`
- ETH / Seed 1
- `scene_tail` 5% validation，`validation_seed=20260830`
- 20 epochs，LR `2.5e-4`，EMA `0.995`
- oracle coverage auxiliary 保持 `0.20`

`ModeQueryLoss` 中 diversity 项真实形式为：

$$
L_{div}=\operatorname{mean}_{i<j}\max(0,m-\|e_i-e_j\|_2)
$$

总损失中乘以 `DIVERSITY_WEIGHT`。因此降低 weight 是最直接的单变量干预。

## P1：Weight Sweep（先运行）

固定 margin = 0.4，只改变：

- W0: 0.000
- W1: 0.010
- W2: 0.025
- W3: 0.050（原始基线）

所有条件从同一个 EXP-173 H3 checkpoint 初始化，decoder correction 初始化 seed 也保持一致。

### 一键运行

把整个 `DIVERSITY_LOSS_ABLATION_SERIES` 文件夹复制到：

```text
C:\Users\linke\Documents\New project\
```

然后在该仓库根目录 PowerShell：

```powershell
.\DIVERSITY_LOSS_ABLATION_SERIES\run_p1_weight_sweep.ps1
```

如你的 ASTRA 根目录不是：

```text
C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20
```

则直接运行 Python 并传 `--astra-root`。

## 核心指标

不要只看 minADE。每个条件同时比较：

1. `endpoint_spread`
2. `Support@0.5m / Support@1.0m`
3. `N(ADE<0.5m) / N(ADE<1.0m)`
4. `ΔADE_1,2`
5. `minADE20 / minFDE20`
6. Selected ADE / Selection Regret
7. A/B/C 桶比例

### 我们要验证的机制

若降低 diversity 后出现：

- endpoint spread 明显下降；
- `Support@1m` 上升；
- `ΔADE_1,2` 下降（oracle 不再孤立）；
- `N(ADE<0.5/1m)` 上升；
- 同时 minADE/minFDE 没有明显恶化；

则支持：**原 diversity loss 确实把原本可以聚集于正确未来附近的模式主动推散。**

若 spread 下降但 Support 不升、minADE 反而显著恶化，则说明 diversity 不是主要根因，甚至可能仍对 oracle coverage 有必要。

## P0 门禁：必须检查 baseline reproduction

分析脚本会自动把新的 `w=0.05,m=0.4` 与归档 EXP-173 Seed-1 验证结果比较：

- Selected ADE = `0.5388688445`
- minADE20 = `0.2195655257`
- minFDE20 = `0.3960312903`

如果 baseline 不能近似复现，先停止，不解释 sweep。

## P2：Margin Sweep（P1 后再决定）

如果 P1 表明降低 weight 有明显效果，但完全关闭 diversity 导致 mode collapse，则再运行：

```powershell
C:\coding_toobox\envs\astra2\python.exe -u .\DIVERSITY_LOSS_ABLATION_SERIES\run_diversity_ablation.py `
  --stage margin `
  --fixed-weight 0.050 `
  --margins "0.100,0.200,0.300,0.400" `
  --output-root .\DIVERSITY_LOSS_ABLATION_SERIES\results\margin_sweep
```

然后：

```powershell
C:\coding_toobox\envs\astra2\python.exe -u .\DIVERSITY_LOSS_ABLATION_SERIES\analyze_diversity_ablation.py `
  --sweep-root .\DIVERSITY_LOSS_ABLATION_SERIES\results\margin_sweep
```

## 多随机种子确认

P1 Seed-1 找到最有信息量的 1–2 个非基线设置后，再跑 seeds 1/2/3。不要一开始把所有组合三种子全部跑完。

示例：

```powershell
C:\coding_toobox\envs\astra2\python.exe -u .\DIVERSITY_LOSS_ABLATION_SERIES\run_diversity_ablation.py `
  --stage weight `
  --weights "0.000,0.025,0.050" `
  --fixed-margin 0.400 `
  --seeds "1,2,3" `
  --output-root .\DIVERSITY_LOSS_ABLATION_SERIES\results\weight_multiseed
```

## 测试集纪律

本包的 analyzer **只读取 validation**。在 validation 上选定最终方案并完成多 seed 确认之前，不用 test 做任何超参数判断。
