# CEQ1：条件经验未来分布量化

这个实验直接针对“候选被强行铺开，但高概率候选没有贴近真正可行终点”的问题。

它不再修改 diversity weight，也不重复已经完成的 motion-conditioned proposal residual 或固定尺度 mixture NLL。CEQ1 使用训练折建立条件经验未来分布：先用纯观测历史和稀疏邻居状态检索相似训练样本，再用按相似度加权的 full-trajectory Lloyd quantization 将概率质量压缩成 20 条候选。每个候选的分数就是对应 cluster mass，因此候选位置和选中概率来自同一个分布对象。

这是一轮 validation-only、零神经网络训练的可行性试验。它回答一个更靠前的问题：在不施加全局排斥、固定方向模板或硬配额时，训练数据中的条件未来分布能否同时改善 oracle candidate 与 top-1 selection。只有通过门槛，才值得把该分布蒸馏成新的神经候选头。

## 一键运行

```powershell
& "C:\Users\linke\Documents\New project\CONDITIONAL_ENDPOINT_QUANTIZATION_SERIES\run_experiment.ps1"
```

只做身份锁定和单元测试，不执行正式 validation：

```powershell
& "C:\Users\linke\Documents\New project\CONDITIONAL_ENDPOINT_QUANTIZATION_SERIES\run_experiment.ps1" -DryRun
```

先跑 64 个 validation 样本的烟雾检查：

```powershell
& "C:\Users\linke\Documents\New project\CONDITIONAL_ENDPOINT_QUANTIZATION_SERIES\run_experiment.ps1" -Smoke
```

正式运行通常只需要数分钟；它不会训练模型，也不会读取 ETH test 文件。结果写入 `results/ceq1_eth_seed1/`，主报告为 `CEQ1_REPORT.md`。

## 设计边界

- 检索特征只包含观测历史、速度变化和最近四个有效邻居的当前相对状态。
- 超参数只在 train 内部分出的 calibration 子集上选择；固定 validation 只评估一次。
- validation 查询会排除相邻 sample ID 的训练窗口，降低重叠轨迹造成的近邻泄漏风险。
- 20 个候选由条件概率质量量化产生；没有 pairwise repulsion、固定 10/10 配额或静态 endpoint prototype。
- 分数等于 cluster probability mass，不额外训练 post-hoc reranker。

## 决策规则

只有 oracle 质量、Selected ADE、Support@1m、高运动子集和重复候选检查共同通过，才进入 neural distillation。若只改善 oracle 而选择失败，则先停止蒸馏并分析 cluster-mass score；若 oracle 也不改善，就终止该候选族。
