# PIF1：无场景物理意图条件流

## 研究问题

C00 相对 H3 改善了候选几何，但没有改善候选概率与 oracle 质量的对应关系。本实验不再调多样性、margin、固定原型、经验轨迹检索或外置 reranker，而是检验：冻结的 F16 运动/社会表征能否条件化一个连续的行人物理意图分布。

静态场景不会进入本实验。此前 SDD 的全局场景意图、逐候选空间 DINO、拓扑
proposal、静态残差和路径 affordance 实验均未通过各自的推进门槛。历史数值以以下
机读输出和由其生成的报告为准，不在本 README 中复制：

- `SCENE_CONDITIONED_INTENT_SERIES/EXP_183A_SCENE_CONDITIONED_INTENT_PROBE/`
- `SCENE_CONDITIONED_INTENT_SERIES/EXP_183A_R2_SPATIAL_DINO_MODE_SCENE/`
- `SCENE_TOPOLOGY_SERIES/EXP_178A_TOPOLOGY_REACHABILITY/results/`
- `SCENE_CONDITIONED_INTENT_SERIES/EXP_189_SCENE_RESIDUAL_PROBE/`
- `SCENE_CONDITIONED_INTENT_SERIES/EXP_190_PATH_AFFORDANCE_PROBE/`

这些结果表明，静态场景能限制“哪里可走”，但不能稳定判断“当前行人想走哪条可行路径”。PIF1 因此只使用观测历史和现有社会运动表征。

## 模型

PIF1 冻结 C00 的 F16 编码器与原有解码器参数，只训练一个 6 层、隐藏维 96 的轻量条件仿射流。六维物理意图为三段四步运动的：

1. 对数距离/当前速度比例；
2. 第一段相对航向；
3. 后两段转向增量。

训练集统计只使用 C00 的 train IDs。20 个候选由固定 Sobol 正态点经过条件流逆变换得到；候选分数就是同一条件流在该候选处的对数密度，不存在独立评分头。训练保留既定 oracle coverage=.20，但显式 diversity weight 固定为 0。额外使用权重 0.10 的目标意图 NLL。

## 两个停止点

### Stop 1：零训练候选边际价值审计

```powershell
& "C:\Users\linke\Documents\New project\PHYSICAL_INTENT_FLOW_SERIES\run_experiment.ps1" -Phase Audit -Device "cuda:0"
```

审计输出位于 `results/mode_utility_audit/`。它计算逐 mode leave-one-out oracle 贡献、按原生分数排名/候选半径的贡献，以及 K=1/3/5/10/15/20 预算曲线。这是验证集机制诊断，不能转化成使用未来标签的推理门控。

### Stop 2：ETH seed-1 验证集 feasibility pilot

复核 Stop 1 后运行：

```powershell
& "C:\Users\linke\Documents\New project\PHYSICAL_INTENT_FLOW_SERIES\run_experiment.ps1" -Phase Pilot -Device "cuda:0"
```

该命令依次生成 train-only 意图统计、训练一个 PIF1、对 best/final 两个 checkpoint 做相同 1,516 样本验证推理和 10,000 次配对 bootstrap。训练命令没有测试集路径。预计运行约 10–20 分钟，已有完整结果会自动跳过。

仅重新分析：

```powershell
& "C:\Users\linke\Documents\New project\PHYSICAL_INTENT_FLOW_SERIES\run_experiment.ps1" -Phase Analyze -Device "cuda:0"
```

## 预注册判定

best 和 final 必须方向一致：

- 生成门槛：`J=minADE+0.5*minFDE` 相对 C00 改善至少 2%；
- 安全门槛：Selected ADE 恶化不超过 0.5%，出现 0.1 m 内重复候选的样本比例不超过 10%；
- 排名信号：Selected ADE 改善至少 2%，或 Recall@1 提高至少 3 个百分点。

同时通过时才允许扩展 generator seeds 2/3；只通过生成门槛时记录为 `GENERATION_ONLY_REVIEW_SCORE_ALIGNMENT`，不自动扩展；否则停止 PIF1。整个 feasibility 阶段不读取测试集、不运行五折。
