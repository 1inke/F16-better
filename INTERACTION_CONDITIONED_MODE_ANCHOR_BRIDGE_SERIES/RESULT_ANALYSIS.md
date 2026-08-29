# Interaction-Conditioned Mode–Anchor Bridge（IC1）Seed 1 分析

## 最终判定

**IC1 的机制链路成功激活，但没有形成跨 General/Challenge 稳定的 oracle 与 coverage 提升。按预设标准，它不进入 seed 2/3，也不进入 IC2/IC3。**

本轮属于预设“情况2”：anchor/mode geometry 明显变化，但变化没有稳定朝向真实未来。不能通过扩大 rank、增加 bridge 容量或调 gate 强度解决。

这并不否定“交互信息直接参与候选生成”的总体思想；它否定的是当前 `raw gated-social token + 低秩 mode bridge + 原训练目标` 这一具体组合。若以后重启，应先改变 interaction representation 或监督方式，而不是继续微调桥接容量。

## 数据完整性

- General：1,516个严格配对样本；
- Challenge：3,032个严格配对样本；
- 四项训练与自动几何诊断全部完成；
- 离线推理完整复现10项训练结果指标，最大绝对误差约 `1.2e-7`；
- B0/IC1使用相同初始化源、数据划分、训练设置和冻结解码底座；
- IC1仅增加2,652个参数，总参数从355,048增至357,700（+0.747%）。

## 精度结果

负号表示误差下降。

| 划分 | minADE20 | minFDE20 | same-path FDE | Top-1 ADE | Top-1 FDE |
|---|---:|---:|---:|---:|---:|
| General | +0.017% | **-0.485%** | **-0.454%** | **-1.004%** | **-1.065%** |
| Challenge | +0.061% | -0.122% | **+0.451%** | +0.024% | -0.016% |

General 的 FDE、same-path 和 Top-1 有一致改善，但关键 minFDE20 仍略低于预设的0.5%门槛，minADE20基本不变。Challenge 没有复现：minFDE仅改善0.122%，same-path FDE反而退化0.451%。

## 配对不确定性

| 划分/子集 | 指标 | 变化 | 95% bootstrap区间 |
|---|---|---:|---:|
| General overall | minADE20 | +0.017% | [-0.747%, +0.795%] |
| General overall | minFDE20 | -0.485% | [-1.552%, +0.595%] |
| Challenge overall | minADE20 | +0.061% | [-0.255%, +0.375%] |
| Challenge overall | minFDE20 | -0.122% | [-0.555%, +0.294%] |

所有 oracle 区间都跨过零。General Top-1 ADE/FDE约1%的改善区间不跨零，但这不能替代 oracle candidate quality 证据。

## High-interaction子集

| 划分 | 样本数 | minADE20 | minFDE20 | same-path FDE | Top-1 ADE/FDE |
|---|---:|---:|---:|---:|---:|
| General | 416 | -0.788% | -0.609% | -0.031% | -1.423% / -1.404% |
| Challenge | 674 | +0.337% | -0.090% | +1.115% | +0.636% / +0.615% |

General 的高交互子集出现弱正信号，但只有0.6%～0.8%，没有达到2%～3%；Challenge 同一子集方向相反，ADE、same-path和Top-1均退化。因此不能解释为可泛化的交互收益。

## 机制证据

### Anchor确实发生变化

| 划分 | step 4 mean | step 8 mean | step 12 mean | mode-differential mean |
|---|---:|---:|---:|---:|
| General | 0.0341 | 0.0741 | 0.1180 | 0.0353 |
| Challenge | 0.0249 | 0.0457 | 0.0679 | 0.0176 |

Oracle winner发生变化的比例：

- General：23.42%；
- Challenge：7.55%。

所以IC1不是“没有改变候选”；它明显改变了General中的模式和锚点，Challenge中的改变相对较弱。

### 没有候选坍缩

| 划分 | endpoint diversity | anchor diversity | trajectory diversity |
|---|---:|---:|---:|
| General | -0.187% | -0.485% | -0.655% |
| Challenge | +0.218% | +0.279% | +0.359% |

多样性变化均小于0.7%，可以排除candidate collapse。

### GT coverage改善不一致

| 划分 | step 4 | step 8 | step 12 | endpoint |
|---|---:|---:|---:|---:|
| General | +0.511% | **-1.144%** | -0.223% | **-0.485%** |
| Challenge | +0.503% | +0.013% | +0.308% | -0.122% |

General的step 8与endpoint coverage改善，但step 4退化；Challenge前三个anchor coverage均没有改善。说明桥接更容易影响中后段/终点，却没有学到稳定的整条路径修正。

## Bridge激活情况

| 划分 | social norm mean | bridge norm mean | gate std mean | mode-delta std mean |
|---|---:|---:|---:|---:|
| General | 111.45 | 1.487 | 0.0748 | 0.2371 |
| Challenge | 76.88 | 0.792 | 0.0595 | 0.0668 |

Bridge和mode gate均正常激活，但其强度在两个划分间差异较大。General的mode-delta均值还明显高于中位数（中位数仅0.0302），表明作用集中在少量大修正样本，而不是稳定覆盖全体样本。

更关键的是，anchor shift、bridge norm、mode-differential shift与逐样本ADE/FDE收益的Spearman相关系数绝对值都不超过约0.024。**候选移动得更多，并不意味着预测得更准。**

## 同检查点Bridge开/关反事实

为了排除B0/IC1分别微调编码器造成的归因混淆，在同一个IC1最佳检查点上将bridge输出投影置零，保持编码器参数完全相同。

| 划分 | 直接minADE贡献 | 直接minFDE贡献 | 直接same-path贡献 | step 12直接shift |
|---|---:|---:|---:|---:|
| General | +0.008% | **-0.676%** | -0.149% | 0.0490 |
| Challenge | +0.000% | -0.040% | +0.027% | 0.0232 |

这说明General的终点收益确实主要来自Bridge本身，而不是普通编码器训练波动；但是该直接贡献在Challenge中几乎消失，也没有改善minADE。因此结论是“有效但高度划分依赖”，而不是“稳定的新候选生成机制”。

## 按预设标准判定

### A. 精度证据

- 至少一个主要oracle指标稳定改善不低于0.5%：**未通过**；
- Challenge收益应高于或至少复现General：**未通过**；
- high-interaction改善达到2%～3%：**未通过**。

### B. 机制证据

- anchor发生可测量变化：**通过**；
- mode-specific变化非零：**通过**；
- diversity未明显下降：**通过**；
- GT coverage具有跨划分改善趋势：**未通过**。

综合判定：机制传递成功，但预测价值不足，不晋级。

## 研究含义与下一步边界

当前实验回答了一个重要问题：

> F16社会交互信息能够通过2.65K参数的轻量桥，直接改变20个mode和三锚点；瓶颈已不再是信息传不进去，而是交互表示/训练目标不能稳定告诉各mode“应该向哪里移动”。

因此不应进行：

- seed 2/3；
- IC2 shared gate、IC3 full context消融；
- 增大rank或hidden dimension；
- 调整gate强度、学习率或简单增加层数。

如果以后继续该大方向，下一轮必须是概念层面的改变，例如使用尺度受控、意图相关的interaction token，或对mode/anchor增加明确的交互条件监督；不能只把当前bridge做大。
