# EXP-165～EXP-168C 实验链说明

## 研究问题

这一阶段围绕两个问题展开：

1. ETH/UCY 五折差异是否可以用速度尺度或密度条件解释，并据此修复 H3/V2；
2. Structured anchor candidates 与 free kinematic candidates 的互补性，能否在固定 K=20 下被稳定利用。

统一对照为 `F16 + H3 + V2 + original score`，即 EXP-164 seed-1；决策指标为：

```text
J = minADE20 + 0.5 × minFDE20
```

## EXP-165：Fold Shift + Candidate Geometry Causal Audit

本实验不训练新模型，使用五个官方 fold 检验：

- ETH 是否存在速度尺度导致的候选半径失配；
- Univ 是否存在高密度、低速度条件下的横向覆盖不足。

结果：

- ETH scale SMD = 0.658，没有达到 0.8；
- oracle gap 不随速度单调恶化；
- correction magnitude 与速度的相关性不足；
- Univ 的退化没有集中在预设的高密度条件，横向收窄假设也未成立。

结论：EXP166A 和 EXP167 均不允许启动。

## EXP-168A：Joint Hybrid Generator K20

模型固定输出 K=20：

- 10 个 structured modes：H3 三锚点 + 联合训练的 V2 bounded correction；
- 10 个 free modes：matched-K20 kinematic/free generator；
- 两类候选共享 F16 encoder 和 coverage objective；
- 不存在 K40 后压缩，不加入 family balancing loss。

### 五折结果

| Fold | minADE20 | minFDE20 | J | J 相对 EXP-164 V2 |
|---|---:|---:|---:|---:|
| ETH | 0.4138 | 0.6298 | 0.7287 | 退化 2.11% |
| Hotel | 0.1374 | 0.2179 | 0.2463 | 改善 1.57% |
| Univ | 0.2209 | 0.3977 | 0.4197 | 改善 7.59% |
| Zara1 | 0.1825 | 0.3378 | 0.3514 | 退化 0.72% |
| Zara2 | 0.1328 | 0.2418 | 0.2538 | 改善 1.49% |
| Macro | 0.2175 | 0.3650 | 0.4000 | 改善 1.22% |

与 EXP-164 seed-1 相比：

- macro minADE 改善 2.24%；
- macro minFDE 基本不变，退化 0.03%；
- same-path FDE 退化 1.87%；
- macro J 改善 1.22%，低于 2% 晋级门槛；
- ETH J 退化 2.11%，超过允许的 2%。

两类 family 后期责任均超过 10%，所有 mode 都成为过 winner，因此失败原因不是 family collapse。最终判定为 FAIL，不补 seeds 2/3，也不直接尝试 14/6。

## EXP-168B：Family Attribution Audit

本实验拆分 structured K10、free K10 和 union K20，在 34,161 个 outer-test 样本上分析候选责任与几何。

### ADE winner fraction

| Fold | Structured | Free |
|---|---:|---:|
| ETH | 43.7% | 56.3% |
| Hotel | 47.2% | 52.8% |
| Univ | 28.8% | 71.2% |
| Zara1 | 16.3% | 83.7% |
| Zara2 | 16.6% | 83.4% |

Structured family 更宽、更远、无重复，负责低频尾部；free family 更集中，负责常见运动区域。两类 family 没有同质化。

Univ 中 structured winner 随速度从 Q1 的 13.3% 上升到 Q4 的 39.8%，说明 family need 与 observation condition 存在关系。但固定 10/10 配额不能同时适配 ETH 高速尾部和 Univ/Zara 的常见运动分布。

## EXP-168C：Family Need Predictability Audit

本实验不读取 outer-test，而是在每个 outer fold 保存的 5% inner-validation subset 上，使用 observation-only 特征预测哪个 family 包含最低 `ADE + 0.25×FDE` 候选，共 6,227 个样本。

### Full-feature logistic regression

| Fold | Balanced accuracy | ROC-AUC | 相对 constant-free 的质量变化 |
|---|---:|---:|---:|
| ETH | 0.618 | 0.671 | 退化 3.46% |
| Hotel | 0.616 | 0.663 | 退化 0.97% |
| Univ | 0.785 | 0.858 | 退化 0.68% |
| Zara1 | 0.616 | 0.667 | 退化 4.06% |
| Zara2 | 0.639 | 0.680 | 改善 4.22% |
| Macro | 0.654 | 0.708 | 退化 0.99% |

分类门槛通过，但候选质量门槛失败。原因是二元 winner 标签忽略胜出幅度，false-positive 的非对称代价抵消了正确分类的收益。

## 最终结论

这一实验链支持以下结论：

> Structured 与 free candidates 存在真实、可预测的功能差异，但这种互补性既不能通过全局固定配额稳定利用，也不能通过 observation-only 硬 family gate 转化为可靠的质量收益。

因此关闭：

- EXP166/167 的 scale/density targeted fixes；
- EXP168 固定配额搜索；
- family classifier、阈值校准和硬动态配额；
- EXP168 seeds 2/3。

下一阶段需要提出新的候选生成机制假设，不再沿 scorer、selector、quota 或 hard gate 路线继续调参。

## 仓库数据范围

GitHub 保存：

- 实验与审计脚本；
- README 和 RESULT_ANALYSIS；
- 聚合 summary JSON/CSV；
- 每折训练 history、metrics 和模型契约；
- family responsibility 与分箱统计。

GitHub 不保存：

- 数据集；
- 模型 checkpoint；
- split indices；
- console logs；
- 逐 batch family diagnostics；
- 可由脚本重新生成的大型逐样本 CSV。
