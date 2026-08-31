# EXP-168C Family Need Predictability Audit 结果分析

## 1. 状态与最终判定

实验完整完成，共使用五个 outer fold 内部的 6,227 个 validation 样本；没有读取 outer-test，也没有训练新的轨迹模型。

最终判定：**conditional allocator 不允许晋级**。

Family winner 具有一定 observation-level 可预测性，但当前预测无法转化为候选质量收益。分类准确不等于分配决策有效。

## 2. Macro 结果

### Speed-only

- balanced accuracy：约 0.601；
- ROC-AUC：约 0.625；
- 相对 constant-free 的质量变化：约退化 5.08%。

### Full observation features

- balanced accuracy：**0.6545**；
- ROC-AUC：**0.7076**；
- 相对 constant-free 的质量变化：**退化 0.99%**；
- oracle regret closed：**-4.07%**。

增加加速度、社会特征等信息显著提高了分类可分性，但没有产生宏观候选质量收益。

## 3. 分场景结果（Full features）

| Fold | Structured prevalence | Balanced accuracy | ROC-AUC | 相对 constant-free 质量变化 |
|---|---:|---:|---:|---:|
| ETH | 24.1% | 0.618 | 0.671 | **退化 3.46%** |
| Hotel | 23.8% | 0.616 | 0.663 | 退化 0.97% |
| Univ | 51.8% | **0.785** | **0.858** | 退化 0.68% |
| Zara1 | 23.2% | 0.616 | 0.667 | **退化 4.06%** |
| Zara2 | 28.2% | 0.639 | 0.680 | **改善 4.22%** |

Zara2 是唯一得到正质量收益的 fold。Univ 的分类指标非常好，但决策质量仍略差于 constant-free，说明 winner 数量与 winner 收益幅度并不一致。

## 4. 为什么分类通过而质量失败

五个 fold 的最佳 constant policy 都是始终选择 free family，即使 Univ 中 structured winner 数量达到 51.8%。这说明：

- structured 可能在更多样本上小幅获胜；
- free 在较少样本上的获胜幅度可能更大；
- 只学习二元 winner 标签会忽略质量差值。

Full classifier 的错误结构如下：

| Fold | Structured precision | Structured recall | Free recall | False-positive 平均代价 | False-negative 平均代价 |
|---|---:|---:|---:|---:|---:|
| ETH | 0.366 | 0.525 | 0.710 | 0.409 | 0.345 |
| Hotel | 0.355 | 0.535 | 0.696 | 0.327 | 0.323 |
| Univ | 0.814 | 0.754 | 0.815 | 0.342 | 0.443 |
| Zara1 | 0.346 | 0.539 | 0.692 | 0.381 | 0.336 |
| Zara2 | 0.428 | 0.587 | 0.691 | 0.316 | 0.338 |

ETH、Hotel、Zara1 和 Zara2 的 structured prevalence 较低；class-balanced logistic regression 为提高 structured recall，产生了大量 false positive。错误切换到 structured 的代价抵消了正确识别 structured winner 的收益。

## 5. 特征解释

标准化 logistic coefficient 的平均绝对值中，贡献较大的特征包括：

1. mean acceleration；
2. last-step speed；
3. mean-step speed；
4. mean neighbour distance；
5. neighbour count within radius 2。

这说明 family need 不只由速度决定，短期运动变化与局部社会环境也参与。但速度特征之间高度相关，coefficient 不能解释为因果效应。

## 6. 晋级门槛

| 条件 | 结果 |
|---|---|
| Macro balanced accuracy ≥ 0.60 | 通过 |
| 所有 fold balanced accuracy ≥ 0.55 | 通过 |
| Macro ROC-AUC ≥ 0.65 | 通过 |
| 候选质量改善 ≥ 3% | **失败：实际退化 0.99%** |

总体：**FAIL**。

## 7. 研究结论

EXP-168B/168C 联合支持以下更精确的结论：

> Structured/free family 的需求在统计上可由 observation features 部分预测，但这种 winner 可预测性不足以支持可靠的硬 family 决策；错误分配的非对称代价会抵消分类收益。

因此不应进入基于该二元 gate 的动态配额训练，也不应继续调分类阈值、类别权重或更复杂分类器。后者会重新进入 scorer/calibration 路线，并增加 inner-validation 过拟合风险。

当前证据支持关闭固定配额和硬条件分配两条分支，把 EXP-168 保留为“跨 family 互补存在，但在固定 K20 内难以稳定利用”的机制结论。
