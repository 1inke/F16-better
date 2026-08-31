# EXP-168B Hybrid Family Attribution Audit 结果分析

## 1. 状态与结论

五个官方 outer-test fold 均已完成，共审计 34,161 个样本。本实验没有训练新模型。

核心结论：

1. EXP-168A 的失败不是 family responsibility collapse；
2. structured 和 free family 在所有 fold 中都提供独有的 oracle winner；
3. 固定 10/10 配额存在明显的场景适配冲突；
4. family 需求与 observation speed 存在可预测关系，密度证据相对弱；
5. 该结果支持研究 observation-conditioned family utilization，但不允许直接根据 outer-test 手调固定配额。

## 2. 五折 family attribution

| Fold | Structured K10 J | Free K10 J | Union K20 J | Structured ADE winner | Free ADE winner |
|---|---:|---:|---:|---:|---:|
| ETH | **1.0861** | 1.3941 | **0.7287** | 43.7% | 56.3% |
| Hotel | 0.4398 | **0.3359** | **0.2463** | 47.2% | 52.8% |
| Univ | 0.6574 | **0.5776** | **0.4197** | 28.8% | 71.2% |
| Zara1 | 0.9082 | **0.4400** | **0.3514** | 16.3% | 83.7% |
| Zara2 | 0.5071 | **0.3560** | **0.2538** | 16.6% | 83.4% |

FDE winner 也呈现相同结构：structured 占 17.2%–44.5%，free 占 55.5%–82.8%。因此 structured 不是冗余分支，但其需求量具有强烈场景差异。

注意：Union 相对单独 family 的大幅改善同时包含 K10→K20 的候选数量效应，不能全部归因于跨 family 互补。更可靠的互补证据是两个 family 在每个 fold 都存在数量可观的独有 winner，以及两类候选几何明显不同。

## 3. 几何差异

Structured family：

- 分支内路径距离约 1.65–2.18；
- lateral std 约 1.52–1.95；
- duplicate rate 为 0；
- 负责更宽、更远、低频的尾部覆盖。

Free family：

- 分支内路径距离约 0.71–0.85；
- lateral std 约 0.63–0.80；
- 少量重复候选；
- 负责更密集的常见运动区域。

两类 family 的功能没有同质化。

## 4. ETH 失败机制

ETH 中 structured 是单独表现更好的 family，但 free 仍贡献 56.3% 的 ADE winner 和 55.5% 的 FDE winner。因此 ETH 不能简单退回“全 structured”，而是需要两类候选。

ETH 的最高速度组 Q4 包含 192/364 个样本：

| Family | Q4 J | ADE winner | FDE winner |
|---|---:|---:|---:|
| Structured | **1.3018** | 43.2% | 45.3% |
| Free | 1.6508 | 56.8% | 54.7% |
| Union | **0.8401** | — | — |

高速度样本需要 structured 提供更远、更宽的覆盖，但仍有超过一半样本由 free 获胜。固定削减任一 family 都会损失覆盖；10 个 structured modes 对 ETH 的尾部终点可能不足，而简单增加 structured 又可能删除必要的 free winner。

## 5. Univ 改善机制

Univ 的 free family 负责大多数常见轨迹，structured 保留少量但重要的尾部模式。Structured ADE winner 随速度单调上升：

| Speed bin | Structured ADE winner | Structured FDE winner |
|---|---:|---:|
| Q1 | 13.3% | 19.6% |
| Q2 | 29.3% | 32.5% |
| Q3 | 33.1% | 34.7% |
| Q4 | 39.8% | 42.7% |

这说明速度越高，structured family 的必要性越强；低速和常见样本主要由 free family 覆盖。这是 Univ 获得 7.59% J 改善的主要机制解释。

## 6. Density 证据

Univ 中 free ADE winner fraction：

- high density：72.1%；
- medium density：64.9%；
- low density：69.0%。

变化不是清晰单调关系。ETH 样本按 outer-train 阈值全部落入 low-density，无法在该 fold 内检验密度效应。因此目前速度是更可信的条件变量，密度不适合作为下一轮唯一 gate。

## 7. 对 EXP-168 路线的最终判断

EXP-168A 10/10 仍维持 FAIL：不补 seeds 2/3，不按原指导直接追加 14/6。

但 mixed-family 机制本身不应被解释为无效。更准确的结论是：

> 两类候选具有真实且可预测的功能差异，但固定、全局一致的候选配额不能同时适配 ETH 的高速尾部和 Univ/Zara 的常见运动分布。

如果继续该路线，下一步必须在 source-scene inner validation 中设计并冻结 observation-conditioned family utilization；禁止根据本次 outer-test 的 ETH/Univ 数字直接设定速度阈值或具体配额后重复读取同一测试折。
