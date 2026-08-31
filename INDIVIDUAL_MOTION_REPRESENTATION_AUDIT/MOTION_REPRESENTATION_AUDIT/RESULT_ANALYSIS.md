# Individual Motion Representation Audit 结果分析

## 1. 完整性

- 五折、seed 1 正式审计已完成，无模型训练或 checkpoint 修改。
- EXP-164 official F16+H3+V2 的 minADE、minFDE、same-path FDE 均逐折复现；最大绝对误差为 `5.96e-8`。
- 因此，本次内部表征与误差对应关系建立在正确的正式 baseline 上。

## 2. 高加速度确实是稳定的困难条件

Q4 使用每个 outer setup 的 source-train Q75 阈值，而不是目标测试集调参。

| Fold | High-acc samples | minADE gap | minFDE gap | J gap |
|---|---:|---:|---:|---:|
| ETH | 325/364 | +118.27% | +167.87% | +136.81% |
| Hotel | 724/1197 | +261.11% | +282.81% | +270.29% |
| Univ | 7620/24334 | +48.37% | +40.16% | +44.43% |
| Zara01 | 403/2356 | +29.73% | +29.26% | +29.50% |
| Zara02 | 806/5910 | +114.06% | +107.87% | +111.12% |
| Macro | — | +114.31% | +125.59% | +118.43% |

五折方向完全一致。加速度与 J 的 Spearman 相关分别为 ETH 0.244、Hotel 0.700、Univ 0.422、Zara01 0.201、Zara02 0.697。

这证明高加速度是可靠的误差分层条件，但不能把分桶差异直接解释成加速度的因果效应；速度、场景组成和未来不确定性仍可能共同影响误差。ETH 中 89.3% 样本超过 source Q75，首先也是明显的跨场景分布偏移证据。

## 3. 不支持“加速度信息在编码器中消失”

source-inner 拟合、outer-fold 测试的加速度线性 probe：

| Representation | Macro outer R² | Macro Pearson | Macro Spearman |
|---|---:|---:|---:|
| robust_state | **0.447** | **0.834** | **0.850** |
| temporal | 0.030 | 0.805 | 0.754 |
| post_social | 0.284 | 0.786 | 0.753 |
| encoder_output | 0.336 | 0.824 | 0.799 |

`robust_state` 在五折的加速度 R² 全部为正（0.246–0.737），说明基础运动状态编码已经明确保存了加速度信息。经过 temporal、social 和最终融合后，加速度排序仍然高度可解码，最终 encoder 的 macro Spearman 仍为 0.799。

因此，当前证据不支持再增加一个更复杂的 acceleration encoder，也不支持重复 Enhanced Kinematic State、速度尺度分支或宽范围运动增强。

## 4. 范数变化不是可信的单独机制

高加速度相对非高加速度的平均 effect size：

- robust norm：-0.534
- temporal norm：+0.388
- post-social norm：-0.507
- final encoder norm：-0.491

但各折符号明显不一致。例如 temporal norm 在 Hotel/Univ/Zara02 随加速度增加，在 ETH/Zara01 中反向；最终 encoder 也同时存在正负方向。`robust_saturation_fraction` 在所有分桶中为 0%，不存在 tanh 输出贴近 ±0.9 的数值饱和证据。

所以不能据此提出 norm rescaling、norm gate 或高加速度放大器。LayerNorm 后的向量方向携带了大量信息，单看模长会丢失这一事实。

## 5. 真正异常：信息可排序，但跨场景绝对映射失准

多个 probe 出现“相关性高、R²却很低或为负”：

- ETH encoder acceleration：Spearman 0.790，R² -0.177。
- Hotel temporal acceleration：Spearman 0.932，R² -0.696。
- Hotel temporal future-J：Spearman 0.766，R² -6.991。

这不是没有信息，而是 source-inner 学到的绝对映射不能直接校准到 held-out scene。数据均值也显示明显 shift：

- ETH mean acceleration：inner 0.0275 → outer 0.1253；J：0.4190 → 0.7136。
- Hotel mean acceleration：inner 0.0283 → outer 0.0526，但 J：0.4343 → 0.2503。
- Zara02 mean acceleration：inner 0.0309 → outer 0.0162；J：0.4754 → 0.2576。

未来 J 的跨域线性 R² 在四个表征阶段总体为负，而后段表征的 rank correlation 通常仍有约 0.37–0.40。模型知道哪些样本更困难的一部分排序信息，但 source→target 的绝对尺度关系不稳定。

## 6. 机制结论

本次审计排除了两个简单解释：

1. 不是 social branch 在 ETH-LDHA 中伤害预测（EXP-169B 已否定）。
2. 不是 robust/temporal encoder 完全没有编码加速度（本实验否定）。

剩余更可能的断点位于：

> 已编码的个体运动状态，是否被候选生成与锚点修正以正确的幅度和几何方向使用。

这属于 representation-to-candidate transfer / decoder conditioning 问题，而不是再堆叠一个运动编码器的问题。

## 7. 下一步建议

下一步先做零训练的 `Motion-to-Candidate Transfer Audit`，不要直接训练新结构。需要测量：

- 候选终点半径、纵向/横向 spread、锚点位移与观测加速度/非线性的关系；
- source-inner 与 outer-fold 中上述响应斜率是否发生尺度或符号偏移；
- 高加速度分桶内 GT future geometry 是否超出当前 V2 候选 envelope；
- encoder 中可解码的加速度排序是否传递到候选几何排序；
- 在保持终点/候选预算不变时，当前 bounded correction 是否对高加速度样本系统性欠响应。

只有当该审计确认候选几何对运动状态欠响应，才应设计轻量的 continuous motion-conditioned anchor/correction 模块；不能从当前结果直接批准新的 encoder 分支。
