# EXP-169B 结果分析：Official Sparse-Social Counterfactual Audit

## 完整性与因果口径

- 状态：完整，ETH、Hotel、Univ、Zara01、Zara02 五折，seed 1。
- 模型：EXP-164 official V2 seed 1，同一 checkpoint、同一输入、同一 decoder。
- 干预：精确复用 CSNA 的 focal receive-row self-loop，仅移除目标行人接收的他人社交信息。
- Full-social 推理逐折复现 EXP-164；最大指标绝对误差为 `5.96e-8`，审计实现通过一致性检查。

正的 Full advantage 表示保留完整社交信息优于 Self-only。

## Outer-test 结果

| Fold | ALL ADE | ALL FDE | ALL J | LDHA ADE | LDHA FDE | LDHA J |
|---|---:|---:|---:|---:|---:|---:|
| ETH | 3.69% | 4.45% | 4.01% | 3.90% | 4.64% | **4.21%** |
| Hotel | 7.48% | 11.79% | 9.36% | 8.81% | 13.89% | 11.03% |
| Univ | 0.63% | 1.38% | 0.98% | -0.72% | -1.39% | -1.03% |
| Zara01 | 6.84% | 9.29% | 8.00% | 6.33% | 5.75% | 6.06% |
| Zara02 | 2.38% | 3.04% | 2.69% | 4.61% | 4.68% | 4.64% |
| Macro | 4.20% | 5.99% | 5.01% | 4.59% | 5.51% | 4.98% |

ETH 的 LDHA 桶包含 325 个样本，Full Social 的 J 优势为 4.21%，超过预注册的 2% 门槛。说明 ETH 的低密度、高加速度困难并不是 social over-injection；去掉交互反而使 ADE、FDE 同时恶化。

Univ-LDHA 是唯一 outer 负向桶，但幅度约 1%，而且与其他四折方向不一致，不能据此设计全局 social suppression。

## Source-inner 复核

五个 source-inner LDHA proxy 的 Full J 优势分别为：

- ETH：+2.28%
- Hotel：+0.14%
- Univ：-0.76%
- Zara01：+0.66%
- Zara02：+1.77%

4/5 个 inner setup 方向为正，满足 B1 对可重复方向的要求。虽然其中三个绝对变化小于 1%，但没有任何 inner setup 显示 Self-only 优势达到 1%，因此不存在支持 B3 的稳定社交伤害证据。

## 预注册判定

判定为 `B1_full_social_beneficial`：

- ETH outer LDHA Full J 优势：4.21%。
- source-inner 中 Full 更优：4/5。
- `EXP169C_ALLOWED = false`。

因此禁止继续：

- density gate；
- social suppression；
- Self-only fallback；
- EXP-169C support audit；
- EXP-171A support-calibrated social residual。

## 结论与下一步

F16 的社交分支总体有效，在 ETH-LDHA 中也不是短板。ETH 的剩余误差更可能来自目标行人自身的高加速度/非线性运动表征不足。

按指导文件，下一步应先做零训练的 `individual motion representation audit`，检查 `robust_state`、temporal feature、encoder output 的范数/方向与观测加速度、非线性和未来误差之间的关系。只有定位到明确的 representation failure 后，才设计新的编码模块，避免重复已经失败的速度尺度分支、难度辅助损失和宽范围运动增强。
