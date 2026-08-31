# EXP-169A 结果分析：Cross-Family Horizon Complementarity Audit

## 完整性与口径

- 状态：完整，ETH、Hotel、Univ、Zara01、Zara02 五折，seed 1。
- 数据源：EXP-168A 的 jointly-trained `10 structured + 10 free` 候选。
- 本实验是 oracle 诊断，不是部署指标。下文的“增益”表示 union K20 相对 structured K10 的 oracle 边际收益，不能解释为模型相对 EXP-164 V2 已获得同等提升。

## 核心结果

| Horizon | Macro prefix oracle gain | Macro point oracle gain | Reach@0.5×V2 | Reach@1.0×V2 |
|---|---:|---:|---:|---:|
| t4 | 31.68% | 35.23% | 47.17% | 80.17% |
| t8 | 38.90% | 49.30% | 25.90% | 33.98% |
| t12 | 41.15% | 48.69% | 27.95% | 36.14% |

五折在 t4、t8、t12 的 union oracle 收益方向全部一致，说明 structured/free 的几何互补性真实且很强。但是，收益并非集中在早中段：t12 的 prefix 与 point 增益不低于 t4/t8，因此“只修正早中段并保持终点”的假设不成立。

更关键的是可达性。按设计要求的半 V2 correction budget，t8 和 t12 只有 25.90% 与 27.95% 的 free winner 能从最近 structured candidate 局部到达；即使放宽到完整 V2 budget，也只有 33.98% 与 36.14%。这说明两族之间多数有价值差异不是小 residual，而是候选落点/模式层面的较大差异。

winner family 在三个时刻始终不变的样本占 63.31%，发生切换的占 36.69%。切换确实存在，但不足以推翻低 reachability 的主要限制。

## 分折观察

- ETH：半预算 reach 在 t4/t8/t12 仅 24.5%/10.6%/14.9%，局部吸收尤其困难。
- Zara01：oracle 互补性最强之一，但半预算 t8/t12 reach 仅 5.3%/5.0%，典型地表现为“价值高但不可局部吸收”。
- Zara02：可达性最好，半预算 t4/t8/t12 为 69.2%/50.9%/49.9%，但单折证据不能支持跨场景统一模块。
- Hotel：t12 半预算 reach 为 48.6%，接近阈值但仍未达到 endpoint 分支要求。

## 预注册判定

判定为 `C_complementary_but_unreachable`：

- `EXP170A_ALLOWED = false`
- `EXP170B_ALLOWED = false`
- 不应扩大 residual cap 强行追逐 oracle。

因此，EXP-168 的正确信息是“两类候选覆盖不同未来”，而不是“structured 候选可通过轻微锚点修正吸收 free 候选”。固定配额、hard family selector、早中段 dual-basis residual 和 full-anchor dual-basis 都不应继续。

## 结论

该路线的瓶颈不是互补性不足，而是互补区域距离 structured 候选过远，同时现有评分机制又无法可靠兑现 oracle。按照指导文件，本分支停止，不启动 EXP-170A/170B。
