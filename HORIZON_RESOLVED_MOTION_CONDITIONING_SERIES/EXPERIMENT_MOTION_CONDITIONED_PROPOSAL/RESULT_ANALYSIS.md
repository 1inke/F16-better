# Horizon-Resolved Motion Conditioning 结果分析

## 完整性

- P1/C1 均完成 ETH、Hotel、Univ、Zara01、Zara02 五折 seed 1。
- 正式对照为 EXP-164 的同折 seed-1 V2。
- 每个变体新增 8,580 个可训练参数，F16、H3、V2 主体与 score 均冻结。
- 初始化与 V2 完全一致，并通过真实数据前向、反向、checkpoint 保存和评估器试运行。

## Macro 结果

| Model | minADE20 | minFDE20 | same-path FDE | J |
|---|---:|---:|---:|---:|
| V2 | 0.222457 | 0.364897 | **0.433931** | 0.404905 |
| P1 proposal residual | **0.221492** | **0.363671** | 0.434807 | **0.403328** |
| C1 correction residual | 0.222315 | 0.364399 | 0.434184 | 0.404515 |

P1 相对 V2 的 macro 改善约为 minADE 0.44%、minFDE 0.34%、J 0.39%。C1 仅改善约 0.06%/0.14%/0.10%。两者均未达到值得补多种子的幅度。

## 分折结果

| Fold | P1 minADE change | P1 minFDE change | C1 minADE change | C1 minFDE change |
|---|---:|---:|---:|---:|
| ETH | -0.00% | -0.28% | +0.02% | +0.26% |
| Hotel | -0.28% | -0.27% | -1.97% | -0.89% |
| Univ | +1.70% | +0.75% | +1.31% | +0.48% |
| Zara01 | +0.44% | +0.11% | +0.13% | +0.01% |
| Zara02 | +0.31% | +2.03% | +0.10% | +0.33% |

收益主要来自 Univ 和 Zara02，ETH/Hotel 没有稳定改善。

## 运动条件结果

P1 high-acc macro：

- minADE：+0.77%
- minFDE：+0.94%
- J：+0.84%

P1 non-high-acc macro：

- minADE：-0.78%
- minFDE：-3.10%
- J：-1.77%

P1 学到的平均 adapter norm 随 horizon 增大：t4/t8/t12 为 0.0176/0.0430/0.0819。模块确实产生了动态响应，但它对普通运动存在过度修正，并且没有改善 same-path FDE。

## 决策

- C1 淘汰。
- P1 作为“proposal 端优于 correction 端”的消融证据保留，但不替代 V2。
- 不补 seeds 2/3。
- 不继续调整 residual cap、网络深度或 gate；这些变化不可能把当前约 0.4% 的收益扩大到目标量级。
- 当前运动小残差路线结束。下一阶段若继续提高候选覆盖，需要重新设计直接产生不同候选几何/模板的机制，并建立新的预注册实验指导。
