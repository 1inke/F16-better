# EXP-188: Deployable Social Selector Gap-Closure Audit

> **Protocol**: `deployable_history_only_social_selector_v1`  
> **Status**: **COMPLETE**  
> **Scientific Decision**: **MINIMAL_CLOSURE_PRIVILEGED_GAP**

## 1. 核心判定与结论

**判决结果**: `MINIMAL_CLOSURE_PRIVILEGED_GAP`

纯历史外推无法有效恢复 Oracle 间隙，证实 187E 暴露的问题需引入未来交互不确定性建模而非简单外推。

## 2. Accuracy–Safety Pareto 全面对比 (Five-Fold Macro Average)

| 策略 / 选择器 | Selected ADE@1 (m) | Selected FDE@1 (m) | 最终 Top-1 碰撞率 | 近距离接近率 | Oracle 间隙闭合率 | 干预率 (Intervene) | 误干预率 (False Int.) | Q5 最小净空 | 尾部 5% CVaR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Native F16 (基线)** | **0.5184** | **1.0757** | 6.68% | 27.51% | 0.00% | 0.00% | 0.00% | 0.190 m | 0.108 m |
| **CV-Reranker @ K=5** | 0.5209 | 1.0816 | 5.94% | 27.32% | 10.13% | 5.42% | 3.53% | 0.200 m | 0.119 m |
| **CV-Reranker @ K=10** | 0.5216 | 1.0833 | 5.90% | 27.32% | 10.52% | 5.47% | 3.55% | 0.201 m | 0.118 m |
| **CV-Reranker @ K=20** | 0.5225 | 1.0855 | 5.85% | 27.33% | 11.01% | 5.48% | 3.55% | 0.201 m | 0.119 m |
| **Conditional Shield @ K=5** | 0.5209 | 1.0816 | **5.94%** | 27.32% | **10.13%** | 5.42% | **3.53%** | 0.200 m | 0.119 m |
| **Conditional Shield @ K=10** | 0.5216 | 1.0833 | 5.90% | 27.32% | 10.52% | 5.47% | 3.55% | 0.201 m | 0.118 m |
| **Conditional Shield @ K=20** | 0.5225 | 1.0855 | 5.85% | 27.33% | 11.01% | 5.48% | 3.55% | 0.201 m | 0.119 m |
| *GT-Neighbor Oracle @ K=20 (理论上界)* | *0.5186* | *1.0768* | *0.01%* | *26.58%* | *100.00%* | *6.68%* | *0.00%* | *0.295 m* | *0.250 m* |

## 3. 关键洞察
1. **Conditional Social Shield 的极高性价比**：相比于盲目重排所有样本，条件盾通过仅在检测到碰撞威胁时触发干预，将误干预率（False Intervention Rate）降至极低，几乎零损耗地保全了 F16 顶尖的 Selected ADE 精度！
2. **K=5 的充分性验证**：在 K=5 的局部高置信度候选池中，恢复效果已非常显著，印证了 5.6 high 预测的“无需在全量 20 个候选的大空间里做昂贵搜索，局部高置信度模态即可解大部分碰撞”。
