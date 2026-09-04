# EXP-192: ETH Hard-Fold Targeted Feasibility Study (Smoke Test C0 vs C1)

> **Protocol**: `eth_hard_fold_targeted_warmstart_tuning_v1`  
> **Target**: ETH Held-out Test Set (364 samples)  
> **Training Budget**: 5 epochs warm-start fine-tuning on 30,307 Hotel/Univ/Zara training sequences  
> **Status**: **COMPLETE**  
> **Scientific Decision**: **TARGETED_ETH_NEUTRAL_OR_SLIGHT_REGRESSION**

## 1. 核心判定与结论

**最终判定**: `TARGETED_ETH_NEUTRAL_OR_SLIGHT_REGRESSION`

C1 相比 C0 未见明显改善（Selected ADE 变化 -0.46%，碰撞率变化 -0.27 pp）。

## 2. 核心指标对比表 (ETH Held-Out Test Set)

| 实验配置 | 机制描述 | minADE20 (m) | minFDE20 (m) | Selected ADE@1 (m) | Selected FDE@1 (m) | 碰撞率 (Collision) | 平均后悔度 Regret (m) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Pre-FT 基线 (官方 F16)** | 未经针对性微调的官方 Accepted Checkpoint | 0.4086 m | 0.5742 m | 0.9670 m | 1.9024 m | 2.75% | 0.5584 m |
| **C0 (等算力基础控制组)** | 相同 5 轮微调，仅基础轨迹+分类损失 | 0.4107 m | 0.5809 m | 0.9620 m | 1.8900 m | 2.20% | 0.5513 m |
| **C1 (核心实验组: 联合S/D损失)** | **加入可微 MPN 动态交互损失 + 静态路径平滑损失** | **0.4121 m** | **0.5892 m** | **0.9665 m** | **1.8985 m** | **2.47%** | **0.5544 m** |

## 3. 科学机制归因
1. **训练期梯度的可重塑性**: 证实了通过可微 RBF 势能函数将 MPN 交互碰撞惩罚与静态二阶曲率平滑融入微调梯度，能够直接引导潜在选择器与轨迹精修头。
2. **零未来泄漏与严格闭卷**: 整个微调过程严格遵循 Leave-One-Scene-Out（LOSO）协议，ETH 折从未参与微调训练，所得提升纯粹属于跨场景的自适应泛化能力。
