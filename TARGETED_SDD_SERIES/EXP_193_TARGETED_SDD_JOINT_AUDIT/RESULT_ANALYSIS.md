# EXP-193: SDD Targeted Joint Training Feasibility Study (C0 vs C1)

> **Protocol**: `sdd_standard_31_12_17_split_targeted_joint_tuning_v1`  
> **Target**: Full SDD Test Split (2,679 samples across 17 held-out videos)  
> **Training Budget**: 4 epochs warm-start fine-tuning on 8,418 training sequences  
> **Status**: **COMPLETE**  
> **Scientific Decision**: **SDD_TARGETED_NEUTRAL_OR_SLIGHT_REGRESSION**

## 1. 核心判定与结论

**最终判定**: `SDD_TARGETED_NEUTRAL_OR_SLIGHT_REGRESSION`

C1 相比 C0 未见明显改善（Selected ADE 变化 -1.51%，Regret 变化 -1.89%）。

## 2. 核心指标对比表 (SDD Held-Out Test Set)

| 实验配置 | 机制描述 | minADE20 (px) | minFDE20 (px) | Selected ADE@1 (px) | Selected FDE@1 (px) | 平均后悔度 Regret (px) | 困难 little 场景 ADE (px) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Pre-FT 基线 (EXP-175S0)** | 官方已归档 Accepted Checkpoint | 7.577 px | 12.387 px | 17.267 px | 34.581 px | 9.690 px | 25.869 px |
| **C0 (等算力基础控制组)** | 相同 4 轮微调，仅基础轨迹+分类损失 | 7.519 px | 12.273 px | 17.620 px | 35.272 px | 10.101 px | 25.900 px |
| **C1 (核心实验组: 联合S/D损失)** | **加入可微 MPN 动态交互损失 + 静态路径平滑损失** | **7.595 px** | **12.424 px** | **17.887 px** | **35.729 px** | **10.291 px** | **26.298 px** |

## 3. 科学归因
1. **真实校园多场景泛化验证**: SDD 包含 17 个完全未见过的独立测试视频，测试了真正的跨场景、多主体泛化表现。
2. **后处理错配鸿沟的实测表现**: 评估了动态交互避碰势能核与静态时空平滑损失对 9.69 px 巨大 Regret 鸿沟的直接驱动与压缩能力。
