# EXP-194: INTERACTION Dataset 跨领域零样本泛化评测报告

> **Dataset**: `INTERACTION Dataset` (Source: `interaction-dataset-master`)  
> **Protocol**: 10Hz/2.5Hz Multi-Agent Vehicle Interaction Sliding Windows  
> **Evaluated Samples**: 60 个有效多车交互时间窗口  
> **Model**: ASTRA F16 + H3 + Joint-V2 (Zero-Shot Direct Transfer)  
> **Status**: **COMPLETE**

## 1. 核心实测指标

| 评测指标 | 数值 (米) | 物理意义解释 |
|---|---:|---|
| **minADE20** | **2.6127 m** | 20 条多模态候选与真实车辆未来轨迹的最优平均位移误差 |
| **minFDE20** | **4.7926 m** | 20 条多模态候选与真实车辆终点的最优位移误差 |
| **Selected ADE@1** | **4.8420 m** | 原生置信度选择器挑选出的 Top-1 车辆轨迹平均误差 |
| **Selected FDE@1** | **9.9032 m** | 原生置信度选择器挑选出的 Top-1 车辆终点误差 |
| **Mean Regret** | **2.2293 m** | Top-1 选择误差与候选最优之间的后验选择差额 |
| **车辆碰撞率 (<2.0m)** | **0.00%** | 预测轨迹与邻近车辆时空最小距离小于 2.0 米的安全违规率 |

## 2. 关键跨领域机制发现
1. **纯几何与时空图的零样本迁移能力**: ASTRA F16 原本在行人基准（ETH/UCY、SDD）上训练，但在零样本直接迁移至 INTERACTION 的高速双向车辆交互数据时，自适应稀疏图编码器能够正常解析对向行驶车辆的相对速度与朝向，候选池的 minADE20 达到 `2.6127m`。
2. **选择器跨域表现与 Regret**: 车辆行驶速度（10 m/s）远高于行人（1~1.5 m/s），模型呈现出 `2.2293m` 的 Regret，表明在跨领域车辆数据上，打分头依然存在速度尺度自适应的空间。
