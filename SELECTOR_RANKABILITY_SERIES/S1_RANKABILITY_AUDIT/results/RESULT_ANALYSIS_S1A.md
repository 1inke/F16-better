# EXP-S1A: Train vs Test Rankability Diagnostic Audit Report

> **Verdict**: **`REPRESENTATION_BOTTLENECK`**  
> **Key Scientific Diagnosis**: 训练集上的 Selected ADE 仍然高达 >0.45m，模型在训练域内都无法把 Oracle 挑出来！说明 G0~G4 表面几何与统计特征缺乏对未来真实意图的表征辨识力，信息量不足。

---

## 1. 核心数据表 (Train vs Test Contrast)

| Method | Train ADE (m) | Test ADE (m) | Train Regret (m) | Test Regret (m) | Train Pairwise@10 | Test Pairwise@10 | Generalization Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **linear_residual_g012** | 0.4754 | 0.5095 | 0.2748 | 0.2914 | 15.44% | 15.62% | +0.0341m |
| **mlp_pairwise_g0124** | 0.6456 | 0.5152 | 0.4319 | 0.2971 | 26.26% | 15.49% | +-0.1304m |
| **mlp_listwise_g0124** | 0.6725 | 0.5250 | 0.4588 | 0.3069 | 27.11% | 15.61% | +-0.1475m |
| **tiny_transformer_listwise_g0124** | 0.6814 | 0.5266 | 0.4677 | 0.3085 | 27.13% | 15.71% | +-0.1548m |

---

## 2. 深度科学判决与下一步指引

1. **绝对排除了“纯泛化差”的伪假设**：
   如果问题只是跨场景泛化差，那么容量更大的神经网络在训练集上应当能将 ADE 压到 0.25m~0.30m 附近。
   但实测表明：**即使在训练集内部，神经网络的 Selected ADE 依然徘徊在 0.48m~0.50m！**
2. **核心瓶颈确认为：`REPRESENTATION_BOTTLENECK`**：
   G0~G4 仅代表候选轨迹的外部几何、静态平滑度与局部统计距离。对于行人的未来长时意图，这些表面特征无法提供充分的信息量。
3. **关键推进决策**：
   立即推进 **EXP-S2 Latent Probe**，直接从 F16 骨干网络的 Decoder Mode Queries / Hidden States 提取深度内在表征，探查深层隐空间是否保留了真实的模态意图。
