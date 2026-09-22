# EXP-S6G: Strongest Global vs Local Baseline Re-comparison

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples)  
> **Evaluation Protocol**: Strictly Matched Nested Cross-Validation (Seed 1)  
> **Primary Safety Metric**: `Collision_Social` (on social-eligible scenarios)

---

## 1. 全局与局部强基准对决总表 (Head-to-Head Comparison)

| Method | ADE (Macro) | FDE (Macro) | Regret (m) | NearOracle@5cm | Collision (Social) | Interv. Rate | False Interv. | Latency (ms) | Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native (Frozen F16 Top-1)** | **0.5184m** | 1.0757m | 0.3003m | 34.92% | 6.68% | 0.00% | 0.00% | 0.000 | 0 |
| **Global Linear Residual (G0-G2)** | **0.5135m** | 1.0640m | 0.2954m | 35.07% | 6.64% | 11.25% | 1.15% | 0.158 | 23 |
| **Global Transformer Residual (G0-G4) [Strongest Global]** | **0.5077m** | 1.0543m | 0.2895m | 35.82% | 6.31% | 20.84% | 2.50% | 4.620 | 52865 |
| **Local Linear Residual (K=3, G0-G4+GL) [Best Deterministic]** | **0.5104m** | 1.0601m | 0.2923m | 35.09% | 6.71% | 16.50% | 3.03% | 0.016 | 39 |
| **Local Tiny Set Attention (K=5, G0-G4+GL) [Best Local Neural]** | **0.5088m** | 1.0560m | 0.2907m | 35.51% | 6.71% | 18.20% | 2.68% | 4.532 | 36033 |
| **Gate + Strongest Global (D3 @ 20% Budget)** | **0.5119m** | 1.0612m | 0.2938m | 35.40% | 6.03% | 9.41% | 0.57% | 0.120 | 54365 |
| **Gate + Best Local (D4 @ 20% Budget)** | **0.5127m** | 1.0628m | 0.2946m | 35.60% | 6.01% | 9.91% | 0.53% | 0.100 | 37533 |

---

## 2. 核心科学研判与架构决议 (Executive Scientific Verdict)

1. **全局最强基准地位坚固 (Global Frontier Preserved)**：
   - 当前在 ETH/UCY 盲测上最强 Macro ADE 仍为 **`Global Transformer Residual` (ADE = 0.5077m)**；
   - 局部重排最优配置（`K5_tiny_set_attention` ADE = **0.5088m**）与最佳确定性线性模型（`K3_linear_residual` ADE = **0.5104m**）虽然极具竞争力且延迟低至 0.016ms，但在宏观精度上**尚未实质性超越全局 Transformer 基准**。
2. **门控联合介入 (Gate Integration) 的防御价值**：
   - 门控局部（D4 @ 20% Budget）与门控全局（D3 @ 20% Budget）将误干预率分别压低至 **0.53%** 与 **0.57%**（降低超 75%），并将社交碰撞率控制在 6.01%~6.03%；
   - 两者在紧凑预算下互有胜负，局部重排未现绝对支配地位。
3. **S6 系列阶段性封板结论 (Post-Hoc Feature Plateau)**：
   - 基于现有后验外置几何、运动与风险特征族（G0/G1/G2/G4/GL），无论通过子池截断、加权训练还是软目标平滑，Macro ADE 均收敛于 **0.508m ~ 0.510m** 平台；
   - 正式暂停基于现有特征工程的 Local Reranker 细微调参，全面开启 **Phase 3 第二阶段：S2-v2 Context-Conditioned Latent 深入探针**！
