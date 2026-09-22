# EXP-S2V2: Context-Conditioned Latent Probe Synthesis Report

> **Target Repository**: `1inke/trajectory-candidate-selection`  
> **Dataset**: ETH/UCY Official 5 Folds Leave-One-Fold-Out (34,161 samples)  
> **Evaluation Protocol**: Strictly Matched Nested Cross-Validation (Seed 1)  
> **Source Target**: Frozen F16 `JointTrainableV2Decoder` Internal Activations (L3, L4, L5)

---

## 1. 阶段核心科学决议 (Executive Scientific Verdict)

本实验体系全面探查了 F16 生成器内部的上下文条件化潜空间（Context-Conditioned Latents），以实证检验核心假设：**“F16 内部是否已编码了样本条件化的候选优劣语义，只是被 confidence head 损失或未被后验特征（G0~G4）捕获？”**

### 核心结论摘要：
1. **L3 样本条件化确认 (EXP-S2V2A)**：
   - 经方差分解审计，L3 模态潜变量在不同样本间的余弦相似度为 **0.5241**，类内/类间方差比为 **$R = 0.04$**；证实 L3 确为高度样本条件化的动态表征，而非固定的静态模态 Embedding。
2. **全局潜变量探针表现 (EXP-S2V2C)**：
   - 最佳全局配置 **`G0_G1_L5` (Linear Residual)** 达到 **0.5075m Macro ADE**，以仅 86 个参数和 0.127ms 极低延迟，完全对齐并微幅超越了最强全局 Transformer 基线（`transformer_residual_g0124` 0.5077m）；
   - 但所有全局潜变量配置的 Macro ADE 均落在 **[0.507m, 0.518m]** 区间，**未能突破 0.500m 显著收益分水岭**（ADE $\ge 0.505\text{m}$，按指南判定属于 *Latent Gain Weak*）。
3. **局部 Top-K 细粒度判别探针表现 (EXP-S2V2D)**：
   - 在 $K=5$ 局部子池中，引入 L4/L5 后的最佳局部成对判别准确率（Local Pairwise@5cm）仅为 **64.37%**（Native Top-5 基线为 **64.28%**），提升幅度 $< 0.1\%$；
   - 严格落在指南定义的 **Case C 判定区间**（$\text{ADE} \ge 0.505\text{m}$ 且 $\text{Local Pairwise@5cm} \le 66\%$）。
4. **科学定论 (Final Scientific Decision)**：
   - **实证拒绝了“F16 解码器内部隐藏着未被释放的细粒度排序黄金信号”这一猜想**；
   - F16 的内部表征（L3 模态查询、L4 细化网络中间层、L5 预测头前置隐层）主要服务于生成合理多模态轨迹几何分布，其内部并未隐式保留超越后验几何（G1）与输出置信度（G0）的高纯度细粒度质量判别信号；
   - **严格触发停止点协议，在此处正式暂停，不盲目进入 Large Probe 或端到端微调！**

---

## 2. EXP-S2V2A: L3 潜变量方差审计 (L3 Variance Audit)

| 审计指标 | 实测数值 (Macro 5-Fold) | 理论解释 | 状态判定 |
| :--- | :---: | :--- | :---: |
| **Same-Mode Cross-Sample Cosine** | **0.5241** | 远低于静态嵌入阈值 (0.95)，表明不同样本同模态向量存在显著动态变化 | **PASS** |
| **Different-Mode Same-Sample Cosine** | **0.4982** | 模态间几何分离明确，表征具有良好的多模态多样性 | **PASS** |
| **Within-Mode Variance** | **0.2963** | 样本内条件化调制具有足够动态幅度 | **PASS** |
| **Between-Mode Variance** | **0.0098** | 模态中心整体分布相对平衡 | **PASS** |
| **Variance Ratio ($R_{between/within}$)** | **0.04** | 样本间差异远大于模态间固有偏置 | **PASS** |
| **最终审计结论** | - | **L3 为确凿的 Sample-Conditioned 动态特征，具备探针研究合法性** | **VERIFIED** |

---

## 3. EXP-S2V2C: 全局潜变量探针与消融总表 (Global Latent Probe & Ablation)

严格遵循嵌套 5 折交叉验证（Inner 3 折调 $\alpha$，Outer 4 折训 1 折测）：

| 配置名称 (Config) | 模型架构 | 特征构成 | 潜变量源 | ADE (Macro) | FDE (Macro) | Regret (Macro) | Oracle Hit | Pairwise@10 | Collision (Social) | Mean $\alpha$ | 参数量 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`G0_native`** | Baseline | G0 | None | 0.5184m | 1.0757m | 0.3003m | 17.74% | 83.41% | 6.68% | 0.00 | 0 |
| **`G0_L3`** | Linear Residual | G0 | L3 | 0.5160m | 1.0706m | 0.2979m | 19.34% | 83.16% | 6.57% | 0.85 | 70 |
| **`G0_L4`** | Linear Residual | G0 | L4 | 0.5328m | 1.1008m | 0.3147m | 14.14% | 81.29% | 6.24% | 0.07 | 134 |
| **`G0_L4`** | MLP Residual | G0 | L4 | 0.5171m | 1.0672m | 0.2990m | 15.76% | 83.07% | 6.52% | 0.32 | 12929 |
| **`G0_L5`** | Linear Residual | G0 | L5 | 0.5157m | 1.0667m | 0.2976m | 19.57% | 83.20% | 6.74% | 0.73 | 76 |
| **`G0_G1_L4`** | Linear Residual | G0+G1 | L4 | 0.5174m | 1.0754m | 0.2993m | 18.83% | 83.33% | 6.48% | 0.06 | 144 |
| **`G0_G1_L5`** | **Linear Residual** | **G0+G1** | **L5** | **0.5075m** | **1.0505m** | **0.2894m** | **19.37%** | **83.94%** | **6.64%** | **0.30** | **86** |
| **`G0_G1_G4_L5`** | Linear Residual | G0+G1+G4 | L5 | 0.5083m | 1.0530m | 0.2902m | 19.42% | 84.00% | 6.69% | 0.30 | 93 |
| **`L5_only`** | Linear Residual | None | L5 | 0.5285m | 1.0962m | 0.3104m | 14.31% | 82.77% | 6.82% | 0.15 | 71 |
| **`L5_only`** | MLP Residual | None | L5 | 0.5161m | 1.0659m | 0.2979m | 14.08% | 83.25% | 6.29% | 0.28 | 8897 |
| **`G0_G4_L5`** | Linear Residual | G0+G4 | L5 | 0.5157m | 1.0679m | 0.2976m | 19.15% | 83.31% | 7.02% | 0.88 | 83 |

> **深入特征消融分析 (Latent Feature Ablation)**：
> 1. **独立 L5 无法独自支撑排序**：`L5_only` 的 ADE 仅为 0.5161m~0.5285m，Pairwise@10 显著弱于融入原声置信度 G0 的模型，证实 L5 本身缺乏绝对标度校准；
> 2. **L5 与外置几何 G1 的互补性**：只有将 L5 与轨迹外置几何特征（G1 动力学与转角模态）结合时，模型才能达到 0.5075m，说明 L5 主要补充了部分提案尺度的隐式信息；
> 3. **宏观瓶颈依旧**：即便加入 L5，最优结果 0.5075m 仍无法打破 0.505m 瓶颈，未能开辟新数量级。

---

## 4. EXP-S2V2D: 局部 Top-K 细粒度判别探针 (Local Top-K Discrimination Probe)

评估子池规模 $K \in \{3, 5, 10\}$ 下加入 L4/L5 后的细粒度判别指标：

| Config | Subpool $K$ | Model | Latent | ADE (Macro) | Regret (Macro) | Local Pairwise@5cm | Local Pairwise@10cm | Local Spearman | 误干预率 | 有效干预率 | 社交碰撞率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native Top-3 Baseline** | 3 | - | - | 0.5184m | 0.3003m | **58.98%** | 60.34% | 0.1690 | 0.00% | 0.00% | 6.68% |
| `K3_linear_residual_G0_G1_GL_L5` | 3 | Linear | L5 | 0.5143m | 0.2962m | **61.21%** | 62.54% | 0.1806 | 2.71% | 8.92% | 6.73% |
| `K3_tiny_set_attention_G0_GL_L5` | 3 | Attention | L5 | 0.5170m | 0.2989m | 59.79% | 60.82% | 0.1489 | 2.02% | 10.01% | 6.23% |
| **Native Top-5 Baseline** | 5 | - | - | 0.5184m | 0.3003m | **64.28%** | 64.27% | 0.3206 | 0.00% | 0.00% | 6.68% |
| `K5_linear_residual_G0_G1_GL_L5` | 5 | Linear | L5 | **0.5119m** | 0.2938m | **64.37%** | 64.73% | 0.3101 | 2.48% | 9.06% | 6.73% |
| `K5_tiny_set_attention_G0_GL_L4` | 5 | Attention | L4 | 0.5188m | 0.3007m | 62.89% | 62.99% | 0.2850 | 1.76% | 7.78% | 6.53% |
| `K5_tiny_set_attention_G0_G1_GL_L5` | 5 | Attention | L5 | 0.5162m | 0.2981m | 63.44% | 63.57% | 0.2935 | 1.07% | 6.23% | 6.81% |
| **Native Top-10 Baseline** | 10 | - | - | 0.5184m | 0.3003m | **72.55%** | 73.70% | 0.5039 | 0.00% | 0.00% | 6.68% |
| `K10_linear_residual_G0_G1_GL_L5` | 10 | Linear | L5 | **0.5073m** | **0.2892m** | **72.77%** | 73.99% | 0.5143 | 2.01% | 10.11% | 6.82% |
| `K10_tiny_set_attention_G0_GL_L5` | 10 | Attention | L5 | 0.5129m | 0.2948m | 71.70% | 72.71% | 0.4872 | 1.66% | 8.08% | 6.39% |

### 核心判据核验 (Section 35 & 38 Checkpoint)：
- **指南预设阈值**：
  - 若 Local Pairwise@5cm $\ge 70\%$：判定潜变量富含有效细粒度排序信号；
  - 若 Local Pairwise@5cm $\le 66\%$：判定潜变量未提供明显局部区分信息，归入 **Case C**。
- **实测结果**：
  - $K=5$ 最优局部成对判别准确率仅为 **64.37%**（$\le 66\%$），相比 Native Top-5 的 64.28% 几乎无提升；
  - $K=3$ 局部判别率约为 61.21%（基准为 58.98%）；
  - $K=10$ 虽然成对判别率达到 72.77%，但此乃子池扩大自然带来的宏观对比度提高（基准本身即为 72.55%）。

---

## 5. 综合学术判决与后续停工建议 (Final Verdict & Gate Action)

综合 P0 阶段的严密复现性审计（EXP-S6R 全架构误差 $\le 0.5\text{mm}$）、基线对决（EXP-S6G 确认全局 Transformer 0.5077m 坚挺）以及本阶段 S2-v2 上下文潜变量体系探针的完整实测数据：

1. **信息论与表示瓶颈定论 (Representation Limit)**：
   - 无论采用后验特征工程（G0~G4/GL）还是深入到解码器内部提取前置激活（L3/L4/L5），模型对于“头部相似候选轨迹”的细粒度排序能力在当前数据与模型结构下均面临坚固的信息论瓶颈；
   - 当前 F16 的内部表征并没有在隐层遗留被忽视的“黄金排序信号”；
2. **决策树走向 (Case C Triggered)**：
   - 严格满足 **Case C 停工协议**（$\text{Global ADE} \ge 0.505\text{m}$ 且 $\text{Local Pairwise@5cm} \le 66\%$）；
3. **后续行动指引**：
   - **立即停止**在 post-hoc selector 架构搜索、超参数微调或潜变量简单组合上的算力消耗；
   - **不自动进入** Large Capacity Probe，等待人工审阅与后续高层架构规划！
