# Stage 04B2 (Scene Balanced Training) Result Analysis

## 1. 实验目标与科学设定
- **CONTROL**: Stage 04A Best Model (69.64% Zara outer coverage) 及 Static K24 (67.58% coverage)。
- **TREATMENT**: 在 04A Proposal-Conditioned Correction Head 架构不变的前提下，使用场景反频率加权采样（`WeightedRandomSampler`），平衡 Hotel ($N=759$) 与 University ($N=15,578$) 在每个训练 batch 中的出现概率。
- **HYPOTHESIS**: 数据集场景失衡导致模型偏向 University 动态特征，平衡采样可缓解场景支持度偏移并提升 Zara outer 泛化表现。
- **FALSIFICATION CRITERION**: 若 Zara outer coverage 相比 04A 没有提升（gain $\le 0$），或相比 Static K24 出现恶化，则认为场景再平衡不能解决残差漂移问题。

---

## 2. 实验结果

| 模型 / 配置 | Zara Outer Complete Coverage | minADE (m) | minFDE (m) | Mean J | Effective Clusters | Lost vs Static | Rescued vs Static |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static K24** | 1055 (67.58%) | 0.1406 | 0.2462 | 0.7953 | 1.1019 | - | - |
| **04A Best (Seed 3)** | 1087 (69.64%) | 0.1442 | 0.2464 | 0.8122 | 1.3413 | 82 | 114 |
| **04B2 Seed 1** | 1038 (66.50%) | 0.1481 | 0.2541 | 0.8361 | 1.3004 | 139 | 122 |
| **04B2 Seed 2** | 1018 (65.21%) | 0.1488 | 0.2558 | 0.8391 | 1.3671 | 153 | 116 |
| **04B2 Seed 3** | 1037 (66.43%) | 0.1477 | 0.2497 | 0.8333 | 1.3421 | 141 | 123 |
| **04B2 Mean** | **1031.0 (66.05%)** | 0.1482 | 0.2532 | 0.8362 | 1.3365 | 144.3 | 120.3 |

---

## 3. 统计观察与科学解读

### OBSERVATION
1. **Zara Outer 表现全面下降**：04B2 三个 seed 的 coverage 分别为 66.50%、65.21%、66.43%，均值仅为 66.05%。不仅比 04A 下降 **-3.59 pp**，甚至比未经任何训练的 **Static K24 (67.58%)** 还低 **-1.54 pp**。
2. **严重 Over-Editing（误伤加剧）**：Lost vs Static 从 04A 的 82 条飙升至 139~153 条（均值 144.3 条），且救援能力并未显著提升（120 条 rescued 无法弥补 144 条 lost）。
3. **运动学校验通过**：速度 p95、加速度 p95、加加速度 p95 均未发生物理越界，证明下降并非由于数值发散或动力学生成崩溃，而是由于预测几何偏差。

### INTERPRETATION
- 强行对低频场景 Hotel 进行高达 20 倍的重采样（从 4.6% 强拉到 50% 采样比率），破坏了模型在主导场景上的平滑动态分布估计。
- 模型在稀疏场景噪声上的过拟合导致生成的轨迹修正残差更加不稳定，进一步放大了对原生高质量候选的破坏（Over-editing）。

### NOT PROVEN
- 不能证明“数据增强或采样调节在所有场景预测模型中均无效”；
- 但已严格证伪“在保持简单残差头的前提下，通过场景重平衡能挽救 learned correction 的跨场景漂移”。

---

## 4. 决策流向

根据预注注册分支规则（`DECISION_TREE.md` 第 15-18 条）及 `decision_engine.py`：
- **04B2 gain $\le 0$**；
- 预注册授权的所有 learned correction 分支（04A, 04B1, 04B2）均已探索穷尽；
- **正式终止 learned correction 路线**，不再尝试任何新的 head / loss / sampling 变体；
- 状态机自动锁定：
  $$\text{selected\_generator} = \text{"STATIC\_K24"}$$
- 流向下一研究阶段：**05P_RANKABILITY**（候选感知打分器可排序性探针检验）。
