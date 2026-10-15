# Stage 05P (Candidate-Aware Rankability Probe) Result Analysis

## 1. 实验目标与科学设定
- **CONTROL**: 原始 F16 Native Prior（仅依据模态分类器先验选择 Top-1 候选）。
- **TREATMENT**: `CandidateAwareScorer` 轻量可排序性探针（结合观测历史 GRU、候选未来轨迹 GRU、State32 动力学特征、机动类型 Embedding、Transformer Set Encoder 集合交互以及 Native Logits Prior 残差）。
- **HYPOTHESIS**: 在静态冻结 Static K24 候选集合（Oracle 覆盖率 67.58%）的前提下，候选感知表示能够在未见过的 Zara Outer 上展现显著优于 Native Prior（40.23%）的选择增益，为全量打分器训练提供表征依据。
- **FALSIFICATION CRITERION**: 若新表示未能在至少 2/3 seeds 上实现 Top-1 选择率超越 Native Prior，或 Pairwise Accuracy 相比基线未现优势，则证伪“当前简单 GRU+SetTransformer 级联表示已具备充分可排序性信号”（判定 `rankability_signal = False`）。

---

## 2. 核心实验对比结果

| 指标 | Native Prior 基线 | 探针 Seed 1 | 探针 Seed 2 | 探针 Seed 3 | 探针 3-Seed 均值 | 相对基线增量 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zara Outer Top-1 Good Rate** | **40.23%** (628/1561) | 38.69% | 38.95% | 39.14% | **38.93%** | **-1.30 pp** |
| **Zara Outer Pairwise Acc@0.10m** | **82.90%** | 81.35% | 81.48% | 81.44% | **81.42%** | **-1.47 pp** |
| **Zara Outer Spearman Rho** | **0.7505** | 0.7208 | 0.7215 | 0.7212 | **0.7212** | **-0.0293** |
| **Selected ADE (m)** | **0.3513** | 0.3668 | 0.3659 | 0.3656 | **0.3661** | **+0.0148m (恶化)** |
| **Selected J** | **2.0271** | 2.1192 | 2.1154 | 2.1146 | **2.1164** | **+0.0893 (恶化)** |
| **Positive Seeds (> Base)** | - | 0 | 0 | 0 | **0 / 3** | - |

---

## 3. 统计观察与科学解读

### OBSERVATION
1. **全 Seed 均无法超越 Native Prior**：所有 3 个独立 seed 的 Top-1 Good Rate（38.69%~39.14%）均低于原始 Native Prior（40.23%），平均净下降 **-1.30 pp**；
2. **候选间相对判别能力下降**：在 ADE 差异 $\ge 0.10$ m 的候选对排序准确率上，探针为 81.42%，相比 Native Prior（82.90%）下降 **-1.47 pp**，Spearman 秩相关系数由 0.7505 下降至 0.7212；
3. **Selected 误差指标恶化**：Top-1 选出轨迹的 ADE 从 0.3513m 劣化至 0.3661m，J 从 2.0271 恶化至 2.1164。

### INTERPRETATION
- 独立单向的 GRU 编码器（独立处理 past 与 trajectory）结合浅层 Set Transformer，缺乏**历史观测细粒度时空状态与未来候选轨迹之间的跨注意力对齐（Cross-Attention Alignment）**；
- 简单将历史向量与候选向量在最后拼接，无法让模型在每一步物理时序上比对“该候选加速度/转向是否符合历史行人意图”，导致训练时拟合了训练集局部伪相关性，在 Zara Outer 上泛化反而扰动了原本稳健的 Native 先验。

### NOT PROVEN
- 不能证明“候选感知打分器在理论上不可行”；
- 但已严格确证“无细粒度 Cross-Attention 的浅层级联表示无法提供足以战胜原生先验的可排序性信号”。

---

## 4. 决策流向

根据预注册决策树（`DECISION_TREE.md` 第 25 条）与状态机规则：
- **`rankability_signal = False`**；
- 禁止盲目尝试调参或换 ranking loss；
- 按照状态机预设，系统直接流向：
  $$\textbf{05B\_REPRESENTATION}$$
  （即进入候选-历史 Cross-Attention / 细粒度时空交互表示升级阶段）。
