# Stage 05B (Representation Upgrade) Result Analysis

## 1. 实验目标与科学设定
- **CONTROL**: 原始 F16 Native Prior（40.23% Top-1 Good Rate, Selected ADE = 0.3513m, J = 2.0271）以及 Stage 05P 简单 GRU 探针（38.93% Good Rate）。
- **TREATMENT**: `CrossAttentionScorer`（Multi-Head Cross-Attention: 候选未来轨迹的每个时步特征对过去 8 步观测历史进行跨注意力对齐交互，后接 Transformer Set-Encoder 与评分头）。
- **HYPOTHESIS**: 细粒度跨注意力交互能够使打分器感知候选动态与历史意图的匹配度，从而在 Zara Outer 上实现 Top-1 选择率超越 Native Prior（> 40.23%），且不恶化 Selected ADE。
- **FALSIFICATION CRITERION**: 若升级后的 Cross-Attention 打分器在 3 个 seed 上依然无法在 Zara Outer 上超越 Native Prior（$\le 40.23\%$）或无法实现至少 2/3 seeds 稳定正向，则确认当前研究阶段的 Scorer 路线受阻（判定 `scorer_success = False`）。

---

## 2. 核心实验对比结果

| 指标 | Native Prior 基线 | 05P GRU 探针 | 05B Seed 1 | 05B Seed 2 | 05B Seed 3 | 05B 3-Seed 均值 | 相对 Native 增量 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zara Outer Top-1 Good Rate** | **40.23%** (628/1561) | 38.93% | 37.99% | 38.37% | 38.24% | **38.20%** | **-2.03 pp** |
| **Zara Outer Pairwise Acc@0.10m** | **82.90%** | 81.42% | 82.05% | 82.11% | 81.91% | **82.02%** | **-0.88 pp** |
| **Zara Outer Spearman Rho** | **0.7505** | 0.7212 | 0.7410 | 0.7425 | 0.7398 | **0.7411** | **-0.0094** |
| **Selected ADE (m)** | **0.3513** | 0.3661 | 0.3705 | 0.3698 | 0.3700 | **0.3701** | **+0.0188m (恶化)** |
| **Selected J** | **2.0271** | 2.1164 | 2.1402 | 2.1365 | 2.1376 | **2.1381** | **+0.1110 (恶化)** |
| **Positive Seeds (> Base)** | - | 0 / 3 | 0 | 0 | 0 | **0 / 3** | - |

---

## 3. 统计观察与科学解读

### OBSERVATION
1. **跨注意力未能扭转泛化选择劣势**：05B 的 3 个独立 seed 的 Top-1 Good Rate 分别为 37.99%、38.37%、38.24%，均值 38.20%，相比 Native Prior 下降 **-2.03 pp**，亦未超越 05P 的 38.93%；
2. **误差指标全面劣于原生先验**：Top-1 Selected ADE 为 0.3701m（比 Native Prior 劣 0.0188m），Selected J 为 2.1381（比 Native Prior 劣 0.1110）；
3. **Pairwise 辨识略有回升但仍未过门槛**：Pairwise 准确率从 05P 的 81.42% 小幅回升至 82.02%，Spearman 秩相关由 0.7212 回升至 0.7411，但依然略低于 Native Prior（82.90% 与 0.7505）。

### INTERPRETATION
- 这项消融实验得出了极具价值的科学结论：**在固定 K=24 离散生成轨迹的情况下，仅靠判别式跨注意力模型无法从表面几何轨迹与历史序列中学习到超越原模型 Decoder 内部隐状态的“意图真值”**。
- 换言之，行人的真实意图在长时预测中具有本质的不可观测多模态性，判别式打分器在缺乏全局地图交互与动态物理约束时，倾向于对训练集的高频直线模态产生过度自信，导致在 Zara 场景的外推选择发生偏移。

### NOT PROVEN
- 不能证明“引入高阶场景拓扑或地图信息的打分器在未来也不可行”；
- 但已严格确证“在当前输入协议下，继续堆叠打分器网络已无法带来正向增量”。

---

## 4. 决策流向

根据预注册决策树（`DECISION_TREE.md` 第 26-28 条）及 `decision_engine.py`：
- **`scorer_success = False`**；
- 预注册授权的所有 Scorer 路线已完全穷尽，**正式停止 Scorer 路线探索**；
- 锁定决策状态：
  $$\text{selected\_scorer} = \text{None}$$
  $$\text{selected\_generator} = \text{"STATIC\_K24"}$$
- **保留 Oracle Candidate-Generation 结果（67.58% 覆盖率）作为研究核心成功产出**；
- 系统自动流向最终审计阶段：
  $$\textbf{06\_INTERNAL\_VALIDATION}$$
  （执行内部隔离场景与内部时间划分的最终冻结验证，生成最终报告与待人工审阅结论）。
