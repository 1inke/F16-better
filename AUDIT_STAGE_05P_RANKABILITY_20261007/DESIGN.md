# DESIGN: Stage 05P Candidate-Aware Rankability Probe

## 1. 科学问题与研究背景
在 Stage 04A/04B 中，我们严格确认了 Static K24（Native 20 + Top-4 Maneuver Proposals）为可靠的候选生成端（Oracle 覆盖率 67.58%，保持 Lost=0）。
然而，目前系统的选择端仍然受制于 F16 原始分类器：
- 原始 Native Prior 仅在 Native 20 条候选上产生 logits，对 4 条显式提议无法直接打分；
- 原始 Native Prior 在 Zara Outer 上的 Top-1 真实好候选选择率仅为 **40.23%**（Selected ADE = 0.3513m, J = 2.027），存在高达 27.35 pp 的选择落差；
- 既往实验（EXP-S1A）表明，仅依赖低阶表面几何特征（G0~G4）存在严重的 `REPRESENTATION_BOTTLENECK`。

本阶段（05P）的科学问题是：
**整合了观测历史（GRU）、候选未来轨迹序列（GRU）、运动学物理状态（State32）、候选来源与机动类型嵌入（Provenance/Type Embedding）、全集合自注意力交互（Transformer Set Encoder）以及原始原生对数先验（Native Logits Prior）的 Candidate-Aware Representation，是否具备显著的可排序性信号（Rankability Signal）？**

---

## 2. 实验对照设计
- **CONTROL**: 原始 F16 Native Logits Prior（仅基于历史与原生模态分类，argmax 选择 Top-1）。
- **TREATMENT**: `CandidateAwareScorer` 轻量探针（冻结 Static K24 生成端，训练 3 个独立 seed：1, 2, 3）。
- **HYPOTHESIS**: 新候选感知表示能够显著提升候选对之间排序的相对准确率（Pairwise Accuracy > 60%），并在未见过的 Zara Outer 上使 Top-1 好候选选择率超越 Native Prior（> 40.23%），为大模型训练提供充分表征支持。
- **FALSIFICATION CRITERION**: 若新表示在训练域内或 Zara Outer 上的 Pairwise Accuracy $\le 55\%$，或 Top-1 选择率无法超越 Native Prior，则判定新表示依然存在瓶颈（`rankability_signal = False`），系统将流向 `05B_REPRESENTATION`（进行 Cross-Attention 结构升级）。

---

## 3. 严格冻结与审计保证
1. **生成端完全冻结**：Static K24 候选序列在整个探针训练与评估期间严格不变。
2. **零未来信息泄漏**：除训练监督使用的 GT Future 误差目标外，模型前向输入仅包含过去观测历史与候选轨迹本身。
3. **隔离评估**：训练仅在 `grouped_train_index` ($N=16,337$) 上进行，Zara Outer ($N=1,561$) 与 Inner CAL ($N=1,773$) 严格隔离。
4. **决策判定规则**：`rankability_signal = True` 当且仅当 Zara Outer 上 Top-1 Good Rate 相对 Native Prior 取得正向增益且 Pairwise Accuracy $\ge 0.60$。
