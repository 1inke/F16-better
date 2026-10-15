# DESIGN: Stage 05B Representation Upgrade (Candidate-History Cross-Attention)

## 1. 科学问题与研究背景
在 Stage 05P 中，简单 GRU 级联 + Set Transformer 探针在 3 个独立 seed 上均未表现出超越 Native Prior 的可排序性信号（38.93% vs 40.23%）。
分析表明：独立单向 GRU 在末尾拼接历史与轨迹，无法实现候选未来每个时步物理状态与过去运动历史之间的精细交互匹配。

根据预注册决策树（`DECISION_TREE.md` 第 25 条）与授权方案：
**只允许进行一次表示升级（One Representation Upgrade）：引入 Candidate-History Cross-Attention 与 Set-Aware Representation，不再同时改变训练 Loss。**

科学问题是：
**通过让候选未来轨迹的每个时间步查询（Queries）与观测历史时间步（Keys/Values）进行跨注意力对齐（Cross-Attention Alignment），能否赋予打分器细粒度时空辨识力，从而在未见过的 Zara Outer 上实现 Top-1 选择率超越 Native Prior（40.23%）并降低选择误差？**

---

## 2. 实验对照设计
- **CONTROL**: 原始 F16 Native Prior（40.23% Top-1 Good Rate, Selected ADE = 0.3513m, J = 2.0271）以及 05P 简单 GRU 探针（38.93% Good Rate）。
- **TREATMENT**: `CrossAttentionScorer`（Multi-Head Cross-Attention: 候选未来轨迹各时步对历史观测序列进行细粒度跨注意力交互，后接 Transformer Set-Encoder 与残差评分头）。
- **HYPOTHESIS**: 细粒度跨注意力对齐使模型能感知候选动态与历史意图的一致性，从而在 Zara Outer 上实现 Top-1 好候选选择率稳定超越 Native Prior（> 40.23%），且不增加 Selected ADE。
- **FALSIFICATION CRITERION**: 若升级后的 Cross-Attention 打分器在 3 个 seed 上依然无法在 Zara Outer 上超越 Native Prior（$\le 40.23\%$）或 3 seed 中正向少于 2 个，则确认打分器路线彻底受阻（`scorer_success = False`）。按照预注册规则，终止后续所有 scorer 路线，将系统锁定为 Generator-Only（Static K24）进入 Stage 06 终审内部验证。

---

## 3. 严格冻结与审计保证
1. **生成端完全冻结**：Static K24 生成候选与对应动力学状态保持冻结。
2. **损失函数严格单一变量**：完全保持与 05P 一致的温度质量交叉熵损失（$\tau=0.25$），不换 ranking loss。
3. **数据隔离与零泄漏**：训练仅在 `grouped_train_index` ($N=16,337$) 上进行，Zara Outer ($N=1,561$) 与 Inner CAL ($N=1,773$) 严格隔离。
4. **决策规则**：
   - 若 `scorer_success = True` $\rightarrow$ `06_INTERNAL_VALIDATION`（选择该 Scorer）。
   - 若 `scorer_success = False` $\rightarrow$ `06_INTERNAL_VALIDATION`（终止 Scorer，选定 `selected_scorer = None`，验证生成端独立结论）。
