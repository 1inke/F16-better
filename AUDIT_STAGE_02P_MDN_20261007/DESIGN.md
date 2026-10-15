# DESIGN: Stage 02P-MDN Probabilistic Motion Program

## 1. 科学问题与背景
先前完成的 `MOTION_PROGRAM_TARGET_IDENTIFIABILITY_AUDIT` 给出了决定性的实证结论：
- **实验 A (参数标签简并性)**: 简并率仅 **3.79%**，证明 Oracle 8 维参数在物理上高度唯一且可识别；
- **实验 B (条件未来多模态性)**: 在相似过去表征（余弦相似度 0.885）的条件下，**77.25%** 的 Change 难例呈现强烈的未来角向双峰/多峰分叉（转角跨度高达 111°，涵盖平均 4 种独立机动类别）。

这严谨证明了 Stage 03 中确定性单点预测在 Change 组退化的真正根因是**条件分布的均值平滑陷阱（Conditional Mean Smoothing）**。
因此，本阶段根据用户授权开启预注册的 **`Stage 02P_MDN`** 概率分支：
**能否使用轻量 3-component 对角高斯混合密度网络（MDN）建模 $p(z \mid \text{history}, \text{class})$，并在严格保持 $K=24$（Native 20 冻结 + 4 个提案）预算的前提下，通过联合模式分配器保留极端突变机动，从而在跨协议测试中击败 Static-K24？**

---

## 2. 实验对照设计
- **CONTROL**: `native20 + Static Top4 proposals`（$K=24$）；
- **TREATMENT**: `native20 + Top-4 MDN joint mode proposals`（$K=24$）；
- **MDN 预测器架构**:
  - 输入：P2 编码器时空向量（96 维）+ 8 维动作类别 One-Hot；
  - 骨干网络：`Linear(104, 128) -> GELU -> Linear(128, 128) -> GELU`；
  - 三重输出头：
    1. 混合权重 $\pi \in \mathbb{R}^3$（Softmax 归一化）；
    2. 分量均值 $\mu \in \mathbb{R}^{3 \times 8}$；
    3. 分量对角对数方差 $\log \sigma \in \mathbb{R}^{3 \times 8}$。
- **训练目标**:
  直接最小化连续参数空间负对数似然（Negative Log-Likelihood, NLL）：
  $$\mathcal{L}_{\text{NLL}} = -\log \sum_{m=1}^3 \pi_m \mathcal{N}(z_{\text{oracle}} \mid \mu_m, \text{diag}(\sigma_m^2))$$
- **联合模式分配机制（Joint Mode Allocation under $K=24$）**:
  - 全局候选空间：8 个机动类别 $\times$ 3 个高斯模式 = 24 个候选连续运动程序；
  - 每个模式的联合置信度：
    $$\text{Score}(c, m) = P(c \mid \text{history}) \times \pi_{c, m}$$
  - 分配规则：
    按联合得分降序排列，挑选 Top-4 候选；
    **多样性硬约束：单个动作类别最多占据 2 个 slots**（允许分配如 `left-sharp`, `left-mild`, `decelerate`, `continue`），从而保留极端极点而不挤占其他关键机动。

---

## 3. 门禁与评判准则
- 必须分别在三大隔离协议上评估：
  1. `target-track` ($N=1,035$)
  2. `recording-time` ($N=566$)
  3. `scene-family` ($N=1,561$)
- **晋级准则 (`probabilistic_program_beats_static_in_2of3`)**:
  在至少 2/3 的隔离协议中，重构候选的覆盖率与 J 指标优于 Static-K24。
- 若通过，晋级至 Stage 03 3-Seed 训练；若未通过，则按预注册安全停机 `STOP_PARAMETER_NOT_IDENTIFIABLE_FROM_PAST`。
