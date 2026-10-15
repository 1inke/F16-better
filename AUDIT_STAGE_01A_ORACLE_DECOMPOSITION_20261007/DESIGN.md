# DESIGN: Stage 01A Motion Program Oracle Decomposition

## 1. 科学问题与研究背景
在既有研究中，Static K24（Native 20 + Top-4 Explicit Maneuver Proposals）成功将 Zara Outer 覆盖率从 66.29% 提升至 67.58%，并保持 $Lost \equiv 0$。然而，静态提议使用的是离散默认参数（固定转向角与固定加减速度曲线），无法根据具体历史自适应调节速度与曲率。

Stage 01A 是一项**零部署模型训练**的严格 Oracle 分解实验，回答以下核心科学问题：
1. **连续参数天花板（Parameter Ceiling）**：若保持当前选择的 Top-4 机动类别不变，仅通过真实 GT 优化其连续参数（s1, s2, s3, theta1, theta2, theta3, onset, persistence），能带来多大收益？
2. **类别选择天花板（Class Selection Ceiling）**：若允许从全部 8 类机动中用 GT 选择最优 4 类并优化其参数，能带来多大收益？
3. **分类学与自由程序天花板（Taxonomy Ceiling）**：若彻底摆脱预设的 8 类离散机动语义约束，直接拟合 4 条自由连续 Motion Programs，能带来多大收益？

如果自由连续 Motion Program 相比 Static K24 的综合指标 J 改善不足 3% 且覆盖率增益不足 1 pp，则说明该运动程序体系缺乏足够理论上限，实验将安全停机；若上限充足，则根据参数缺口、类别缺口和分类学缺口的占比，决定后续进入 02P、02C 或 01B。

---

## 2. 实验对照组设计（统一 K=24 预算）
- **S0 (Static-K24)**: Native 20 + 现有静态 Top-4 显式提议。
- **S1 (Top4-OracleParam-K24)**: Native 20 + 与 S0 完全相同的 Top-4 类别，其 8 维连续参数由 GT 离线优化。
- **S2 (AllClass-OracleParam-K24)**: Native 20 + 从全部机动类别中由 GT 选出最优 4 类，且连续参数由 GT 离线优化。
- **S3 (Free4-OracleProgram-K24)**: Native 20 + 4 条自由初始化的连续 Motion Programs，连续参数完全不受离散类别先验约束由 GT 离线优化。

---

## 3. 指标与分解公式
基于连续联合误差指标 $J = \min_k \max(ADE_k / 0.2, FDE_k / 0.4)$：
- $G_{\text{total}} = J(S0) - J(S3)$
- $G_{\text{param}} = J(S0) - J(S1)$
- $G_{\text{class}} = J(S1) - J(S2)$
- $G_{\text{tax}} = J(S2) - J(S3)$
- 分解比例：
  $$\text{parameter\_gap\_share} = G_{\text{param}} / G_{\text{total}}$$
  $$\text{class\_gap\_share} = G_{\text{class}} / G_{\text{total}}$$
  $$\text{taxonomy\_gap\_share} = G_{\text{tax}} / G_{\text{total}}$$

---

## 4. 严格审计与安全保证
1. **零部署 forward 泄漏**：Oracle 拟合仅用于离线评估与建立理论天花板，所得参数仅存为监督/诊断缓存，绝不写入部署模型；
2. **Native 20 严格冻结**：所有评估中 Native 20 保持绝对一致；
3. **分层分群报告**：必须全面报告 overall、stable、transition、change、clear-event 以及时序 prefix 误差（t=4, 8, 12）。
