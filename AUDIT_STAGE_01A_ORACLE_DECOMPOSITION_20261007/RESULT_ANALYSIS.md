# Stage 01A (Motion Program Oracle Decomposition) Result Analysis

## 1. 实验目标与科学设定
- **科学问题**: 现有的 Static K24 方案使用固定经验参数（例如 35°/75° 转向、固定减速曲线）。若使用可微连续 Motion Program，其理论性能上限有多高？更重要的是：**性能瓶颈究竟来自于连续参数不够准（Parameter Gap）、离散分类选择不够准（Class Gap），还是当前的 8 类机动分类学本身不完备（Taxonomy Gap）？**
- **对照实验设置（统一 K=24 预算，Native 20 严格冻结）**:
  - **S0 (Static-K24)**: Native 20 + 现有静态 Top-4 显式提议。
  - **S1 (Top4-OracleParam-K24)**: Native 20 + 与 S0 完全一致的 4 个机动类别，其 8 维连续参数经 GT 离线最优化。
  - **S2 (AllClass-OracleParam-K24)**: Native 20 + 从全部 8 类合法机动中经 GT 选出最优 4 类并参数最优化。
  - **S3 (Free4-OracleProgram-K24)**: Native 20 + 4 条自由初始化的连续 Motion Programs，无离散类别约束由 GT 最优化。
- **评测数据集**: Primary Zara Outer 隔离集（$N=1,561$）。

---

## 2. 核心实验对比结果

### 2.1 全局指标阶梯对比
| 候选集配置 | Complete Coverage (好候选覆盖率) | minADE (m) | minFDE (m) | Mean J | Early Err (t=4) | Mid Err (t=8) | Late Err (t=12) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native 20 基线** | 66.62% (1,040 / 1,561) | 0.1415 | 0.2464 | 0.8151 | 0.0818 | 0.1257 | 0.2170 |
| **S0 Static-K24** | 67.97% (1,061 / 1,561) | 0.1398 | 0.2452 | 0.7982 | 0.0817 | 0.1245 | 0.2132 |
| **S1 Top4-OracleParam** | **93.21% (1,455 / 1,561)** | **0.0841** | **0.1384** | **0.3372** | **0.0573** | **0.0768** | **0.1183** |
| **S2 AllClass-OracleParam** | **94.43% (1,474 / 1,561)** | **0.0792** | **0.1287** | **0.3067** | **0.0547** | **0.0724** | **0.1106** |
| **S3 Free4-OracleProgram** | **95.45% (1,490 / 1,561)** | **0.0664** | **0.1044** | **0.2227** | **0.0487** | **0.0617** | **0.0890** |

---

### 2.2 缺口分解（Gap Decomposition）

根据预注册公式：
- $G_{\text{total}} = J(S0) - J(S3) = 0.7982 - 0.2227 = 0.5754$
- $G_{\text{param}} = J(S0) - J(S1) = 0.7982 - 0.3372 = 0.4610$
- $G_{\text{class}} = J(S1) - J(S2) = 0.3372 - 0.3067 = 0.0305$
- $G_{\text{tax}} = J(S2) - J(S3) = 0.3067 - 0.2227 = 0.0840$

**核心分解比例**:
1. **Total Oracle J Gain Fraction**: **72.09%**（门槛 $\ge 3\%$，**极大幅度通过**）；
2. **Free4 Coverage Gain vs Static**: **+27.48 pp**（净救回 429 个窗口，门槛 $\ge 1.0\text{ pp}$，**极大幅度通过**）；
3. **Parameter Gap Share**: **80.11%**（门槛 $\ge 20\%$，**占绝对主导地位**）；
4. **Class Gap Share**: **5.30%**（门槛 $< 20\%$）；
5. **Taxonomy Gap Share**: **14.59%**（门槛 $< 30\%$）。

---

### 2.3 分群与动力学详细表现
- **Change 困难组表现**:
  - Native 20: 覆盖率 44.25% (200/452)
  - S0 Static: 覆盖率 48.01% (217/452)
  - S1 Top4-Oracle: 覆盖率 **89.60% (405/452)**
  - S3 Free4-Oracle: 覆盖率 **93.81% (424/452)**
  困难组覆盖率翻倍，证实连续动力学参数对大机动行人的巨大拟合潜力。
- **物理运动学合规性**:
  - S1: 速度 p95 = 1.34 m/s，加速度 p95 = 0.52 m/s²，加加速度 p95 = 2.45 m/s³
  - 全部参数均严格位于行人动力学合理范围内，未出现跳变或高频震荡。

---

## 3. 统计观察与科学解读

### OBSERVATION
1. **连续 Motion Program 拥有巨大理论上限**：S3 使整体好候选覆盖率从 67.97% 飙升至 95.45%，平均 J 降低 72.09%；
2. **80% 的收益仅靠参数连续化即可达成**：S1（完全不改变分类器选定的 4 个类别，只做参数微调）就将覆盖率拉到了 **93.21%**（占据了总收益的 80.11%）；
3. **类别与分类学扩展收益非常有限**：S2 相比 S1 仅提升 1.22 pp（贡献 5.30%），S3 相比 S2 仅提升 1.02 pp（贡献 14.59%）。

### INTERPRETATION
- 这项 Oracle 分解给出了非常明确的研究定论：**现有的 8 类机动语义空间（Stop, Decel, Accel, Mild Left/Right, Sharp Left/Right, Reverse）在拓扑上已经高度完备，根本不需要重写或复杂化 Taxonomy（否定 01B 分支）！**
- 同样，离散分类器的准确率也不是瓶颈（否定 02C 分支）。
- **真正的核心瓶颈在于：现有系统使用的是固定死板的几何形状，缺乏从历史观测中预测连续速度比率（s1, s2, s3）、累计转向角度（theta1, theta2, theta3）以及变化发生时刻（onset）的能力。**

---

## 4. 决策流向

根据预注册状态机规则（`decision_engine.py` 第 38 行）：
- `total_oracle_J_gain_fraction = 72.09% >= 0.03`（通过，排除 STOP_NO_MOTION_PROGRAM_CEILING）；
- `taxonomy_gap_share = 14.59% < 0.30`（跳过 01B_TAXONOMY_AUDIT）；
- `parameter_gap_share = 80.11% >= 0.20`（**触发 02P_PARAMETER_IDENTIFIABILITY**）；
- `class_gap_share = 5.30% < 0.20`（跳过 02C_CLASS_IDENTIFIABILITY）；
- 状态机下一阶段唯一裁决：
  $$\textbf{02P\_PARAMETER\_IDENTIFIABILITY}$$
  （即全力攻关从过去观测中预测连续运动程序参数的可辨识性与表征学习）。
