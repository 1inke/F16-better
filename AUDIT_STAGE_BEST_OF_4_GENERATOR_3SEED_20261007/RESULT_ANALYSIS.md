# RESULT ANALYSIS: STAGE B BEST-OF-4 GENERATOR 3-SEED FORMAL VALIDATION

## 1. Executive Summary & Preregistered Decision Rule Evaluation
- **Stage**: `STAGE_B_BEST_OF_4_GENERATOR_3SEED` (Seeds 42, 43, 44)
- **Status**: `STOP_SEED_SPECIFIC`
- **Preregistered Gate Evaluation**:
  1. *Change Coverage $\ge 2/3$ seeds positive*: **FAIL (1/3 seeds positive)**.
     - Seed 42: **+0.47 pp** (41.71% $\to$ 42.18%)
     - Seed 43: **+0.00 pp** (41.71% $\to$ 41.71%)
     - Seed 44: **-0.47 pp** (41.71% $\to$ 41.23%)
  2. *3-Seed Mean Change Coverage gain $\ge +1.0$ pp*: **FAIL (+0.00 pp)**.
  3. *Overall minADE/minFDE harm $\le 1.0\%$*: **PASS** (minADE +1.78% mean gain, minFDE +3.19% mean gain, 0 harm across all seeds).
  4. *Slot Collapse ruled out*: **PASS** (Mean pairwise trajectory distance 0.622 m, max dominance 37.4%, entropy 1.93 bits / 2.0 max).
  5. *Motion Guards ($\le 2.0\%$)*: **FAIL** (Accel p95 diffs: Seed 42 +2.59%, Seed 43 +1.83%, Seed 44 +2.93%).
- **Preregistered Action Triggered**:
  > “若仅 Seed42 为正，输出 `STOP_SEED_SPECIFIC`。” $\implies$ **`STOP_SEED_SPECIFIC`**.

---

## 2. Quantitative Performance Table Across 3 Paired Seeds

### (1) Overall Performance (Outer Set $N=1561$)
| Model / Seed | Coverage (%) | Covered / 1561 | $\Delta\text{Cov}$ (pp) | Rescue | Lost | Net Rescue | Mean $J$ | $\min\text{ADE}$ (m) | $\min\text{FDE}$ (m) | Eff. Clusters | Motion Guard |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Control (Static-K24)** | 67.97% | 1061 | - | - | - | - | 0.7985 | 0.1412 | 0.2475 | 1834 | - |
| Bo4 Seed 42 | 69.44% | 1084 | +1.47 | 40 | 17 | +23 | 0.7816 (+2.11%) | 0.1387 (+1.73%) | 0.2387 (+3.58%) | 1813 | False (+2.59% acc) |
| Bo4 Seed 43 | 69.63% | 1087 | +1.67 | 43 | 17 | +26 | 0.7812 (+2.16%) | 0.1381 (+2.16%) | 0.2413 (+2.53%) | 1810 | True (+1.83% acc) |
| Bo4 Seed 44 | 68.99% | 1077 | +1.02 | 33 | 17 | +16 | 0.7851 (+1.68%) | 0.1391 (+1.44%) | 0.2390 (+3.45%) | 1798 | False (+2.93% acc) |
| **3-Seed Mean** | **69.35%** | **1082.7** | **+1.39** | **38.7** | **17.0** | **+21.7** | **0.7826 (+1.98%)** | **0.1386 (+1.78%)** | **0.2396 (+3.19%)** | **1807.0** | **1/3 pass** |

### (2) Subgroup Breakdown (Mean Across 3 Seeds)
| Subgroup | Sample Count $N$ | Control Coverage | 3-Seed Mean Bo4 Cov | Net Cov Gain | Control $\min\text{ADE}$ | 3-Seed Mean Bo4 ADE | Net ADE Gain | Control $\min\text{FDE}$ | 3-Seed Mean Bo4 FDE | Net FDE Gain |
|---|---|---|---|---|---|---|---|---|---|---|
| **Stable** | 1226 | 74.47% (913) | **76.15%** (933.7) | **+1.68 pp** | 0.1182 m | **0.1157 m** | **+2.11%** | 0.2035 m | **0.1964 m** | **+3.49%** |
| **Transition** | 124 | 48.39% (60) | **49.19%** (61.0) | **+0.81 pp** | 0.2026 m | **0.2010 m** | **+0.79%** | 0.3694 m | **0.3660 m** | **+0.92%** |
| **Change** | 211 | 41.71% (88) | **41.71%** (88.0) | **+0.00 pp** | 0.2383 m | **0.2352 m** | **+1.32%** | 0.4320 m | **0.4168 m** | **+3.52%** |
| **Clear Event** | 0 | - | - | - | - | - | - | - | - | - |

---

## 3. Slot Collapse & Diversity Monitoring (Per-Seed Breakdown)

| Seed | Mean Pairwise Dist (m) | Median Pairwise Dist (m) | Effective Slots ($\ge 0.1\text{m}$) | Slot Entropy (Overall) | Slot Entropy (Change) | Slot Winners Overall [0, 1, 2, 3] (%) | Max Dominance | Collapse Status |
|---|---|---|---|---|---|---|---|---|
| **Seed 42** | 0.638 m | 0.526 m | 3.93 / 4.0 | 1.926 bits | 1.500 bits | [24.0%, 22.4%, 15.6%, 38.1%] | 38.1% | **HEALTHY** |
| **Seed 43** | 0.598 m | 0.497 m | 3.87 / 4.0 | 1.916 bits | 1.540 bits | [24.5%, 24.7%, 13.5%, 37.3%] | 37.3% | **HEALTHY** |
| **Seed 44** | 0.630 m | 0.523 m | 3.90 / 4.0 | 1.938 bits | 1.559 bits | [23.1%, 23.6%, 16.3%, 36.9%] | 36.9% | **HEALTHY** |
| **Mean** | **0.622 m** | **0.515 m** | **3.90 / 4.0** | **1.927 bits** | **1.533 bits** | **[23.9%, 23.6%, 15.1%, 37.4%]** | **37.4%** | **NO COLLAPSE** |

---

## 4. Cross-Seed Consistency Audit

### (1) Overall Consistency ($N=1561$)
- **Rescues**:
  - 2/3 seeds common rescues: **24 samples**
  - 3/3 seeds common rescues: **8 samples** (Sample IDs: `1440, 2436, 2476, 4526, 6255, 6290, 7874, 8373`)
- **Losts**:
  - 2/3 seeds common losts: **18 samples**
  - 3/3 seeds common losts: **14 samples** (Sample IDs: `1388, 1931, 2375, 2797, 4069, 4281, 4416, 4468, 5971, 6204, 6405, 6832, 7597, 8900`)

### (2) Change Subgroup Consistency ($N=211$)
- **Change Rescues**:
  - 2/3 seeds common Change rescues: **2 samples** (`6255, 6290`)
  - 3/3 seeds common Change rescues: **2 samples** (`6255, 6290`)
- **Change Losts**:
  - 2/3 seeds common Change losts: **4 samples** (`4069, 4468, 6204, 6832`)
  - 3/3 seeds common Change losts: **4 samples** (`4069, 4468, 6204, 6832`)

---

## 5. Kinematics Attribution Audit: Acceleration-p95 Source Localization

| Proposal Sub-Pool | Speed Mean (m/s) | Speed p95 (m/s) | Accel Mean (m/s$^2$) | Accel p95 (m/s$^2$) | Jerk Mean (m/s$^3$) | Jerk p95 (m/s$^3$) |
|---|---|---|---|---|---|---|
| **Native 20 Proposals** | 0.7136 | 1.4245 | 0.0978 | **0.6928** | 0.5381 | **3.1627** |
| **Static 4 Proposals** | 0.6224 | 1.4371 | 0.1225 | **0.3164** | 0.0287 | **0.0861** |
| **Static-K24 Full Pool** | 0.6984 | 1.4261 | 0.1020 | **0.5352** | 0.4532 | **2.9815** |
| **Learned 4 Proposals (Seed 42)** | 0.6789 | 1.4427 | 0.0821 | **0.2569** | 0.1222 | **0.4397** |
| **Learned 4 Proposals (Seed 43)** | 0.6776 | 1.4335 | 0.0856 | **0.2619** | 0.1215 | **0.4469** |
| **Learned 4 Proposals (Seed 44)** | 0.6701 | 1.4165 | 0.0841 | **0.2654** | 0.1302 | **0.4661** |
| **Rescued Winners (Seed 42)** | 1.0576 | 1.6187 | 0.0793 | **0.2485** | 0.1249 | **0.3677** |
| **Rescued Winners (Seed 43)** | 1.0199 | 1.6395 | 0.1170 | **0.3156** | 0.1706 | **0.5306** |
| **Rescued Winners (Seed 44)** | 1.1164 | 1.5180 | 0.0926 | **0.2609** | 0.1454 | **0.5208** |

### 动力学归因结论
- **Native 20 本身的加速度 p95 高达 0.6928 m/s$^2$**，是池中主要的高加速度源；
- **Learned 4 本身非常平滑**（Accel p95 仅 0.257 ~ 0.265 m/s$^2$，远低于 Native 20 的 0.6928，甚至低于 Static 4 的 0.3164）；
- **Rescued Winners 同样非常平滑**（Accel p95 仅 0.249 ~ 0.316 m/s$^2$）；
- 全池（K=24）加速度 p95 从 0.5352 上升至 0.5490（+2.59%）并非因为模型输出了激进或不平滑轨迹，而是因为将 Static 4（低速低加样本分布）替换为动态适应的四条轨迹后，全池混合分位点在统计聚合时发生了微小漂移。

---

## 6. 三代生成器在 Change 困难子集上的最终对比

| 生成器范式 | 监督目标 | Overall Cov 增益 | Change Cov 增益 | Change $\min\text{ADE}$ 增益 | Change $\min\text{FDE}$ 增益 | Change 正向种子数 | 门禁判定 |
|---|---|---|---|---|---|---|---|
| **确定性回归 (Stage 03)** | 单一 $z^*$ MSE | +1.00 pp | **-2.37 pp** | **-1.88%** | **-1.67%** | 0/3 | FAIL |
| **高斯 MDN (Stage 03)** | 单一 $z^*$ 高斯 NLL | +0.49 pp | **-2.53 pp** | **-1.81%** | **-1.58%** | 0/3 | FAIL |
| **Best-of-4 (Stage B)** | 轨迹空间 $\min_m \mathcal{L}_{\text{traj}}$ | **+1.39 pp** | **+0.00 pp** | **+1.32%** | **+3.52%** | **1/3** | **STOP_SEED_SPECIFIC** |
