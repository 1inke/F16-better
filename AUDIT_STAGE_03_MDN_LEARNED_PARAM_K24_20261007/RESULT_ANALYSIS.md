# RESULT ANALYSIS: STAGE 03 MDN-AUGMENTED LEARNED-PARAM-K24

## 1. Executive Summary & Gate Decision
- **Stage**: `03_LEARNED_PARAM_K24` (Probabilistic Branch: 3-component Gaussian MDN + Joint Mode Allocator)
- **Status**: `SUCCESS_SCIENTIFIC_NEGATIVE`
- **Promotion Gate Status**: **FAIL (0/3 seeds passed)**
- **Preregistered Gate Criteria**:
  1. Change subgroup $\min\text{ADE}$ or $\min\text{FDE}$ gain $\ge 3.0\%$, OR Change coverage gain $\ge +1.0\text{ pp}$.
  2. Overall ADE harm $\le 1.0\%$ (overall ADE gain $\ge -1.0\%$) and overall FDE harm $\le 1.0\%$.
  3. Kinematics guard pass: $|\Delta\text{speed}_{p95}| \le 2\%$, $|\Delta\text{accel}_{p95}| \le 2\%$, $|\Delta\text{jerk}_{p95}| \le 2\%$.
  4. At least 2 of 3 seeds satisfy all conditions.

---

## 2. Quantitative Evaluation Table

### Overall Performance Across 3 Seeds
| Model | Seed | Overall Coverage (%) | Covered / 1561 | $\Delta\text{Cov}$ (pp) | Rescue | Lost | Net Rescue | Mean $J$ | $\min\text{ADE}$ (m) | $\min\text{FDE}$ (m) | Eff. Clusters | Gate Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Static-K24 (Control)** | - | 67.97% | 1061 | - | - | - | - | 0.7985 | 0.1411 | 0.2476 | 1832 | - |
| MDN-K24 | 42 | 69.19% | 1080 | +1.22 | 35 | 16 | +19 | 0.7949 (+0.45%) | 0.1411 (+0.01%) | 0.2431 (+1.80%) | 1708 | False |
| MDN-K24 | 43 | 68.48% | 1069 | +0.51 | 25 | 17 | +8 | 0.7962 (+0.29%) | 0.1408 (+0.20%) | 0.2434 (+1.67%) | 1658 | False |
| MDN-K24 | 44 | 67.71% | 1057 | -0.26 | 13 | 17 | -4 | 0.8012 (-0.34%) | 0.1415 (-0.26%) | 0.2458 (+0.71%) | 1661 | False |
| **MDN-K24 (Mean)** | - | **68.46%** | **1068.7** | **+0.49** | **24.3** | **16.7** | **+7.7** | **0.7975 (+0.13%)** | **0.1411 (-0.01%)** | **0.2441 (+1.39%)** | **1675.7** | **0/3** |

### Subgroup Coverage & Errors (Mean Across 3 Seeds)
| Metric | Control (Static-K24) | Seed 42 | Seed 43 | Seed 44 | 3-Seed Mean | Net Gain |
|---|---|---|---|---|---|---|
| **Stable Coverage** ($N=1226$) | 74.47% (913) | 76.35% (936) | 75.53% (926) | 74.47% (913) | 75.45% (925.0) | **+0.98 pp** |
| Stable $\min\text{ADE}$ (m) | 0.1181 | 0.1173 | 0.1170 | 0.1179 | 0.1174 | +0.61% |
| **Transition Coverage** ($N=124$) | 48.39% (60) | 49.19% (61) | 49.19% (61) | 49.19% (61) | 49.19% (61.0) | **+0.81 pp** |
| Transition $\min\text{ADE}$ (m) | 0.2030 | 0.2039 | 0.2032 | 0.2026 | 0.2033 | -0.11% |
| **Change Coverage** ($N=211$) | 41.71% (88) | 39.34% (83) | 38.86% (82) | 39.34% (83) | 39.18% (82.7) | **-2.53 pp** |
| Change $\min\text{ADE}$ (m) | 0.2382 | 0.2424 | 0.2426 | 0.2425 | 0.2425 | **-1.81%** |
| Change $\min\text{FDE}$ (m) | 0.4318 | 0.4387 | 0.4392 | 0.4379 | 0.4386 | **-1.58%** |

### Kinematics Summary
| Metric | Control (Static-K24) | Seed 42 | Seed 43 | Seed 44 | Guard Threshold | Passed? |
|---|---|---|---|---|---|---|
| Speed p95 (m/s) | 1.4347 | 1.4489 (+0.99%) | 1.4703 (+2.48%) | 1.4690 (+2.39%) | $\le 2.0\%$ | 1/3 pass |
| Accel p95 (m/s$^2$) | 0.5363 | 0.5343 (-0.37%) | 0.5540 (+3.29%) | 0.5433 (+1.31%) | $\le 2.0\%$ | 2/3 pass |
| Jerk p95 (m/s$^3$) | 2.9815 | 2.9815 (0.00%) | 2.9815 (0.00%) | 2.9815 (0.00%) | $\le 2.0\%$ | 3/3 pass |

---

## 3. Comparison: Deterministic Regression vs Gaussian MDN
| Dimension | Deterministic MLP (Stage 03 Det) | 3-Component Gaussian MDN (Stage 03 MDN) | Static-K24 Baseline |
|---|---|---|---|
| Overall Coverage | 68.97% (+1.00 pp) | 68.46% (+0.49 pp) | 67.97% |
| Stable Coverage | 76.10% (+1.63 pp) | 75.45% (+0.98 pp) | 74.47% |
| Transition Coverage | 50.00% (+1.62 pp) | 49.19% (+0.81 pp) | 48.39% |
| **Change Coverage** | **39.34% (-2.37 pp)** | **39.18% (-2.53 pp)** | **41.71%** |
| Change $\min\text{ADE}$ | 0.2425 (-1.88%) | 0.2425 (-1.81%) | 0.2382 |
| Net Rescue (Mean) | +15.7 | +7.7 | 0 |
| Passing Seeds | 0/3 | 0/3 | - |
