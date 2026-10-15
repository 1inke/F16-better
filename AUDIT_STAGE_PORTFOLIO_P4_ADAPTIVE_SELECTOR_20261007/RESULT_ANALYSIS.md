# Stage P4: GT-Free Adaptive 8->4 Portfolio Selector Audit Report

## 1. Executive Summary & Preregistered Decision
- **Scientific Question**: Can a GT-free, past-conditioned proposal portfolio selector (trained on training-split usefulness without any outer GT leakage) dynamically allocate 4 proposal slots out of 8 (Static4 + Learned4) and outperform fixed allocation portfolios?
- **Preregistered Primary Gate Thresholds**:
  - Overall coverage gain $\ge 0.0$ pp vs Static-K24
  - Change coverage mean gain $\ge +1.0$ pp vs Static-K24
  - Change gain positive in $\ge 2/3$ seeds
  - Change minADE / minFDE at least one improved
  - Kinematics guard passed ($|\Delta \text{accel p95}| \le 2.0\%$)
- **Empirical Outcome**:
  - Mean Overall Coverage Gain: **+1.30 pp** (Passed)
  - Mean Change Coverage Gain: **+0.32 pp** (Seed 42: +0.47, Seed 43: +0.47, Seed 44: +0.00 pp) -> **Failed ($\le +1.0$ pp)**
  - Change Positive Seeds: 2/3 (Passed)
  - Change minADE Gain: +1.63%, minFDE Gain: +3.61% (Passed)
  - Acceleration p95 Diff: **[-8.98%, -9.65%, -8.72%]%** -> **Failed ($> 2.0\%$ shift)**
  - Primary Gate Pass: **FALSE**
- **Preregistered Branch Action**:
  - `FREEZE_BEST_FIXED_OR_STATIC_K24` (specifically, freeze **S1L3** / **S2L2** validated in Stage P3).

---

## 2. Quantitative Results by Seed

| Seed | Overall Cov (%) | Overall Gain (pp) | R / L / Net | Change Cov (%) | Change Gain (pp) | Change ADE (%) | Change FDE (%) | Chosen Static % | Accel p95 Diff (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 42 | 69.31 | +1.35 | 39 / 18 / +21 | 41.71 | +0.47 | +2.06 | +3.93 | 15.4% | -8.98 |
| 43 | 69.57 | +1.60 | 43 / 18 / +25 | 41.71 | +0.47 | +1.89 | +3.70 | 21.9% | -9.65 |
| 44 | 68.93 | +0.96 | 33 / 18 / +15 | 41.23 | +0.00 | +0.95 | +3.19 | 18.0% | -8.72 |
| **Mean** | **69.27** | **+1.30** | **38.3 / 18 / +20.3** | **41.55** | **+0.32** | **+1.63** | **+3.61** | **18.4%** | **-9.12** |

---

## 3. Subgroup Performance Breakdown (3-Seed Mean)

| Subgroup | Static-K24 Cov (%) | Adaptive-K24 Cov (%) | Gain (pp) | minADE (m) | minFDE (m) | J |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Overall | 67.97 | 69.27 | +1.30 | 0.1385 | 0.2395 | 0.7821 |
| Stable | 75.12 | 76.16 | +1.03 | 0.1157 | 0.1962 | 0.6443 |
| Transition | 49.19 | 48.39 | -0.81 | 0.2007 | 0.3668 | 1.1502 |
| Change | 41.23 | 41.55 | +0.32 | 0.2346 | 0.4166 | 1.3670 |

---

## 4. Root Cause Analysis: Why Fixed Allocation Outperforms Adaptive Selection

1. **Lost Windows Persistence ($L = 18$)**:
   - In Stage P0, we discovered that Static4 contributes ~17 windows exclusively that Learned4 cannot cover.
   - When using fixed portfolios like **S2L2** in Stage P3, the top-2 static hypotheses are guaranteed to be present in every single sample, retaining those critical static trajectories and yielding **+0.95 pp gain on Change across all 3 seeds**.
   - The adaptive selector, trained on general usefulness, heavily preferred the smooth Learned proposals (~81.6% learned share chosen). Consequently, on the samples where Static proposals were essential, the selector often dropped all static candidates, causing 18 lost windows.
2. **Kinematic Drift**:
   - By disproportionately selecting Learned proposals over Static proposals, the acceleration p95 shifted downward by ~9%, violating the tight $\pm 2.0\%$ compositional envelope.
3. **Conclusion**:
   - Structural allocation (guaranteeing at least 1-2 slots for Static heuristic diversity, such as in S2L2 or S1L3) provides a stronger and more reliable coverage guarantee than unconstrained top-4 soft scoring from an adaptive selector.
