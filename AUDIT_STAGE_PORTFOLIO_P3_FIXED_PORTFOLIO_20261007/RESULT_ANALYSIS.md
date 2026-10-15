# RESULT ANALYSIS: STAGE P3 FIXED PORTFOLIO K24 AUDIT

## 1. Executive Summary & Preregistered Decision Rule Evaluation
- **Stage**: `PORTFOLIO_P3_FIXED_PORTFOLIO` (Configurations: S4L0, S3L1, S2L2, S1L3, S0L4)
- **Status**: `FIXED_PORTFOLIO_POSITIVE`
- **Preregistered Gate Evaluation**:
  1. *Overall coverage not lower than Static-K24 in 3 seeds*:
     - **S2L2**: PASS (+0.64, +1.02, +0.06 pp, mean **+0.58 pp**).
     - **S1L3**: PASS (+1.22, +1.60, +0.70 pp, mean **+1.17 pp**).
  2. *Change 3-seed mean coverage > Static-K24 (41.71%)*:
     - **S2L2**: **43.13% vs 42.18% (+0.95 pp)** (PASS).
     - **S1L3**: **43.13% vs 42.18% (+0.95 pp)** (PASS).
  3. *Change coverage positive in $\ge 2/3$ seeds*:
     - **S2L2**: **3 / 3 seeds positive** (+0.95, +1.42, +0.47 pp) (PASS).
     - **S1L3**: **3 / 3 seeds positive** (+1.42, +0.95, +0.47 pp) (PASS).
  4. *Change minADE or minFDE stably improved*:
     - **S2L2**: Change minADE **+2.32%**, Change minFDE **+3.95%** (PASS).
     - **S1L3**: Change minADE **+2.11%**, Change minFDE **+3.69%** (PASS).
  5. *Motion guards ($\le 2.0\%$)*:
     - **S2L2**: Accel p95 diffs = **[+0.59%, -0.04%, +1.02%]%** $\implies$ **3 / 3 seeds PASS**.
     - **S1L3**: Accel p95 diffs = **[+1.13%, +0.14%, +1.35%]%** $\implies$ **3 / 3 seeds PASS**.
- **Conclusion**:
  Both **S2L2** (2 Static + 2 Learned) and **S1L3** (1 Static + 3 Learned) solidly pass all 5 preregistered criteria. By retaining both the top classifier static proposals and the learned deceleration/stopping proposals, the mixed portfolio captures the complementary support of both sources under the strict $K=24$ budget.

---

## 2. Quantitative Performance Table Across All 5 Fixed Portfolio Configurations

| Portfolio Configuration | Total Proposals | Total $K$ | Overall Cov Gain (Mean) | Change Cov Gain (Mean) | Change Positive Seeds | Change $\min\text{ADE}$ Gain | Change $\min\text{FDE}$ Gain | Accel p95 Diff (Max across seeds) | Gate Passed? |
|---|---|---|---|---|---|---|---|---|---|
| **S4L0 (Static-K24 Control)** | 4 Static + 0 Learned | 24 | +0.00 pp | +0.00 pp | 0 / 3 | +0.00% | +0.00% | 0.00% | Baseline |
| **S3L1** | 3 Static + 1 Learned | 24 | -0.02 pp | +0.79 pp | 3 / 3 | +2.30% | +3.83% | +1.01% | FAIL (Overall -0.02) |
| **S2L2 (Top-2 S + Top-2 L)** | **2 Static + 2 Learned** | **24** | **+0.58 pp** | **+0.95 pp** | **3 / 3 (100%)** | **+2.32%** | **+3.95%** | **+1.02% (PASS)** | **PASS (POSITIVE)** |
| **S1L3 (Top-1 S + Top-3 L)** | **1 Static + 3 Learned** | **24** | **+1.17 pp** | **+0.95 pp** | **3 / 3 (100%)** | **+2.11%** | **+3.69%** | **+1.35% (PASS)** | **PASS (POSITIVE)** |
| **S0L4 (Bo4-K24 Treatment)** | 0 Static + 4 Learned | 24 | +1.32 pp | **-0.47 pp** | **0 / 3 (0%)** | +1.18% | +3.41% | +2.29% (FAIL) | FAIL (Change negative) |

---

## 3. Seed-by-Seed Breakdown for Winning Portfolios (S2L2 and S1L3)

### S2L2 (2 Static: Decel, Stop + 2 Learned: Slot 3 Decel/Stop, Slot 0 Cruise/Accel)
- **Seed 42**: Overall 68.67% (+0.64 pp, R: 20, L: 10, Net: +10), Change 43.13% (**+0.95 pp**), Acc diff +0.59%
- **Seed 43**: Overall 69.06% (+1.02 pp, R: 26, L: 10, Net: +16), Change 43.60% (**+1.42 pp**), Acc diff -0.04%
- **Seed 44**: Overall 68.10% (+0.06 pp, R: 13, L: 12, Net: +1), Change 42.65% (**+0.47 pp**), Acc diff +1.02%
- **3-Seed Mean**: Overall **+0.58 pp**, Change **+0.95 pp**, Change ADE **+2.32%**, Change FDE **+3.95%**, Accel diff **$\le 1.02\%$**.

### S1L3 (1 Static: Decel + 3 Learned: Slot 3 Decel/Stop, Slot 0 Cruise/Accel, Slot 1 Turn Left)
- **Seed 42**: Overall 69.25% (+1.22 pp, R: 31, L: 12, Net: +19), Change 43.60% (**+1.42 pp**), Acc diff +1.13%
- **Seed 43**: Overall 69.63% (+1.60 pp, R: 37, L: 12, Net: +25), Change 43.13% (**+0.95 pp**), Acc diff +0.14%
- **Seed 44**: Overall 68.74% (+0.70 pp, R: 24, L: 13, Net: +11), Change 42.65% (**+0.47 pp**), Acc diff +1.35%
- **3-Seed Mean**: Overall **+1.17 pp**, Change **+0.95 pp**, Change ADE **+2.11%**, Change FDE **+3.69%**, Accel diff **$\le 1.35\%$**.

---

## 4. Triggered Outcome & Next Step
- Stage P3 officially passed with `FIXED_PORTFOLIO_POSITIVE`.
- Per preregistered instructions:
  > “如果 P3 固定 portfolio 已稳定超过 Static：输出 FIXED_PORTFOLIO_POSITIVE $\implies$ 仍允许继续 P4，仅用于判断 adaptive 是否进一步值得。”
