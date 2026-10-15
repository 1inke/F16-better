# DESIGN: Stage P4 GT-Free Adaptive 8 -> 4 Portfolio Selector

## 1. Scientific Objective
- **Problem**: In Stage P3, fixed portfolios S2L2 and S1L3 achieved +0.95 pp Change coverage gain with 3/3 seeds positive. However, Stage P1 established that Oracle 8 $\to$ 4 selection has an upper bound gain of +2.05 pp on Change (43.29% vs 41.23%).
- **Hypothesis**: A lightweight, past-conditioned proposal portfolio selector can dynamically adapt the 4 slots per sample (allocating more learned proposals to change scenarios and static proposals to routine scenarios), capturing additional headroom toward the Oracle ceiling without exceeding the $K=24$ budget.
- **Strict Invariance**:
  - Native 20 strictly frozen.
  - Deployed budget strictly $K=24$ (20 Native + 4 selected proposals).
  - No future information or GT used during inference.
  - Training strictly on Zara scene-family train split ($N=16,337$); evaluated on isolated outer split ($N=1,561$).
- **Supervision Objective**:
  Set-retention objective: binary cross-entropy on proposal usefulness ($ADE \le 0.2 \land FDE \le 0.4$), with double sample weighting on samples where Native20 failed (rescue-critical retention).
- **Preregistered Comparison Suite**:
  1. Static4 (S4L0)
  2. Learned4 (S0L4)
  3. Best Fixed Portfolio (S1L3 / S2L2)
  4. Adaptive Selector (Adaptive-4of8)
  5. Oracle4Of8 Upper Bound
- **Decision Rules**:
  - If Adaptive Selector satisfies Primary Gate (Change gain $\ge +1.0$ pp, $\ge 2/3$ seeds positive, Overall gain $\ge 0$) $\implies$ `PORTFOLIO_K24_ACCEPTED`.
  - If Adaptive Selector fails on Change $\implies$ `FREEZE_BEST_FIXED_OR_STATIC_K24` (freezing S1L3 / S2L2 as optimal production portfolio).
