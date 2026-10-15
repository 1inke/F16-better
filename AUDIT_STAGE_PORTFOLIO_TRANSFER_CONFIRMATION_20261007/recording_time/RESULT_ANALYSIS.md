# Recording-Time Transfer Confirmation Analysis Report

## 1. Summary of Results vs Canonical Isolated Static-K24 Baseline
- **Canonical Control Baseline**: Overall coverage = **56.01%** (317/566), Change coverage = **43.66%** (62/142).
- **Finalist 1: S1L3 (Native20 + 1 Static + Learned Slots [3, 0, 1])**:
  - Seed 42: Overall = 54.77% (-1.24 pp), Change = **43.66% (+0.00 pp)**, R:0, L:0 on Change.
  - Seed 43: Overall = 56.01% (+0.00 pp), Change = **45.77% (+2.11 pp)**, R:3, L:0 on Change.
  - Seed 44: Overall = 55.65% (-0.35 pp), Change = **44.37% (+0.70 pp)**, R:1, L:0 on Change.
  - **3-Seed Mean**: Overall Gain = **-0.53 pp**, Change Gain = **+0.94 pp** (**2/3 seeds positive, 1/3 neutral, 0/3 negative on Change**).
  - Change minADE Gain = **+0.41%**, Change minFDE Gain = **+0.42%**.
  - Acceleration p95 diff = **+1.54%** (all 3 seeds strictly $\le 2.0\%$ motion guard, passed!).
- **Finalist 2: S2L2 (Native20 + 2 Static + Learned Slots [3, 0])**:
  - Seed 42: Overall = 54.95% (-1.06 pp), Change = **42.96% (-0.70 pp)**, R:0, L:1 on Change.
  - Seed 43: Overall = 55.83% (-0.18 pp), Change = **45.07% (+1.41 pp)**, R:3, L:1 on Change.
  - Seed 44: Overall = 55.48% (-0.53 pp), Change = **43.66% (+0.00 pp)**, R:1, L:1 on Change.
  - **3-Seed Mean**: Overall Gain = **-0.59 pp**, Change Gain = **+0.23 pp** (**1/3 seeds positive on Change**).
  - Acceleration p95 diff = **+0.55%** (all 3 seeds strictly $\le 2.0\%$ motion guard, passed!).

## 2. Invariance and Consistency Verification
- On Recording-Time, S1L3 strictly avoided any lost windows on Change across all 3 seeds ($L=0, 0, 0$).
- S1L3 outperformed S2L2 on Change gain (+0.94 pp vs +0.23 pp) and positive seeds count (2/3 positive vs 1/3 positive).
