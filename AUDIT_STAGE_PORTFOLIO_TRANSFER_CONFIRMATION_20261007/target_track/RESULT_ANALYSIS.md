# Target-Track Transfer Confirmation Analysis Report

## 1. Summary of Results vs Canonical Isolated Static-K24 Baseline
- **Canonical Control Baseline**: Overall coverage = **61.64%** (638/1035), Change coverage = **43.48%** (110/253).
- **Finalist 1: S1L3 (Native20 + 1 Static + Learned Slots [3, 0, 1])**:
  - Seed 42: Overall = 62.71% (+1.06 pp), Change = **45.06% (+1.58 pp)**, R:4, L:0 on Change.
  - Seed 43: Overall = 62.80% (+1.16 pp), Change = **45.45% (+1.98 pp)**, R:5, L:0 on Change.
  - Seed 44: Overall = 63.09% (+1.45 pp), Change = **46.25% (+2.77 pp)**, R:7, L:0 on Change.
  - **3-Seed Mean**: Overall Gain = **+1.22 pp**, Change Gain = **+2.11 pp** (**3/3 seeds positive on Change**).
  - Change minADE Gain = **+1.35%**, Change minFDE Gain = **+1.11%**.
  - Acceleration p95 diff = **+1.35%** (all 3 seeds strictly $\le 2.0\%$ motion guard, passed!).
- **Finalist 2: S2L2 (Native20 + 2 Static + Learned Slots [3, 0])**:
  - Seed 42: Overall = 62.22% (+0.58 pp), Change = **45.06% (+1.58 pp)**, R:4, L:0 on Change.
  - Seed 43: Overall = 62.61% (+0.97 pp), Change = **45.45% (+1.98 pp)**, R:5, L:0 on Change.
  - Seed 44: Overall = 62.80% (+1.16 pp), Change = **46.25% (+2.77 pp)**, R:7, L:0 on Change.
  - **3-Seed Mean**: Overall Gain = **+0.90 pp**, Change Gain = **+2.11 pp** (**3/3 seeds positive on Change**).
  - Change minADE Gain = **+1.36%**, Change minFDE Gain = **+1.60%**.
  - Acceleration p95 diff = **+0.51%** (all 3 seeds strictly $\le 2.0\%$ motion guard, passed!).

## 2. Invariance and Consistency Verification
- On Target-Track, **Lost windows on Change are identically 0 across all 3 seeds for both S1L3 and S2L2**.
- Both finalist portfolios achieved identical Change coverage gains (+2.11 pp mean), with S1L3 providing higher overall coverage gains (+1.22 pp vs +0.90 pp).
