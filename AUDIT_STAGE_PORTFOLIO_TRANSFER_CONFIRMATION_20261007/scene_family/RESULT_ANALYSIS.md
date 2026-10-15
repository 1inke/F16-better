# Scene-Family Transfer Confirmation Analysis Report

## 1. Summary of Results vs Canonical Static-K24 Baseline
- **Canonical Control Baseline**: Overall coverage = **67.97%** (1061/1561), Change coverage = **41.71%** (88/211).
- **Finalist 1: S1L3 (Native20 + 1 Static + Learned Slots [3, 0, 1])**:
  - Seed 42: Overall = 69.31% (+1.35 pp), Change = **43.60% (+1.90 pp)**, R:4, L:0 on Change.
  - Seed 43: Overall = 69.63% (+1.67 pp), Change = **43.13% (+1.42 pp)**, R:4, L:1 on Change.
  - Seed 44: Overall = 68.80% (+0.83 pp), Change = **42.65% (+0.95 pp)**, R:2, L:0 on Change.
  - **3-Seed Mean**: Overall Gain = **+1.28 pp**, Change Gain = **+1.42 pp** (**3/3 seeds positive on Change**).
  - Change minADE Gain = **+2.45%**, Change minFDE Gain = **+3.71%**.
- **Finalist 2: S2L2 (Native20 + 2 Static + Learned Slots [3, 0])**:
  - Seed 42: Overall = 68.80% (+0.83 pp), Change = **43.13% (+1.42 pp)**, R:3, L:0 on Change.
  - Seed 43: Overall = 69.19% (+1.22 pp), Change = **43.60% (+1.90 pp)**, R:4, L:0 on Change.
  - Seed 44: Overall = 68.29% (+0.32 pp), Change = **42.65% (+0.95 pp)**, R:2, L:0 on Change.
  - **3-Seed Mean**: Overall Gain = **+0.79 pp**, Change Gain = **+1.42 pp** (**3/3 seeds positive on Change**).
  - Change minADE Gain = **+2.67%**, Change minFDE Gain = **+4.10%**.

## 2. Motion and Kinematics Attribution
- Proposals themselves have acceleration p95 around $0.26 - 0.28\text{ m/s}^2$ (exceptionally smooth).
- Pooled acceleration p95 shifted downward from $0.5656$ to $0.542 - 0.545\text{ m/s}^2$ (marked as `POOL_COMPOSITION_EFFECT`). No real trajectory pathology detected.
