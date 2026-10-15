# Phase M2: Adaptive Portfolio Composition Analysis

## 1. Key Empirical Findings
Comparing the 4 candidate portfolios across 3 protocols ($N=3,162$) and 3 seeds (42, 43, 44):

1. **$C_2$ (Dual-Learned: `[Native Top-1, Learned-3, Learned-0]`) provides superior proposal performance**:
   - Overall coverage increases from **34.95%** ($C_1$) to **37.12%** (**+2.17 pp gain**).
   - Change coverage increases from **19.70%** ($C_1$) to **20.45%** (**+0.75 pp gain**).
   - Pairwise trajectory diversity reaches **0.538m** (vs 0.409m for $C_1$), representing the highest diversity among all portfolios.
   - **Reason**: Learned Slot 0 contributes a distinct acceleration/steering proposal that complements the deceleration/stopping profile of Learned Slot 3 more effectively than the fixed Static heuristic.

2. **$C_4$ (Pure Native Top-3: `[N1, N2, N3]`)**:
   - Achieves higher overall coverage (42.06%), but suffers from **drastically lower pairwise diversity (0.260m vs 0.538m)**.
   - Native hypotheses cluster closely together around conservative straight-line paths, providing poor qualitative hypothesis separation during abrupt maneuvers.

## 2. Conclusion
$C_1$ remains a strong, stable baseline portfolio, while $C_2$ (incorporating two complementary learned modes) represents an attractive higher-diversity alternative for multi-hypothesis deployment.
