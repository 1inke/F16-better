# RESULT ANALYSIS: STAGE P0 STATIC4 + LEARNED4 COMPLEMENTARITY AUDIT

## 1. Executive Summary & Assertion Audit
- **Stage**: `PORTFOLIO_P0_COMPLEMENTARITY` (Diagnostic K28 Union: Native20 + Static4 + Learned4)
- **Status**: `SUCCESS_COMPLEMENTARITY_CONFIRMED`
- **Strict Invariance Assertions**:
  - `lost_vs_static == 0`: **PASSED (0 lost across all 3 seeds)**.
  - `union_coverage >= static_coverage`: **PASSED (all 3 seeds strictly exceed Static-K24)**.
- **Key Empirical Finding**:
  - Mean Union-K28 Overall Coverage Gain: **+2.48 pp** (67.97% $\to$ 70.45%, mean rescue 38.7, lost 0).
  - Mean Union-K28 Change Coverage Gain: **+2.05 pp** (41.71% $\to$ 43.76%, rescue 4.3, lost 0).
  - Seed 42 Change Coverage: **44.08%** (+2.37 pp, 93 / 211)
  - Seed 43 Change Coverage: **44.08%** (+2.37 pp, 93 / 211)
  - Seed 44 Change Coverage: **43.13%** (+1.42 pp, 91 / 211)
- **Conclusion**:
  Static4 and Learned4 possess strong, non-redundant complementary candidate support. Replacing Static4 with Learned4 in Bo4-K24 sacrificed Static4-unique windows (17 overall, 4~5 on Change). Unifying both raises Change coverage beyond 44%.

---

## 2. Quantitative Performance Table: Static-K24 vs Union-K28

| Seed / Group | Overall Coverage (%) | Covered / 1561 | $\Delta\text{Cov}$ (pp) | Rescue | Lost | Change Coverage (%) | Change $\Delta\text{Cov}$ | Stable Coverage (%) | Transition Coverage (%) |
|---|---|---|---|---|---|---|---|---|---|
| **Control (Static-K24)** | 67.97% | 1061 | - | - | - | 41.71% (88 / 211) | - | 74.47% (913 / 1226) | 48.39% (60 / 124) |
| **Union-K28 Seed 42** | 70.53% | 1101 | +2.56 pp | 40 | 0 | **44.08% (93 / 211)** | **+2.37 pp** | 77.08% (945) | 50.81% (63) |
| **Union-K28 Seed 43** | 70.72% | 1104 | +2.75 pp | 43 | 0 | **44.08% (93 / 211)** | **+2.37 pp** | 77.08% (945) | 53.23% (66) |
| **Union-K28 Seed 44** | 70.08% | 1094 | +2.11 pp | 33 | 0 | **43.13% (91 / 211)** | **+1.42 pp** | 76.67% (940) | 50.81% (63) |
| **3-Seed Mean** | **70.45%** | **1099.7** | **+2.48 pp** | **38.7** | **0** | **43.76% (92.3 / 211)** | **+2.05 pp** | **76.94%** | **51.61%** |

---

## 3. Set-Theoretic Decomposition: Static4 vs Learned4

### (1) Overall Set-Theoretic Distribution ($N=1561$)
| Metric | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---|---|---|---|
| **Static4-only Coverage** (beyond Native20) | 17 (1.09%) | 17 (1.09%) | 17 (1.09%) | 17.0 (1.09%) |
| **Learned4-only Coverage** (beyond Native20) | 40 (2.56%) | 43 (2.75%) | 33 (2.11%) | 38.7 (2.48%) |
| **Both Proposals Overlap** | 293 (18.77%) | 290 (18.58%) | 288 (18.45%) | 290.3 (18.60%) |
| **Union Proposals Coverage** | 729 (46.70%) | 748 (47.92%) | 735 (47.09%) | 737.3 (47.24%) |
| **Jaccard Similarity** (Static4 vs Learned4) | 0.4019 | 0.3877 | 0.3918 | 0.3938 |

### (2) Change Subgroup Set-Theoretic Distribution ($N=211$)
| Metric | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---|---|---|---|
| **Static4-only Coverage** | 4 (1.90%) | 5 (2.37%) | 4 (1.90%) | 4.3 (2.05%) |
| **Learned4-only Coverage** | 5 (2.37%) | 5 (2.37%) | 3 (1.42%) | 4.3 (2.05%) |
| **Both Proposals Overlap** | 24 (11.37%) | 25 (11.85%) | 25 (11.85%) | 24.7 (11.69%) |
| **Union Proposals Coverage** | 62 (29.38%) | 61 (28.91%) | 58 (27.49%) | 60.3 (28.60%) |
| **Jaccard Similarity** (Change) | 0.3871 | 0.4098 | 0.4310 | 0.4093 |

---

## 4. Decision Rule Verification
- **Rule Check**:
  - `has_complementarity = True` (Mean union gain +2.48 pp > 0.5 pp, mean rescues 38.7 >= 15, Change gain +2.05 pp > 0).
- **Triggered Next Step**: Proceed immediately to **Stage P1: Oracle 8 $\to$ 4 Budget Audit**.
