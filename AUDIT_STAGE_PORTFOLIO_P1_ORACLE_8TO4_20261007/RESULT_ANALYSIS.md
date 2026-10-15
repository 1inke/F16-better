# RESULT ANALYSIS: STAGE P1 ORACLE 8 -> 4 BUDGET AUDIT

## 1. Executive Summary & Preregistered Criteria
- **Stage**: `PORTFOLIO_P1_ORACLE_8TO4`
- **Status**: `SUCCESS_BUDGET_CONFIRMED_SUFFICIENT`
- **Core Preregistered Metric**:
  - `Rescue Retention Rate`: $\frac{\text{Rescue}(\text{OracleBest4Of8})}{\text{Rescue}(\text{K28 Union})} \ge 95\%$.
- **Empirical Findings Across All 3 Seeds**:
  - **Overall Rescue Retention**: **100.00%** (Seed 42: 40/40, Seed 43: 43/43, Seed 44: 33/33).
  - **Change Rescue Retention**: **100.00%** (Seed 42: 5/5, Seed 43: 5/5, Seed 44: 3/3).
  - **Oracle K24 Coverage**: **70.38%** (Overall gain +2.48 pp over Static-K24).
  - **Oracle K24 Change Coverage**: **43.29%** (Change gain +2.05 pp over Static-K24).
- **Scientific Conclusion**:
  > **The 4-slot proposal budget ($K=24$ total) is NOT the bottleneck.**
  Knowing GT allows retaining 100% of all K28 union rescues in just 4 proposal slots. The sole research challenge is therefore GT-free portfolio allocation and selection.

---

## 2. Quantitative Performance Table: Union-K28 vs OracleBest4Of8-K24

| Seed | Metric | K28 Union | OracleBest4Of8 (K=24) | Difference | Retention Rate | Threshold Pass |
|---|---|---|---|---|---|---|
| **Seed 42** | Overall Coverage (%) | 70.47% (1100 / 1561) | **70.47% (1100 / 1561)** | 0.00 pp | **100.00%** | $\ge 95\%$ PASS |
| | Change Coverage (%) | 43.60% (92 / 211) | **43.60% (92 / 211)** | 0.00 pp | **100.00%** | $\ge 95\%$ PASS |
| | Rescues (Overall / Change) | 40 / 5 | 40 / 5 | 0 | 100.00% | PASS |
| **Seed 43** | Overall Coverage (%) | 70.66% (1103 / 1561) | **70.66% (1103 / 1561)** | 0.00 pp | **100.00%** | $\ge 95\%$ PASS |
| | Change Coverage (%) | 43.60% (92 / 211) | **43.60% (92 / 211)** | 0.00 pp | **100.00%** | $\ge 95\%$ PASS |
| | Rescues (Overall / Change) | 43 / 5 | 43 / 5 | 0 | 100.00% | PASS |
| **Seed 44** | Overall Coverage (%) | 70.02% (1093 / 1561) | **70.02% (1093 / 1561)** | 0.00 pp | **100.00%** | $\ge 95\%$ PASS |
| | Change Coverage (%) | 42.65% (90 / 211) | **42.65% (90 / 211)** | 0.00 pp | **100.00%** | $\ge 95\%$ PASS |
| | Rescues (Overall / Change) | 33 / 3 | 33 / 3 | 0 | 100.00% | PASS |
| **Mean** | **Overall Coverage (%)** | **70.38%** | **70.38%** | **0.00 pp** | **100.00%** | **PASS** |
| | **Change Coverage (%)** | **43.29%** | **43.29%** | **0.00 pp** | **100.00%** | **PASS** |

---

## 3. Decision Rule Verification
- `is_budget_sufficient = True` (100.00% retention $\ge 95.0\%$).
- **Triggered Next Step**: Proceed immediately to **Stage P2: Learned Slot Semantic Audit**.
