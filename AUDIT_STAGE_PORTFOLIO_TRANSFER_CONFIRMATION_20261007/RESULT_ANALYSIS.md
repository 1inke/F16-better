# Portfolio Transfer Confirmation Suite: Comprehensive Analysis Report

## 1. Executive Summary & Core Decisions
- **Phase 1: Baseline Reconciliation**: PASS.
  - Successfully reconciled the 87 vs 88 vs 89 Change covered windows on Scene-Family Outer.
  - Demonstrated that the difference was caused by unseeded initial weights of `IntentClassifier` in ad-hoc runs flipping the 4th/5th rank on boundary sample ID `6290` (`stop` vs `accelerate`).
  - Audited and established the Canonical Static-K24 baseline ($N=1561$, Overall: 67.97% [1061], Change: 41.71% [88]), strictly matching the formal transfer gate and Best-of-4 benchmark.
- **Phase 2 & 3: Cross-Protocol Transfer Confirmation**:
  - Evaluated **F1 (S1L3)** and **F2 (S2L2)** across three strictly isolated grouped regimes:
    1. Scene-Family Outer ($N=1561$, Scene-Family Unseen)
    2. Target-Track Outer ($N=1035$, Pedestrian-ID Unseen)
    3. Recording-Time Outer ($N=566$, Time-Block Unseen)
- **Transfer Gate Outcomes**:
  - **S1L3**: `S1L3_TRANSFER_PASS` (Passed all 6 gate criteria).
  - **S2L2**: `S2L2_TRANSFER_PASS` (Passed all 6 gate criteria).
- **Final Deployed Portfolio Decision**:
  - Following preregistered Situation A rules, **`FREEZE_S1L3_GENERATOR`** is selected.
  - Rationale: S1L3 achieved higher macro Change gain (+1.49 pp vs +1.25 pp), higher macro Overall gain (+0.66 pp vs +0.37 pp), superior positive seed consistency (8/9 positive runs on Change, 1 neutral, 0 negative), and strictly satisfied all motion and degradation guards.

---

## 2. Macro Performance Summary Table (3 Protocols, 3 Seeds = 9 Evaluation Runs)

| Protocol | Split Size ($N$) | Canonical Static Change Cov (%) | S1L3 Change Cov (%) | S1L3 Change Gain (pp) [Positive Seeds] | S2L2 Change Cov (%) | S2L2 Change Gain (pp) [Positive Seeds] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scene-Family** | 1,561 (Change: 211) | 41.71% (88) | **43.13%** (91.0) | **+1.42 pp** [3 / 3] | **43.13%** (91.0) | **+1.42 pp** [3 / 3] |
| **Target-Track** | 1,035 (Change: 253) | 43.48% (110) | **45.59%** (115.3) | **+2.11 pp** [3 / 3] | **45.59%** (115.3) | **+2.11 pp** [3 / 3] |
| **Recording-Time**| 566 (Change: 142) | 43.66% (62) | **44.60%** (63.3) | **+0.94 pp** [2 / 3] | **43.90%** (62.3) | **+0.23 pp** [1 / 3] |
| **Macro Mean** | **3,162 (Change: 606)** | **42.95%** | **44.44%** | **+1.49 pp** [**8 / 9**] | **44.20%** | **+1.25 pp** [**7 / 9**] |
