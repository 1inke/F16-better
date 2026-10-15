# Stage 04A Primary: Proposal-Conditioned Correction Head Results & Decision Audit

## 1. Executive Summary & Experimental Results

Stage 04A-P tested whether conditioning a shared residual correction head on explicit maneuver proposal hypotheses ($type_k \in \{0..8\}$ via FiLM) improves shared conditional mapping on the Zara scene-family outer evaluation set ($N=1,561$).

### 1.1 Complete Three-Reference Numerical Table (Zara Outer $N=1,561$)
| Model / Arm | Seed | Zara Outer Covered ($N$) | Zara Outer Rate (%) | $\Delta_{TC}$ vs Control (pp) | $\Delta_{TS}$ vs Static K24 (pp) | Lost vs Static ($N$) | Rescued vs Static ($N$) | Joint Error $J$ | Accel p95 (m/s$^2$) | Jerk p95 (m/s$^3$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static K24** | — | **1,056** | **67.65%** | — | — | 0 | 0 | 0.7942 | 1.428 | 2.981 |
| **Control Head** | 1 | 1,093 | 70.02% | Ref | +2.37 pp | 81 | 118 | 0.7712 | 0.534 | 2.910 |
| | 2 | 1,089 | 69.76% | Ref | +2.11 pp | 108 | 141 | 0.7735 | 0.541 | 2.925 |
| | 3 | 1,079 | 69.12% | Ref | +1.47 pp | 104 | 127 | 0.7758 | 0.538 | 2.919 |
| *Control Mean* | | **1,087.0** | **69.63%** | Ref | **+1.99 pp** | 97.7 | 128.7 | **0.7735** | 0.538 | 2.918 |
| **Treatment Head** | 1 | 1,095 | 70.15% | +0.13 pp | +2.50 pp | 82 | 121 | 0.7681 | 0.535 | 2.908 |
| | 2 | 1,092 | 69.96% | +0.19 pp | +2.31 pp | 109 | 145 | 0.7699 | 0.540 | 2.922 |
| | 3 | 1,074 | 68.80% | -0.32 pp | +1.15 pp | 106 | 124 | 0.7725 | 0.539 | 2.915 |
| *Treatment Mean* | | **1,087.0** | **69.64%** | **-0.00 pp** | **+1.99 pp** | 99.0 | 130.0 | **0.7702** | 0.538 | 2.915 |

---

## 2. Scientific Analysis & Guardrail Assessment

### 2.1 Treatment vs Control ($\Delta_{TC} = -0.00\text{ pp}$)
- **Finding**: While 2 out of 3 seeds showed minor positive trends (Seed 1: +0.13 pp, Seed 2: +0.19 pp), Seed 3 showed a minor negative trend (-0.32 pp). The 3-seed pooled mean delta is essentially zero ($-0.0000\text{ pp}$).
- **Interpretation**: Under the current common feature representation, the correction head trunk is predominantly driven by `state32` and observation features, while the FiLM proposal-type embedding modulation produces negligible additional discrimination. Proposal conditioning alone did not resolve the conditional mapping shift.

### 2.2 Treatment vs Static K24 ($\Delta_{TS} = +1.99\text{ pp}$) & The "Over-Edit / Lost" Guardrail
- Both Control and Treatment achieved positive net coverage gains over Static K24 on Zara outer (+1.99 pp, from 1,056 to 1,087).
- However, as anticipated in `ana.md` §4, **evaluating lost candidates against Static K24 revealed substantial over-editing**:
  - Across seeds, the learned correction heads ruined **82 to 109 previously successful Static K24 windows** (`Lost_vs_Static`).
  - Although 121 to 145 previously failing windows were rescued by correction, the destruction of ~100 viable native/static candidates indicates that the learned correction is not purely additive and causes collateral damage to stable trajectories.

---

## 3. Decision Engine Transition
In accordance with the preregistered state machine:
- Condition: $\Delta_{TC} \approx 0$ (no robust proposal-conditioning gain above control).
- Rule in `LONG_RUN_MASTER_PLAN.md` §Stage 04A and `decision_engine.py` line 44:
  $$\text{tc} \le 0 \implies \text{Transition to } \mathbf{04B1\_COMMON\_SUPPORT}$$
- **Automated Engine Output**:
  `{"status": "DECIDED", "from_stage": "04A_PRIMARY", "next_stage": "04B1_COMMON_SUPPORT", "reason": "no robust proposal-conditioning gain"}`
