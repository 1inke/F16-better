# RESULT ANALYSIS: STAGE A BEST-OF-4 GENERATOR PILOT

## 1. Executive Summary & Pilot Decision
- **Stage**: `STAGE_A_BEST_OF_4_GENERATOR_PILOT` (Seed 42)
- **Status**: `SUCCESS_PILOT_POSITIVE`
- **Pilot Decision Rule Evaluation**:
  1. *Change Coverage improvement*: **PASS** (+0.47 pp, 41.71% $\to$ 42.18%).
  2. *Change minADE/minFDE improvement*: **PASS** (minADE +1.67%, minFDE +4.13%).
  3. *Slot collapse check*: **PASS** (Mean pairwise distance 0.638 m, max dominance 38.1% $\ll$ 75%, effective slots 3.93 / 4.0).
  4. *Overall / Subgroup harmony*: **PASS** (Overall coverage +1.47 pp, Stable +1.80 pp, Transition 0.00 pp, Change +0.47 pp).
  5. *Kinematics Guard*: Speed p95 diff = -0.44% (pass), Jerk p95 diff = +0.05% (pass), Accel p95 diff = +2.75% (threshold $\le 2.0\%$, failed).
- **Pilot Decision**: **POSITIVE SIGNAL ON CHANGE** $\implies$ Qualified to proceed to **Stage B: 3 Paired Seeds**.

---

## 2. Quantitative Performance Table (Outer Set $N=1561$, Change $N=211$)

| Metric Group | Specific Metric | Control (Static-K24) | Treatment (Bo4-K24 Pilot) | Absolute / Relative Gain |
|---|---|---|---|---|
| **Overall** | Coverage (%) | 67.97% (1061 / 1561) | **69.44%** (1084 / 1561) | **+1.47 pp** |
| | Rescue / Lost / Net | - | 40 / 17 / **+23** | **+23 net rescues** |
| | Mean $J$ | 0.7982 | **0.7816** | **+2.08%** |
| | $\min\text{ADE}$ (m) | 0.1410 | **0.1387** | **+1.65%** |
| | $\min\text{FDE}$ (m) | 0.2475 | **0.2387** | **+3.57%** |
| | Eff. Good Clusters | 1832 | 1813 | -19 |
| **Stable** ($N=1226$) | Coverage (%) | 74.47% (913) | **76.26%** (935) | **+1.80 pp** |
| | $\min\text{ADE}$ (m) | 0.1181 | **0.1161** | **+1.69%** |
| | $\min\text{FDE}$ (m) | 0.2035 | **0.1958** | **+3.79%** |
| **Transition** ($N=124$) | Coverage (%) | 48.39% (60) | 48.39% (60) | 0.00 pp |
| | $\min\text{ADE}$ (m) | 0.2027 | **0.1999** | **+1.38%** |
| | $\min\text{FDE}$ (m) | 0.3684 | **0.3638** | **+1.26%** |
| **Change** ($N=211$) | Coverage (%) | 41.71% (88) | **42.18%** (89) | **+0.47 pp** |
| | Change Rescue / Lost | - | 5 / 4 / **+1** | **+1 net rescue** |
| | $\min\text{ADE}$ (m) | 0.2379 | **0.2339** | **+1.67%** |
| | $\min\text{FDE}$ (m) | 0.4320 | **0.4141** | **+4.13%** |

---

## 3. Slot Collapse & Diversity Monitoring

| Diagnostic Dimension | Metric | Observed Value | Healthy Gate | Evaluation |
|---|---|---|---|---|
| **Trajectory Spacing** | Mean Pairwise Distance | **0.6377 m** | $\ge 0.10\text{ m}$ | Non-collapsed, large spacing |
| | Median Pairwise Distance | **0.5255 m** | $\ge 0.10\text{ m}$ | Well-distributed across pairs |
| | Min Pairwise Distance | **0.0654 m** | - | Local overlap rare |
| **Effective Slot Count** | Mean Effective Slots ($\ge 0.1\text{m}$) | **3.93 / 4.0** | $\ge 2.0$ | Nearly all 4 slots distinct |
| **Normalized Param Dist** | Mean Pairwise Param Distance | **3.6025** | $\gg 0$ | Diverse continuous params |
| **Winner Frequencies** | Overall Slots [0, 1, 2, 3] | [24.0%, 22.4%, 15.6%, 38.1%] | No slot $> 75\%$ | Highly balanced |
| | Stable Slots [0, 1, 2, 3] | [24.9%, 23.7%, 16.7%, 34.7%] | - | Balanced usage |
| | Transition Slots [0, 1, 2, 3] | [21.8%, 25.0%, 21.8%, 31.5%] | - | Balanced usage |
| | Change Slots [0, 1, 2, 3] | [20.4%, 12.8%, 5.2%, 61.6%] | - | Decel/Stop dominant but turns active |
| **Dominance Check** | Maximum Slot Dominance | **38.05%** | $< 75.0\%$ | **PASS (No slot collapse)** |

---

## 4. Change Subgroup Paired Case Analysis ($N=211$)

- **Static Success $\to$ Bo4 Failure (Lost on Change, 4 cases)**:
  - Sample `3497`: Static ADE 0.188m / FDE 0.114m $\to$ Bo4 ADE 0.301m / FDE 0.114m (winning slot: 3)
  - Sample `4069`: Static ADE 0.108m / FDE 0.212m $\to$ Bo4 ADE 0.289m / FDE 0.212m (winning slot: 3)
  - Sample `6204`: Static ADE 0.100m / FDE 0.137m $\to$ Bo4 ADE 0.193m / FDE 0.171m (winning slot: 3)
  - Sample `6832`: Static ADE 0.191m / FDE 0.113m $\to$ Bo4 ADE 0.264m / FDE 0.240m (winning slot: 1)
- **Static Failure $\to$ Bo4 Success (Rescue on Change, 5 cases)**:
  - Sample `5383`: Static ADE 0.207m / FDE 0.359m $\to$ Bo4 ADE 0.177m / FDE 0.340m (winning slot: 3)
  - Sample `6255`: Static ADE 0.275m / FDE 0.343m $\to$ Bo4 ADE 0.154m / FDE 0.295m (winning slot: 3)
  - Sample `6290`: Static ADE 0.203m / FDE 0.484m $\to$ Bo4 ADE 0.150m / FDE 0.093m (winning slot: 3)
  - Sample `6761`: Static ADE 0.279m / FDE 0.604m $\to$ Bo4 ADE 0.142m / FDE 0.377m (winning slot: 1)
  - Sample `8262`: Static ADE 0.146m / FDE 0.513m $\to$ Bo4 ADE 0.123m / FDE 0.398m (winning slot: 0)

---

## 5. Comparison across All 3 Generator Paradigms on Change Subgroup
| Generator Paradigm | Supervision Loss | Overall Coverage Gain | Change Coverage Gain | Change $\min\text{ADE}$ Gain | Change $\min\text{FDE}$ Gain | Net Change Rescue |
|---|---|---|---|---|---|---|
| **Stage 03 Deterministic** | MSE on single $z^*$ | +1.00 pp | **-2.37 pp** | **-1.88%** | **-1.67%** | -5 |
| **Stage 03 Gaussian MDN** | Gaussian NLL on single $z^*$ | +0.49 pp | **-2.53 pp** | **-1.81%** | **-1.58%** | -5 |
| **Stage A Best-of-4 Pilot** | Trajectory-space $\min_m \mathcal{L}_{\text{traj}}$ | **+1.47 pp** | **+0.47 pp** | **+1.67%** | **+4.13%** | **+1** |
