# RESULT ANALYSIS: STAGE P2 LEARNED SLOT SEMANTIC AUDIT

## 1. Executive Summary & Hungarian Matching Proof
- **Stage**: `PORTFOLIO_P2_SLOT_SEMANTICS` (Seeds 42, 43, 44, Outer $N=1561$, Change $N=211$)
- **Status**: `SUCCESS_CHANGE_SPECIALIZATION_CONFIRMED`
- **Formal Hungarian Matching Results**:
  - Distance cost matrix based on joint parameter profile ($s_{t4}, s_{t8}, s_{t12}, \theta_{t4}, \theta_{t8}, \theta_{t12}, \text{onset}, \text{persistence}$) and trajectory geometry (endpoint radius, curvature):
  - **Seed 42 $\to$ Seed 43**: $\{0 \to 0, 1 \to 1, 2 \to 2, 3 \to 3\}$ (Identity mapping).
  - **Seed 42 $\to$ Seed 44**: $\{0 \to 0, 1 \to 1, 2 \to 2, 3 \to 3\}$ (Identity mapping).
- **Core Conclusion**:
  Slot index 3 across all three seeds is not an artifact or label permuted noise. It reliably and stably converges to the exact same physical motion family: **Deceleration & Full Stop**, capturing $\sim 61\%$ of all Change winners in every single seed.

---

## 2. 4-Slot Semantic Specialization Breakdown Across 3 Seeds

| Slot | Semantic Motion Family | Speed Ratio $s_{t12}$ (Mean Across Seeds) | Cumulative Heading $\theta_{t12}$ | Overall Winner Share | Change Winner Share | Stable Winner Share |
|---|---|---|---|---|---|---|
| **Slot 0** | **Straight Cruise / Accel** | **1.282** (42: 1.28, 43: 1.29, 44: 1.28) | **+0.1°** | **23.9%** | **19.1%** | **24.4%** |
| **Slot 1** | **Left Turn** | **0.998** (42: 1.00, 43: 1.00, 44: 1.00) | **+57.4°** | **23.6%** | **9.2%** | **24.0%** |
| **Slot 2** | **Right Turn** | **0.994** (42: 0.99, 43: 1.00, 44: 0.99) | **-58.1°** | **15.1%** | **10.6%** | **15.5%** |
| **Slot 3** | **Decelerate / Stop (Change)**| **0.331** (42: 0.317, 43: 0.325, 44: 0.350) | **-0.2°** | **37.4%** | **61.1%** | **36.1%** |

---

## 3. Change-Specialized Slot 3 Detailed Profile

- **Motion Family**: Deceleration / Stopping (Mean $s_{t4}=0.74$, $s_{t8}=0.46$, $s_{t12}=0.33$, $\text{onset}=4.8$, $\text{persistence}=2.6$).
- **Change Winner Share**:
  - Seed 42: **61.6%**
  - Seed 43: **61.1%**
  - Seed 44: **60.7%**
- **Rescue Count**:
  - Seed 42: 30 rescues
  - Seed 43: 28 rescues
  - Seed 44: 21 rescues
- **Common Unique Rescue Across All 3 Seeds**: `Sample 6255`.

---

## 4. Decision Rule Verification
- `is_stable_change_specialization_confirmed = True`.
- **Pre-registered Inner/Training Rule for Stage P3**:
  For mixed portfolios retaining $L$ learned slots:
  - If $L=1$: Retain **Slot 3** (the Change-specialized decel/stop slot).
  - If $L=2$: Retain **Slot 3** + **Slot 0** (Decel/Stop + Cruise/Accel, covering longitudinal variance).
  - If $L=3$: Retain **Slot 3** + **Slot 0** + **Slot 1** (Decel/Stop + Cruise/Accel + Left Turn).
- **Triggered Next Step**: Proceed immediately to **Stage P3: Fixed Portfolio K24 Audit**.
