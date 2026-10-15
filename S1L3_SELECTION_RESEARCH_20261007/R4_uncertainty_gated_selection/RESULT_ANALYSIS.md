# Stage R4: Uncertainty-Gated Selection Analysis

## 1. Context & Motivation
Following the discovery in Stage R0 and R2 that the Candidate-Aware (R2) scorer yields strong positive gains on the Change subgroup (+2.12 pp) and unlocks NewSupport proposals (Recall@1 = 27.8%) but causes severe ADE degradation on the overall population (-18.23% harm), Stage R4 implemented an **Uncertainty-Gated Selective Router**:
$$S(h) = \begin{cases} S_{\text{native}}, & u(h) < \tau \\ S_{R2}, & u(h) \ge \tau \end{cases}$$
The objective was to determine whether deployment-time, ground-truth-free uncertainty signals can restrict R2 overrides exclusively to hard/change cases without violating the overall ADE degradation threshold ($\le 1.0\%$).

---

## 2. Quantitative Results

### R4A — Override Predictability
- **Predictability Target**: $Y_{\text{override}} = \mathbf{1}[J_{R2} < J_{\text{Native}}]$.
- **Composite Gate AUROC**:
  - Scene-Family: 0.771
  - Target-Track: 0.739
  - Recording-Time: 0.720
  - **Macro AUROC: 0.743**.
- **Finding**: Ground-truth-free features (P2 kNN mode entropy, native logit margin, Top-1 disagreement, regime change probability) possess solid predictive power (AUROC ~0.74) for identifying when R2 outperforms the Native scorer.

### R4B — Selective Routing Gate Trade-off
Across the override sweep $f \in [0.00, 1.00]$, the macro trade-off between Change gain and ADE harm is monotonic:

| Override Fraction $f$ | Change Top-1 Gain (pp) | Overall ADE Harm (%) | Overall Top-1 Gain (pp) | Severe Regret Drop (pp) | Protocols Passing Gate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0% (Pure Native)** | 0.00 | 0.00% | 0.00 | 0.00 | 0/3 |
| **5%** | +0.32 | **+1.32%** | +0.01 | -0.08 | **0/3** |
| **10%** | +0.78 | **+3.15%** | +0.02 | -0.27 | **0/3** |
| **15%** | +0.85 | **+4.71%** | -0.10 | -0.51 | **0/3** |
| **20%** | +1.04 | **+6.27%** | -0.25 | -0.84 | **0/3** |
| **50%** | +2.11 | **+15.26%** | -0.69 | -1.53 | **0/3** |
| **100% (Pure R2)** | +2.31 | **+18.60%** | -1.16 | -2.39 | **0/3** |

- **GATE VERDICT**: **FAIL (0/3 protocols)**.
- **Physical Root Cause**:
  Even when guided by a 0.74 AUROC uncertainty gate, when the model selects an aggressive R2 proposal (e.g., sharp stop or turn) on a case that turns out to be a subtle native continuation, the displacement penalty on that individual sample is massive ($\Delta \text{ADE} \approx 0.5 - 1.0\text{ m}$). Averaged across the population, even a 5% override fraction incurs +1.32% overall ADE harm, breaching the 1.0% limit before significant Change gains (+0.32 pp) can accumulate.

### R4C — Selective Prediction (Abstention)
- Abstaining on the 20% most uncertain cases improves retained Top-1 good rate by:
  - Scene-Family: **+4.29 pp**
  - Target-Track: **+1.67 pp**
  - Recording-Time: **+3.64 pp**
- Overall AURC: **0.261**; Change AURC: **0.350**.
- **Finding**: Uncertainty ordering is highly calibrated for identifying when to refrain from single deterministic predictions.

### R4D — Track B: Top-3 Multi-Hypothesis Portfolio
When evaluated as a compact multi-hypothesis set `[Native Top-1, Static Top-1, Learned-3]`:
- **Overall Set Good Rate**: **34.95%** (Gain vs Native Top-1: **+4.03 pp**)
- **Change Set Good Rate**: **19.70%** (Gain vs Native Top-1: **+8.16 pp**!)
- **Expected Effective Set Size**: **2.77 / 3.0** distinct candidates.
- **Finding**: Multi-hypothesis output completely sidesteps the single Top-1 false-alarm dilemma, capturing over +8.1 pp of Change support without any degradation to conservative paths.

---

## 3. Definitive Scientific Conclusion
1. **OBSERVATION**: The uncertainty gate achieves 0.743 AUROC in predicting R2 advantage, but cannot prevent false alarm ADE harm from exceeding 1% even at a 5% override rate (+1.32% harm for +0.32 pp Change gain).
2. **INTERPRETATION**: Under past-only focal trajectory representations, behavioral change regimes are intrinsically multimodal. Forcing a single deterministic Top-1 decision creates an unavoidable Pareto trade-off between Change recall and overall displacement precision.
3. **DECISION**: As prescribed by the pre-registered protocol, Stage R4 officially establishes:
   $$\boxed{\text{PAST\_ONLY\_TOP1\_LIMITED}}$$
   Single Top-1 selection research has reached its empirical ceiling. The research program should formally transition to **calibrated multi-hypothesis prediction (Track B)** and **selective forecasting (Track C)**.
