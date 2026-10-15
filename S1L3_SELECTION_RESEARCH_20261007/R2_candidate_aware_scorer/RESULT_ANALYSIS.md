# Stage R2: Candidate-Aware Scorer Analysis

## 1. Key Findings

### Positive Evidence
1. **Unlocking NewSupport Proposals**: Under the frozen native scorer, NewSupport candidates had **0.0% Recall@1** and **0.0% Recall@5** due to the static logit penalty prior. The Candidate-Aware Scorer raised NewSupport Recall@1 to **27.83%** (macro mean) and Recall@5 to **84.38%** across all 3 protocols.
2. **Consistent Change Subgroup Improvement**: Change Top-1 Good Rate increased in 8 of 9 protocol×seed runs:
   - Scene-Family Change: +1.11 pp
   - Target-Track Change: +2.90 pp
   - Recording-Time Change: +2.35 pp
   - Macro Change Gain: **+2.12 pp**.
3. **Severe Regret Reduction**: Severe regret dropped by an average of -2.30 pp across all runs.

### Negative Evidence (Gate Breaches)
1. **Overall Top-1 Good Rate Harm**:
   - Scene-Family: -2.09 pp
   - Target-Track: -0.68 pp
   - Recording-Time: +0.41 pp
   - Protocol Macro: **-0.79 pp**. Only 1 of 3 protocols showed positive macro gain, violating the "$\ge 2/3$ protocols" requirement.
2. **Selected Error Distortion**:
   - Overall ADE degraded by -16.07% (Scene-Family), -20.56% (Target-Track), -18.06% (Recording-Time), averaging **-18.23% harm** (gate threshold $\le 1\%$).
   - Reason: When the model actively promotes proposals to capture difficult Change maneuvers, it incurs high false-alarm costs on ambiguous Stable/Transition cases where a conservative native candidate was closer to ground truth.

## 2. Gate Decision
- Verdict: **FAIL (Gate breached on Overall Top-1 Good Rate and Selected ADE/FDE Harm)**.
- In accordance with the pre-registered research state machine:
  $$\text{R2 Fail} \implies \text{Proceed to Stage R3 (Representation / Ambiguity Audit)}$$
