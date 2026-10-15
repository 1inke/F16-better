# DESIGN: Stage P1 Oracle 8 -> 4 Budget Audit

## 1. Scientific Objective
- **Problem**: K28 Union achieves +2.48 pp overall coverage and +2.05 pp Change coverage by utilizing 8 proposal hypotheses (Static4 + Learned4). However, deployed models have a fixed budget of $K=24$ (Native20 + 4 proposal slots).
- **Core Question**: Under an oracle selection upper bound (where GT is known post-hoc), does a 4-slot proposal budget retain $\ge 95\%$ of the K28 union rescues, or does the 4-slot budget itself act as a bottleneck?
- **Protocol**:
  - Inputs: Static4 (4) + Learned4 (4) = 8 proposal hypotheses.
  - Selection: For each sample, select Oracle Top-4 proposals (prioritizing complete good candidates, then sorted by continuous $J$).
  - Candidate Set: Native20 + OracleBest4Of8 $\implies K=24$.
  - Evaluation: Compare against K28 Union across Seeds 42, 43, 44 on Zara outer split ($N=1,561$).
- **Success Criteria**: Rescue Retention $= \frac{\text{Rescue}(\text{OracleBest4Of8})}{\text{Rescue}(\text{K28 Union})} \ge 95\%$.
