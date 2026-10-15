# DESIGN: Stage P0 Static4 + Learned4 Complementarity Audit (Diagnostic K28 Union)

## 1. Scientific Objective
- **Problem**: In Stage B, replacing Static4 with Learned4 improved Overall coverage (+1.39 pp) and minADE/minFDE, but yielded a net zero gain on Change (mean +0.00 pp across 3 seeds). Since Native20 is identical, rescues represent cases where Learned4 succeeds but Static4 fails, while losts represent cases where Static4 succeeds but Learned4 fails.
- **Hypothesis**: Static4 (rule-based heuristic proposals) and Learned4 (data-driven Best-of-4 proposals) provide complementary candidate coverage. The K28 diagnostic union (Native20 + Static4 + Learned4) establishes the diagnostic upper bound of their combined coverage.
- **Strict Invariance**: Zero training. Native20 and Static4 strictly frozen. Best-of-4 generator checkpoints frozen (Seeds 42, 43, 44).
- **Hard Assertion**:
  `union_coverage >= static_k24_coverage` must hold identically for every seed.
  Relative to Static-K24, lost must be identically 0. If lost > 0, immediate halt with DATA_ALIGNMENT_ERROR.

## 2. Experimental Configuration
- **Dataset**: Zara scene-family protocol (Outer $N=1,561$, Change $N=211$, Stable $N=1,226$, Transition $N=124$).
- **Candidate Sets**:
  - Control: Static-K24 (Native20 + Static4)
  - Treatment: Bo4-K24 (Native20 + Learned4)
  - Diagnostic Union: Union-K28 (Native20 + Static4 + Learned4)
- **Seeds**: 42, 43, 44.
- **Outputs**:
  - Metrics across Overall, Stable, Transition, Change, and Clear Event.
  - Set-theoretic breakdown: Static4-only, Learned4-only, Static4 $\cap$ Learned4, Static4 $\cup$ Learned4.
  - Prefix errors (t4, t8, t12) and candidate motion statistics.
