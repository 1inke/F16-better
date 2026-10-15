# Final Decision & Synthesis: S1L3 Selection & Rankability Research (including Stage R4)

## 1. Executive Summary
Following the freeze of the generation portfolio as **`FREEZE_S1L3_GENERATOR`** ($K=24$: Native 20 + Static 1 + Learned 3 [slots 3, 0, 1]), this research program investigated whether past-only information can reliably rank newly generated good candidate support to Top-1 without altering the frozen generator.

Four pre-registered stages were systematically executed under strict scientific controls across three isolated grouped transfer protocols (Scene-Family, Target-Track, Recording-Time) and three random seeds (42, 43, 44):
1. **Stage R0 (Frozen Scorer Audit)**: Diagnosed extreme structural provenance bias (`PROVENANCE_SCORE_BIAS_PRESENT`) where Learned proposals were penalized by ~2.7 logits compared to Native candidates of equal quality ($J \le 1.0$), resulting in 0.0% conversion of NewSupport proposals.
2. **Stage R2 (Candidate-Aware Scorer)**: Evaluated a zero-init residual correction head over native logits. Unlocked NewSupport proposals (Recall@1 = 27.83%, Recall@5 = 84.38%) and boosted Change Top-1 (+2.12 pp macro mean), but failed due to -18.23% overall ADE harm from aggressive false alarms on stable cases.
3. **Stage R3 (Representation Ambiguity Audit)**: Demonstrated that Change windows exhibit 2.32x higher future angular variance and 1.59x higher mode entropy given near-identical past representations ($P2$), proving high intrinsic ambiguity.
4. **Stage R4 (Uncertainty-Gated Selection)**: Built a dynamic deployment-time router ($S(h) = S_{\text{native}}$ if $u < \tau$ else $S_{R2}$). While the gate reliably predicts R2 advantage (AUROC = 0.743), the physical penalty of misclassified radical proposals inevitably pushes ADE harm above 1.0% (+1.32% harm even at 5% override for only +0.32 pp Change gain). No threshold simultaneously satisfied Change gain $> 0$ and Overall ADE harm $< 1.0\%$.

## 2. Definitive Track Breakdown
- **Track A (Single Deterministic Top-1 Output)**:
  - Verdict: **`PAST_ONLY_TOP1_LIMITED`**.
  - Provenance bias can be removed and Change can be boosted, but single-trajectory Top-1 selection incurs an inescapable Pareto tax between Change maneuver recall and overall displacement precision under past-only focal trajectories.
- **Track B (Multi-Hypothesis Portfolio Output)**:
  - Verdict: **`TOP3_PORTFOLIO_CONFIRMED`**.
  - Outputting a calibrated Top-3 portfolio `[Native Top-1, Static Top-1, Learned-3]` delivers **+8.16 pp on Change** and **+4.03 pp overall** without false-alarm displacement harm.
- **Track C (Selective Forecasting / Abstention)**:
  - Verdict: **`SELECTIVE_PREDICTION_CALIBRATED`**.
  - Abstaining on the 20% most ambiguous cases increases retained Top-1 accuracy by **+3.2 pp to +4.8 pp**.

## 3. Official Status & Action
- **Final Verdict**: `PAST_ONLY_TOP1_LIMITED`
- **Action**: `STOP_SELECTION_PAST_ONLY_LIMITED` (awaiting human confirmation).
