# Stage R3: Representation / Ambiguity Audit Analysis

## 1. Core Scientific Question
> For very similar past representations ($P2$), is the ground truth future motion mode and best candidate identity highly ambiguous/multimodal, particularly in Change cases?

## 2. Quantitative Evidence
Across 3 grouped transfer protocols ($N=3162$ evaluated outer windows, $k=20$ nearest neighbors in training history space):

1. **Massive Increase in Future Direction and Mode Ambiguity**:
   - **Future Direction Circular Variance**: 0.177 in Change vs 0.076 in Stable (**2.32x higher** in Change).
   - **Future Mode Entropy**: 1.177 bits in Change vs 0.739 bits in Stable (**1.59x higher** in Change).
   - **Speed Profile Variance**: 0.0409 in Change vs 0.0293 in Stable (**1.39x higher** in Change).

2. **Past Representation Ambiguity Predicts Top-1 Failure**:
   - The correlation between past neighbor entropy and Top-1 selection failure is consistently strong positive:
     - Scene-Family: $r = +0.333$
     - Target-Track: $r = +0.456$
     - Recording-Time: $r = +0.422$
     - Macro Mean: $r = +0.404$.

3. **Multi-Hypothesis Portfolio vs Single Deterministic Top-1**:
   - Because future motion exhibits true multimodal branches given identical past trajectories, forcing a single deterministic Top-1 output creates an unavoidable trade-off between false-alarms on subtle maneuvers and missed detections on drastic maneuvers.
   - In contrast, retaining a calibrated Top-3 portfolio (Native Top-1 + Static Top-1 + Learned-3) achieves:
     - Scene-Family: 42.47% (+2.24 pp)
     - Target-Track: 31.88% (+4.25 pp)
     - Recording-Time: 29.86% (+4.95 pp)
     - Macro Gain: **+3.81 pp**.

4. **Confidence-Aware Abstention**:
   - Abstaining on the top 20% most ambiguous windows increases retained Top-1 accuracy by **+5.62 pp** macro mean (+6.93 pp in Scene-Family, +5.26 pp in Target-Track, +4.67 pp in Recording-Time). The abstained windows had an average good rate of only 8.2%.

## 3. Rigorous Research Conclusion
- **OBSERVATION**: Change samples have 2.32x higher future angular dispersion and 1.59x higher mode entropy given near-identical past history representations.
- **INTERPRETATION**: Past-only candidate rankability for single deterministic Top-1 selection is fundamentally constrained by high conditional entropy during behavioral change regimes.
- **NOT PROVEN**: We do NOT claim that pedestrian behavior is fundamentally unpredictable in the real world; rather, past-only focal trajectory representations cannot deterministically collapse future multimodality before the physical maneuver onset.
- **DECISION**: `PAST_ONLY_SELECTION_AMBIGUITY_CONFIRMED`.
