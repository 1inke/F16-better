# Phase M3: Selective Interface Analysis

## 1. Multi-Tiered System Performance
The three-tiered selective interface demonstrates how probabilistic calibration and uncertainty awareness integrate seamlessly into real-world robotic/AV decision stacks:

1. **Tier 1 (High-Confidence Deterministic Forecasting)**:
   - Covers **26.9% of windows** (simple, high-confidence straight continuations).
   - Achieves **37.16% Top-1 accuracy** while emitting only **1.0 trajectory**.

2. **Tier 2 (Calibrated Multi-Hypothesis Portfolio)**:
   - Covers **62.7% of windows** (multimodal transition regimes).
   - Emits the 3-hypothesis portfolio with calibrated probabilities $[( \tau_N, p_N ), ( \tau_S, p_S ), ( \tau_L, p_L )]$.
   - Achieves **34.31% set coverage** without incurring catastrophic single-choice false alarms.

3. **Tier 3 (Safety Abstention / Warning)**:
   - Flags the **top 10.5% highest entropy/risk windows**.
   - These windows exhibit extremely low baseline predictability, meaning abstaining protects the downstream planning system from blind, uncalibrated collisions.

4. **Overall System Efficiency**:
   - The system achieves an overall retained accuracy of **36.33%** while requiring an expected average of only **2.15 hypotheses per window** (compared to $K=24$ or $K=20$).
