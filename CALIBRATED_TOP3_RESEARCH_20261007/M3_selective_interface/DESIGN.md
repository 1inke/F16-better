# Phase M3: Selective Interface Design

## Objective
Design and evaluate a three-tiered deployment interface uniting probability, entropy, and confidence:
- **Tier 1 (Single Top-1 Output)**: For low-uncertainty, high-confidence windows ($p_N \ge 0.60$ and $u(h) < \text{median}$), output single Native Top-1.
- **Tier 2 (Calibrated Top-3 Portfolio)**: For multimodal, medium-uncertainty windows, output calibrated 3-hypothesis portfolio $(\tau_1, p_1), (\tau_2, p_2), (\tau_3, p_3)$.
- **Tier 3 (Abstention / Safety Warning)**: For extreme uncertainty ($u(h) \ge 90\text{th}$ percentile), abstain and issue safety warning.

## Evaluation Metrics
- Allocation fractions across the 3 tiers.
- Individual tier performance (Tier 1 Accuracy, Tier 2 Coverage, Tier 3 Baseline Accuracy).
- Overall system retained accuracy and expected hypothesis count per window.
