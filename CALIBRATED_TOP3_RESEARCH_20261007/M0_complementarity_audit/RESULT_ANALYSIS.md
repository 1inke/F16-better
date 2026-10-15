# Phase M0: Top-3 Complementarity Audit Analysis

## 1. Core Findings
Across 3 grouped transfer protocols ($N=3,162$) and 3 seeds (42, 43, 44), the zero-training complementarity audit reveals:

1. **Origins of the +8.16 pp Change Gain**:
   - **Learned Slot 3 alone accounts for 80.3% of all Change rescues** (+6.56 pp net rescue).
   - **Static Top-1 accounts for 13.6% of Change rescues** (+1.11 pp net rescue).
   - **Shared overlap between Static and Learned is only 6.0%** (+0.49 pp).
   - **Conclusion**: The two proposals are almost completely non-redundant. Learned-3 provides the primary engine of behavioral change rescue, while Static captures a distinct subset of standardized deceleration/maneuver profiles.

2. **Overall Population Support**:
   - Native Top-1 alone: **30.93%**.
   - Top-3 Portfolio Set: **34.95%** (+4.03 pp gain).
   - Overall minADE drops from 0.419m to **0.354m** (a **+15.4% improvement**).

3. **Geometric Diversity and Mode Separation**:
   - The average pairwise trajectory distance among the 3 candidates is **0.409m**.
   - The expected number of geometrically distinct modes (endpoint distance $>0.1\text{m}$) is **2.77 out of 3.0**, proving that the portfolio does not collapse into clustered redundant hypotheses.

4. **Distribution of #Good Trajectories**:
   - 0 of 3 good: 65.05%
   - 1 of 3 good: 28.53%
   - 2 of 3 good: 5.72%
   - 3 of 3 good: 0.70%
   - Over 81% of successful cases have *exactly one* good candidate, demonstrating strong mode specialization among the three slots.

## 2. Gate Decision & Next Phase
Phase M0 confirms that the fixed `[Native, Static, Learned-3]` composition is highly effective and complementary. We proceed to:
**Phase M1: Top-3 Probability Calibration** (learning $P(\text{mode}_i \mid \text{history})$ over the 3 hypotheses).
