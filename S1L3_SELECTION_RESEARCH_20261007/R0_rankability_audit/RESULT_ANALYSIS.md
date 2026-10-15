# Stage R0: Frozen Scorer Baseline, Rankability & Score Bias Audit Report

## 1. Executive Summary & Diagnostic Findings
1. **Selection Conversion Efficiency = 0.00%**:
   - Across all three protocols (Scene-Family, Target-Track, Recording-Time) and all 3 seeds, the frozen native scorer yields exactly **0.00% Selection Gain** despite positive Generation Gains (+1.28 pp to +2.11 pp).
   - In 100% of NewSupport windows (samples where S1L3 uniquely generates good candidates), the native scorer fails to select them.
2. **Provenance Score Bias = PROVENANCE_SCORE_BIAS_PRESENT**:
   - Under quality-matched stratification ($J \le 1.0$), Native candidates receive a mean logit of **+2.861**, whereas Learned candidates receive a mean logit of **+0.148** (a massive 2.7-point systemic deficit).
   - The median rank of successful NewSupport candidates is **18.0** ($> 10$).
3. **Discriminative Signal in Representation**:
   - Lightweight history-candidate probes achieve **92.0% - 94.0% pairwise accuracy** on distinguishing good vs bad candidates, demonstrating that the past representation contains rich signal to score candidate quality.
4. **State Machine Branch Decision**:
   - Criteria for Branch B (`PROVENANCE_BIAS_DOMINANT`) are fully met:
     - Learned candidates systematically lower-scored under matched quality.
     - NewSupport good candidates frequently rank $> 10$ (median rank = 18.0).
   - **Branch Output**: **`PROVENANCE_BIAS_DOMINANT`** $\implies$ **Stage R2 — Candidate-Aware Scorer**.
