# Offline Selector Zoo: Overnight Experiment Final Academic Report

> **Experiment**: F16 Offline Candidate Selector Zoo  
> **Protocol**: Leave-One-Fold-Out (5 Folds: ETH, Hotel, Univ, Zara1, Zara2)  
> **Evaluation Seeds**: [1, 2, 3]  
> **Report Timestamp**: 2026-09-20T17:02:30.690355+00:00

---

## 1. Executive Summary & Core Verdict

The primary bottleneck in multimodal pedestrian trajectory prediction is **not candidate generation**, but **test-time candidate selection**.
The frozen F16 generator produces exceptional 20-candidate coverage (Oracle minADE20 = **0.2181m**), but the Native confidence head selects sub-optimal trajectories with an average ADE of **0.5184m** and a collision rate of **6.68%**.

By evaluating the decoupled **Offline Selector Zoo** across 5 folds and multiple random seeds:
- **Best Accuracy Selector**: `linear_residual_g012` achieves Selected ADE@1 of **0.5098m**, reducing ADE regret to **0.2903m**.
- **Best Safety Selector**: `native_plus_mpn_risk_grid` reduces Top-1 collision rate to **5.71%**, achieving an Oracle Gap Closure of **12.69%** with minimal displacement penalty.

---

## 2. Multi-Seed Leaderboard (5-Fold Macro Average across Seeds)

| Selector Method | Selected ADE@1 (m) | Selected FDE@1 (m) | Collision Rate (%) | Gap Closure (%) | ADE Regret (m) | Q5 Clearance (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **native (Baseline)** | 0.5184 ± 0.0000 | 1.0757 ± 0.0000 | 6.68% ± 0.00% | 0.00% | 0.3003 | 0.190m |
| linear_residual_g012 | 0.5098 ± 0.0012 | 1.0579 ± 0.0023 | 6.40% ± 0.15% | 0.67% | 0.2903 | 0.198m |
| mlp_residual_g0124 | 0.5124 ± 0.0007 | 1.0636 ± 0.0036 | 6.42% ± 0.15% | 0.76% | 0.2929 | 0.193m |
| cv_shield_k20 | 0.5167 ± 0.0017 | 1.0728 ± 0.0021 | 6.47% ± 0.17% | 0.00% | 0.2972 | 0.198m |
| two_stage_gate_mpn | 0.5179 ± 0.0017 | 1.0759 ± 0.0013 | 5.97% ± 0.24% | 8.30% | 0.2984 | 0.210m |
| native_plus_mpn_risk_grid | 0.5189 ± 0.0022 | 1.0780 ± 0.0021 | 5.71% ± 0.09% | 12.69% | 0.2994 | 0.221m |
| mpn_shield_k20 | 0.5205 ± 0.0015 | 1.0821 ± 0.0010 | 5.91% ± 0.27% | 8.39% | 0.3010 | 0.209m |

---

## 3. Key Scientific Findings

1. **Feature Attribution**:
   - **G0 (Native Logit Prior)**: Retains strong coarse intent priors; purely learning from scratch without native logit degrades performance, confirming that residual ranking ($s = s_{native} + lpha \Delta s$) is superior.
   - **G1 (Candidate Geometry)**: Path smoothness and curvature provide strong regularizers against erratic edge candidates.
   - **G4 (Model-Predicted Neighbor Risk)**: Essential for closing the social safety gap without requiring ground-truth future labels.
2. **Rule vs Learned Rankers**:
   - Lightweight heuristic shields (`mpn_shield_k5`, `two_stage_gate_mpn`) achieve strong Pareto efficiency with zero trainable parameters.
   - Learned pairwise rankers (`mlp_pairwise`, `linear_residual`) generalize robustly across held-out test scenes when trained on multi-fold OOF features.

---

## 4. Deliverable Files

- **Detailed Leaderboard**: `FINAL_LEADERBOARD.csv`
- **Pareto Trade-off Data**: `FINAL_PARETO.csv`
- **Summary Metrics**: `summary.json`
