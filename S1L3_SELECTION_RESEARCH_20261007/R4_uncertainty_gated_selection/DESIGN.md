# Stage R4: Uncertainty-Gated Selection Design

## Objective
Evaluate whether deployment-time, ground-truth-free uncertainty features can dynamically gate between the conservative Native scorer and the aggressive Candidate-Aware (R2) scorer:
$$S(h) = \begin{cases} S_{\text{native}}, & u(h) < \tau \\ S_{R2}, & u(h) \ge \tau \end{cases}$$
The goal is to test whether we can capture R2's Change gains on hard windows while preserving the Native scorer on ordinary windows to keep Overall ADE harm below $1.0\%$.

## Experimental Tracks
1. **R4A — Override Predictability Audit**:
   - Target: $Y_{\text{override}} = \mathbf{1}[J_{R2} < J_{\text{Native}}]$.
   - Features: Native margin, Native entropy, R2 entropy, Top-1 disagreement, R2 chose proposal, non-continuation regime probability, $P2$ $k$-NN neighbor mode entropy, candidate endpoint spread, learned advantage.
   - Evaluated by ROC-AUC on outer eval splits using a Logistic Gate fit strictly on the training split.
2. **R4B — Native / R2 Selective Routing**:
   - Sweep override fractions $f \in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00]$.
   - Gate Criterion: Across $\ge 2/3$ protocols, there exists an operating point with Change Gain $> 0$, Overall ADE harm $< 1.0\%$, and severe regret decreased.
3. **R4C — Selective Prediction (Coverage-Risk Curve)**:
   - Abstain on top $[5\%, 10\%, 15\%, 20\%, 30\%, 40\%]$ highest uncertainty samples.
   - Evaluate selective risk, retained accuracy, and AURC.
4. **R4D — Track B: Top-3 Multi-Hypothesis Portfolio**:
   - Fixed portfolio: [Native Top-1, Static Top-1, Learned Slot 3].
   - Evaluate set coverage, oracle minADE/FDE, and effective set diversity.
