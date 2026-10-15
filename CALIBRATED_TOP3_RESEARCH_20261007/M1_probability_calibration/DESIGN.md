# Phase M1: Top-3 Probability Calibration Design

## Objective
Train a lightweight probabilistic calibrator predicting $(p_N, p_S, p_L)$ over the three fixed hypotheses:
$$\tau_1 = \text{Native Top-1}, \quad \tau_2 = \text{Static Top-1}, \quad \tau_3 = \text{Learned Slot 3}$$
with $\sum_{i=1}^3 p_i = 1$. The goal is to provide well-calibrated confidence and expected mass on valid trajectories ($P_{\text{good}}$) rather than forcing a deterministic single choice.

## Calibrator Architecture & Inputs
- Model: `Top3ProbabilityCalibrator` (12-d input $\to 32 \to 16 \to 3$ logits $\to \text{Softmax}$).
- Inputs:
  1. Native Top-1 logit and logit margin
  2. R2 scores for Native Top-1, Static Top-1, and Learned Slot 3
  3. R2 advantage differences ($S_L - S_N$, $S_S - S_N$)
  4. Native vs R2 Top-1 disagreement indicator
  5. IntentClassifier non-continuation regime probability
  6. $P2$ $k$-NN neighbor mode entropy
  7. Pairwise trajectory distances $D(N, S)$ and $D(N, L)$
- Loss: Cross-entropy to soft utility target distribution $U_c = -2 \cdot J_c + 3 \cdot \mathbb{I}(\text{good}_c)$.
- Training: 10 epochs with AdamW (lr=2e-3), trained strictly on the training partition.

## Evaluation Metrics
- NLL and Brier score
- ECE (Expected Calibration Error) across 10 confidence bins
- Expected mass on good trajectories: $P_{\text{good}} = \sum_{c: \tau_c \text{ is good}} p_c$
- Probability shifts across Stable, Transition, and Change subgroups.
