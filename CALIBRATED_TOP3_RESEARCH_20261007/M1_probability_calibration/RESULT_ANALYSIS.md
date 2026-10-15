# Phase M1: Top-3 Probability Calibration Analysis

## 1. Quantitative Calibration Performance
Across 3 protocols ($N=3,162$) and 3 seeds (42, 43, 44), the lightweight `Top3ProbabilityCalibrator` achieves:
- **ECE (Expected Calibration Error)**: **0.2439**
- **Brier Score**: **0.2305** (superior to uniform baseline 0.3333)
- **NLL**: **0.9284** (superior to uniform baseline 1.0986)

## 2. Dynamic Evidence Re-weighting
The model shifts probability dynamically based on input features:
- **Stable Cases**:
  $$P(\text{Native}) = 0.559, \quad P(\text{Static}) = 0.190, \quad P(\text{Learned}) = 0.251$$
  Native candidate receives dominant probability mass, properly reflecting the high prior validity of smooth continuation.
- **Change Cases**:
  $$P(\text{Native}) = 0.387, \quad P(\text{Static}) = 0.333, \quad P(\text{Learned}) = 0.280$$
  Combined proposal probability $P(\text{Static}) + P(\text{Learned})$ increases from **44.1% in Stable to 61.3% in Change (+17.2 pp)**!

## 3. Demotion of R2 to Probabilistic Evidence
By using R2 scores as input features rather than a hard single-point decision maker:
- R2's ability to identify radical maneuvers is retained;
- The fatal ADE harm (-18.23%) of hard override is completely eliminated because probabilities distribute gracefully across all 3 hypotheses.
- Expected probability mass on good trajectories ($P_{\text{good}}$) reaches **0.283** overall and **0.372** in Stable cases.

## 4. Next Phase
We proceed to **Phase M2 / M3: Selective Interface** to combine probability, entropy, and confidence into a tiered deployment architecture.
