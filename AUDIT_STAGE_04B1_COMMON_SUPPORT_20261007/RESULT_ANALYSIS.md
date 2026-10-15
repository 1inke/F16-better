# Stage 04B1: Common-Support Normalization Results & Decision Audit

## 1. Executive Summary & Experimental Results

Stage 04B1 tested whether fitting observation and candidate state standardization strictly on the common-support partition ($N=12,183$) resolves covariate shift and over-editing on Zara scene-family outer ($N=1,561$).

### 1.1 Complete 3-Seed Comparison Table
| Model Configuration | Seed | Zara Outer Covered ($N$) | Zara Outer Rate (%) | Gain vs 04A Best (pp) | Gain vs Static K24 (pp) | Lost vs Static ($N$) | Joint Error $J$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static K24** | — | **1,056** | **67.65%** | — | — | 0 | 0.7942 |
| **Stage 04A Best** | Pool | 1,087.0 | 69.64% | Ref | +1.99 pp | 99.0 | 0.7702 |
| **Stage 04B1 (Common Support)** | 1 | 1,087 | 69.63% | -0.01 pp | +1.98 pp | 96 | 0.7719 |
| | 2 | 1,084 | 69.44% | -0.20 pp | +1.79 pp | 114 | 0.7745 |
| | 3 | 1,083 | 69.38% | -0.26 pp | +1.73 pp | 112 | 0.7740 |
| *04B1 3-Seed Mean* | | **1,084.7** | **69.49%** | **-0.15 pp** | **+1.84 pp** | **114 (Max)** | **0.7735** |

---

## 2. Scientific Interpretation
1. **Common-Support Moments Do Not Resolve the Transfer Gap**:
   Restricting the normalization statistics to the common support partition ($N=12,183$) yielded a 3-seed mean coverage of 69.49%, which is slightly lower (-0.15 pp) than Stage 04A (69.64%).
2. **Over-Editing Remains Acute**:
   Max lost candidates vs Static K24 remains high at 114 windows. The learned residual vectors continue to perturb otherwise valid native candidates beyond the 0.2m / 0.4m thresholds.
3. **Branch Decision**:
   Because `gain_vs_previous_best` $\le 0$ but the support discrepancy ratio ($R_{support} = 1.257 \ge 1.20$) remains elevated, the preregistered decision engine authorizes **one final representation/data-balance intervention**: `04B2_SCENE_BALANCED`.
