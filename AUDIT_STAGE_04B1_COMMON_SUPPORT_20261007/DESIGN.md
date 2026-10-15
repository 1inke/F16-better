# Stage 04B1: Common-Support Normalization & Whitening Experiment

## 1. Scientific Objective
In Stage 04A, explicit maneuver-type conditioning was shown to produce essentially identical performance to unconditioned control ($\Delta_{TC} = -0.00\text{ pp}$), while both models suffered from over-editing (ruining 82–109 previously valid Static K24 windows).

Stage 04B1 tests the **Support Shift Hypothesis**:
- Is the transfer gap and over-editing driven by feature scale and support domain shift between the training domain (University + Hotel) and the outer domain (Zara)?
- **Intervention**: Fit feature standardization and whitening strictly on the preregistered `common_statistics_index` ($N=12,183$), which captures the covariate intersection between scene environments.

---

## 2. Experimental Controls & Invariants

### Single Factor Changed:
- **Feature Normalization**: Instead of standardizing observation features using the raw `grouped_train_index` moments, moments are estimated on `common_statistics_index` with feature whitening applied.

### Invariants Maintained:
- **Candidate Pool**: Identical K=24 candidate set.
- **Architecture**: `ProposalConditionedCorrectionHead` (obs_dim=193, hidden=128, state_dim=32, type_dim=16, num_types=9).
- **Optimizer & Budget**: AdamW, lr=$3 \times 10^{-4}$, weight_decay=$10^{-4}$, batch_size=128, epochs=50.
- **Seeds**: 1, 2, 3.
- **Loss**: Identical hard quality objective + residual L2 + smoothness + acceleration penalty.
- **Frozen Models**: F16 backbone frozen, decoder frozen, scorer frozen.
