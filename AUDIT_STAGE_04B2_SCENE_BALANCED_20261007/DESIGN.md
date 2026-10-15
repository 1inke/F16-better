# Stage 04B2: Scene-Balanced Training Experiment

## 1. Scientific Objective
In Stage 04B1, common-support normalization failed to improve outer transfer (-0.15 pp vs 04A), while over-editing persisted.

Stage 04B2 implements the **Preregistered Final Representation / Data-Balance Intervention**:
- In the training set (`grouped_train_index`), University represents 95.3% (15,578 samples) and Hotel represents 4.7% (759 samples).
- Does balancing scene representation during training alleviate scene-specific bias and improve zero-shot transfer to the unseen Zara scene family ($N=1,561$)?

---

## 2. Experimental Controls & Invariants

### Single Factor Changed:
- **Scene-Balanced Batch Sampling**: Using `WeightedRandomSampler` on `grouped_train_index` with weights inversely proportional to training scene frequencies (`w_hotel = 1/759`, `w_univ = 1/15578`).
- Zara outer is NEVER used to choose sampling weights.

### Invariants Maintained:
- **Candidate Pool**: Identical K=24 candidate set.
- **Architecture**: `ProposalConditionedCorrectionHead` (obs_dim=193, hidden=128, state_dim=32, type_dim=16, num_types=9).
- **Normalization**: Best normalization from Stage 04B1.
- **Optimizer & Budget**: AdamW, lr=$3 \times 10^{-4}$, weight_decay=$10^{-4}$, batch_size=128, epochs=50.
- **Seeds**: 1, 2, 3.
- **Loss**: Identical hard quality objective + residual L2 + smoothness + acceleration penalty.
- **Frozen Models**: F16 backbone frozen, decoder frozen, scorer frozen.
