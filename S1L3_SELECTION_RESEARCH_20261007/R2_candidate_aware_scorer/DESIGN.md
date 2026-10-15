# Stage R2: Candidate-Aware Scorer Evaluation

## Objective
Evaluate whether a candidate-aware scoring architecture incorporating history tokens ($P2$), candidate trajectory and geometry, proposal provenance, and set context can overcome the structural provenance bias discovered in Stage R0 and convert newly generated support into Top-1 predictions.

## Architecture
- Backbone: Frozen $P2$ tokens encoder.
- Inputs:
  1. $P2$ pooled representation (96-d).
  2. Candidate trajectory coordinates (24-d flat).
  3. Candidate geometry: endpoint distance, path length, average speed, heading angle (4-d).
  4. Candidate provenance: 1-hot 5-d (Native, Static, Learned-3, Learned-0, Learned-1).
  5. Set context: mean and std of endpoint $(x, y)$ across all 24 candidates (4-d).
- Residual Formulation:
  $$\text{score}_{\text{new}} = \text{score}_{\text{native}} + \Delta(P2, \text{cand}, \text{set})$$
  with the final linear layer of $\Delta$ initialized strictly to zero to prevent uninitialized corruption of native priors.
- Loss: Cross-entropy to soft utility target distribution:
  $$U_k = -2.0 \cdot J_k + 3.0 \cdot \mathbb{I}(\text{good}_k)$$
  $$p^* = \text{softmax}(U)$$

## Evaluation Scope
- 3 Isolated Grouped Protocols: Scene-Family, Target-Track, Recording-Time.
- 3 Seeds: 42, 43, 44.
- Total evaluations: 9 runs.
