# DESIGN: Stage P2 Learned Slot Semantic Audit

## 1. Scientific Objective
- **Problem**: In Stage A & B, Slot 3 accounted for ~61% of all winners on the Change subset in all three seeds. We must determine whether this represents a genuine, stable semantic specialization or merely a coincidental index naming artifact across random initializations.
- **Protocol**:
  - For Seeds 42, 43, 44, extract per-slot parameter profiles ($s_{t4}, s_{t8}, s_{t12}, \theta_{t4}, \theta_{t8}, \theta_{t12}, \text{onset}, \text{persistence}$) and trajectory geometries (path length, endpoint radius, curvature, accel, jerk).
  - Compute usage frequencies: overall winner share, subgroup winner share (Stable, Transition, Change), rescue count, and unique rescue count.
  - Perform formal **Hungarian Matching** (`linear_sum_assignment`) between Seed 42 (anchor) and Seeds 43 / 44 based on joint parameter profile and trajectory geometry distance matrices.
  - Verify whether a stable `change-specialized slot` emerges across all 3 seeds under the permutation.
