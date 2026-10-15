# DESIGN: Stage P3 Fixed Portfolio K24 Audit

## 1. Scientific Objective
- **Problem**: K28 Union proved that combining Static4 and Learned4 yields +2.48 pp overall coverage and +2.05 pp Change coverage. Under a hard budget of $K=24$ (20 Native + 4 proposals), can a fixed static/learned portfolio allocation capture both advantages without training an adaptive selector?
- **Configurations Evaluated**:
  All combinations strictly enforce proposal budget $= 4$ and total candidate count $K=24$:
  1. **S4L0**: 4 Static + 0 Learned (Control: Static-K24)
  2. **S3L1**: 3 Static + 1 Learned (Retaining top 3 classifier Static + Learned Slot 3 Change-specialist)
  3. **S2L2**: 2 Static + 2 Learned (Retaining top 2 classifier Static + Learned [Slot 3, Slot 0])
  4. **S1L3**: 1 Static + 3 Learned (Retaining top 1 classifier Static + Learned [Slot 3, Slot 0, Slot 1])
  5. **S0L4**: 0 Static + 4 Learned (Treatment: Bo4-K24)
- **Pre-registered Retention Rules (Zero Outer Leakage)**:
  - Static retention: Sorted by classifier intent probability (Top-1, Top-2, Top-3, Top-4).
  - Learned retention: Locked from Stage P2 specialization profile (Slot 3 decel/stop $\to$ Slot 0 cruise/accel $\to$ Slot 1 left turn $\to$ Slot 2 right turn).
- **Preregistered Success Criteria**:
  1. Overall coverage not lower than Static-K24 in all 3 seeds;
  2. Change 3-seed mean coverage > Static-K24 (41.71%);
  3. Change coverage positive in $\ge 2/3$ seeds;
  4. Change minADE or minFDE stably improved;
  5. No severe motion pathology.
  If satisfied $\implies$ output `FIXED_PORTFOLIO_POSITIVE`.
