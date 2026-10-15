# Phase M2: Adaptive Portfolio Composition Design

## Objective
Evaluate whether the canonical Top-3 composition tested in M0/M1 (`[Native Top-1, Static Top-1, Learned Slot 3]`) is optimal, or whether alternative compositions drawn from the frozen S1L3 candidate pool (such as dual learned slots) provide superior set coverage and trajectory diversity.

## Evaluated Compositions
1. **$C_1$ (Canonical Fixed Portfolio)**: `[Native Top-1, Static Top-1, Learned Slot 3]`
2. **$C_2$ (Dual-Learned Portfolio)**: `[Native Top-1, Learned Slot 3, Learned Slot 0]`
3. **$C_3$ (All-Learned Portfolio)**: `[Native Top-1, Learned Slot 3, Learned Slot 1]`
4. **$C_4$ (Pure Native Top-3)**: `[Native Top-1, Native Top-2, Native Top-3]`

## Evaluation Protocols
- Scene-Family ($N=1561$), Target-Track ($N=1035$), Recording-Time ($N=566$).
- 3 Seeds: 42, 43, 44.
- Metrics: Overall Coverage, Change Coverage, minADE@3, and Pairwise Trajectory Diversity.
