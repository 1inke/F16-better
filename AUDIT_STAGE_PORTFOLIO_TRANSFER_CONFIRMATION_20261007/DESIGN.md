# Portfolio Transfer Confirmation Suite

## Scientific Problem
Verify whether the candidate generation gains observed on Zara scene-family for fixed hybrid candidate portfolios F1 (S1L3) and F2 (S2L2) transfer robustly across three isolated grouped evaluation regimes:
1. Scene-Family Outer ($N=1561$, Scene-Family Unseen)
2. Target-Track Outer ($N=1035$, Pedestrian-ID Unseen)
3. Recording-Time Outer ($N=566$, Time-Block Unseen)

## Hard Constraints
- Total candidate budget $K=24$ strictly maintained.
- Native20 completely frozen.
- Complete-good definition: ADE <= 0.2 m and FDE <= 0.4 m (same candidate).
- Zero outer exposure during training of any selector or generator (`train_ids ∩ outer_ids = ∅`).
- Final scorer training strictly prohibited.
