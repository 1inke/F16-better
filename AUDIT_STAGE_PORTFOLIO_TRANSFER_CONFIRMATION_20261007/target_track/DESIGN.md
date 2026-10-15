# Protocol 2: Target-Track Grouped Transfer Confirmation

## Objective
Verify performance of frozen finalist candidate portfolios F1 (S1L3) and F2 (S2L2) on Target-Track outer evaluation ($N=1035$, Pedestrian-ID unseen) under strict isolation (zero outer exposure during training of either IntentClassifier or Best-of-4 generator).

## Control
- Canonical Isolated Static-K24: Native20 + Top-4 Heuristic Proposals from Isolated IntentClassifier trained on `grouped_train_index` ($N=18909$).

## Treatment Finalists
- **F1 (S1L3)**: Native20 + Static proposal 0 + Learned slots [3, 0, 1] from Bo4 Generator trained on Target-Track train split (Total $K=24$).
- **F2 (S2L2)**: Native20 + Static proposals [0, 1] + Learned slots [3, 0] from Bo4 Generator trained on Target-Track train split (Total $K=24$).

## Seeds
- 42, 43, 44
