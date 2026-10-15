# Protocol 1: Scene-Family Grouped Transfer Confirmation

## Objective
Verify performance of frozen finalist candidate portfolios F1 (S1L3) and F2 (S2L2) on Zara Scene-Family outer evaluation ($N=1561$) relative to the frozen Canonical Static-K24 baseline ($N=1561$, Change cov = 41.71%).

## Control
- Canonical Static-K24: Native20 + Top-4 Heuristic Proposals from Canonical IntentClassifier (`canonical_intent_classifier.pth`).

## Treatment Finalists
- **F1 (S1L3)**: Native20 + Static proposal 0 + Learned slots [3, 0, 1] (Total $K=24$)
- **F2 (S2L2)**: Native20 + Static proposals [0, 1] + Learned slots [3, 0] (Total $K=24$)

## Seeds
- 42, 43, 44
