# Phase M0: Top-3 Complementarity Audit Design

## Objective
Deconstruct the +8.16 pp Change gain observed in the Top-3 portfolio `[Native Top-1, Static Top-1, Learned-3]` without training any new model.
Determine:
1. Exact Venn diagram breakdown (Native-only, Static-only, Learned-only, and pairwise/triple overlaps).
2. The relative share of Change rescues attributable to Learned-3 vs Static-1.
3. Geometric diversity, pairwise trajectory distance, effective mode count, and continuous oracle metrics (minADE@3, minFDE@3).

## Candidate Set Definition
- Candidate 1: Native Top-1 ($k_{\text{native}} = \text{argmax}(\text{logits}[:20])$)
- Candidate 2: Static Top-1 (slot 20, derived from canonical IntentClassifier top non-continuous maneuver)
- Candidate 3: Learned Slot 3 (slot 21, derived from frozen Bo4 generator)

## Evaluations
- Protocols: Scene-Family ($N=1561$), Target-Track ($N=1035$), Recording-Time ($N=566$).
- Seeds: 42, 43, 44.
- Metrics: 8-way Venn decomposition, net rescues, minADE@3, minFDE@3, pairwise trajectory distances, distinct mode count ($>0.1\text{ m}$ threshold), and distribution of #good candidates $[0/3, 1/3, 2/3, 3/3]$.
