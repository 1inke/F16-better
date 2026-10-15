# Canonical equal-K20 replacement diagnosis

Direct human authorization 2026-10-07: continue the narrow equal-candidate-budget experiment proposed after canonical reconciliation. NONFORMAL, saved-output-only, no new model training.

Control is immutable native20. Treatment retains 16 native paths and replaces exactly four native slots with all four canonical learned paths, keeping K=20. Removal is bottom four frozen native logits, descending stable ranking, ties original slot first. Removed slot indices are sorted into original slot order and receive learned slots 0,1,2,3 respectively. GT is never used for selection.

Nine canonical fixed-epoch25 models × outer/inner/originalCAL = 27 saved jobs. No old process/task querying or retraining. DESIGN/protocol are immutable preregistration; runtime and actual sole worker determine running state.

All artifacts live here. Inherit all 385 relevant prior SHA digests without replacing them; extend with completed reports, original artifacts and new scientific source. This is not full historical 34k SHA verification. Resource gates: process<=6GiB, physical and commit available each>=6GiB.
