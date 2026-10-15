# Stage R0: Frozen Scorer Baseline, Rankability & Score Bias Audit

## Objective
Evaluate the rankability and scoring bias of the frozen native scorer on the frozen S1L3 candidate set ($K=24$: Native20 + Static1 + Learned3 [slots 3, 0, 1]) without any new scorer training.

## Scientific Questions
1. Does the frozen native scorer convert newly generated candidate support into Top-1 accuracy?
2. Does the native scorer exhibit structural provenance bias against Static/Learned proposals?
3. Does past-only representation contain sufficient discriminative signal to rank good candidates over bad candidates?

## Evaluations
- Candidate existence (Oracle coverage/minADE/minFDE/J) vs Actual selection (Top1 metrics).
- Selection conversion efficiency: Selection Gain / Generation Gain.
- Good candidate rank distribution: Recall@1..20, MRR, median rank across Overall, Stable, Transition, Change, clear-event, and NewSupport.
- NewSupport scorer witness audit: individual sample inspection of lost opportunities.
- Score provenance bias audit: quality-matched stratification of scores across Native, Static, and Learned proposals.
- Pairwise rankability audit: pairwise classification accuracy and AUC.
- History-only rankability probes (P0..P3).
