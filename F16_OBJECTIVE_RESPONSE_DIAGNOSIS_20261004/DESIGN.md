# Registered original-objective response audit

## Question and method

Does the unchanged training objective favor existing same-path good candidates,
and how does hard-winner/path-gradient allocation differ from strict adequacy?
This is a descriptive within-window comparative computational audit, not a causal
training intervention or benchmark improvement. Local primary evidence is the
locked original code, saved checkpoints, FIT/CAL paths and completed reachability
results. AI-assisted design/audit; inline adversarial reviews are not external review.

## Fixed scope / comparisons

FIT25912 and CAL2879 only; all 28791 IDs unchanged. Native K20, three original
CAL-constrained learned pulse paths, GT-assisted full pulse set, and single-slot
pulse witness replacement. In the last comparison all other 19 native slots are
exactly unchanged. Choose among production-validated strict pulse witnesses by
lowest original ADE+.5FDE, tie original slot order. If none exists leave native
unchanged and label no-witness; never fit or search anew. Full oracle set is an
expressivity comparison, not a proposed collapse/teacher/production candidate set.

ADE<=.2 AND FDE<=.4 on the SAME candidate remains primary; .3/.5 is fixed sensitivity.
No deletion of hard windows. Distinct-good quantity is available from the prior
audited complete-linkage outputs, not recomputed for synthetic replacement here;
raw counts in this audit must not be called effective diversity.

## Objective and allocation

Use checkpoint-original config, all-FIT class weights, original FIT protection
scales, ADE+.5FDE hard winner, cross-native-family diversity, increment residual
and increment smoothness, .1 protection and .01 guard unchanged. Extract original
functions via AST without editing them or importing trainer side effects.
Independent decomposition and numpy arithmetic must match originals on batches.

Per-row loss contributions use fixed POPULATION denominators: winner /N,
diversity sum across cross-family pairs /total_cross, protection sum /Nprotected,
residual and smoothness element means. Hence mean(per-row contributions) exactly
represents the full population objective; these are not literal per-window losses
or exact shuffled-minibatch training gradients. Per-block original batch equality
is checked separately; population gradient is chunk independent.

Autograd variable is saved coordinate path. Residual increments are differenced
from output minus native path in float64, with zero-prepend. This reconstructs the
mathematical correction but NOT the original float32 head residual bit pattern;
roundtrip must be within original 2e-5m. Original objective equality applies to
these reconstructed inputs. Gradient direction/norm/cosine is output-space local,
not parameter-gradient accessibility, shared mapping or convergence proof.

Report winner-good mismatch, raw good count/coverage, per-component deltas,
quality/regularizer/protection gradient norms, quality-total cosine, good-slot
gradient mass, directional derivative along single-witness change and per-window
speed/acceleration/jerk (dt .4). Protection independently permits different ADE/FDE
oracle slots, unlike same-path adequacy. Penalizing motion costs here does not add
a new training loss. Window p95 aggregates are explicitly not formal global p95 gates.

## Validation / resource / decision rules

Lock prior whole manifest plus all latest reachability/failure/report artifacts,
sources and 450 input done/output SHAs before and after. Validate explicit IDs,
finite K20, native cache error alignment, original FIT weights/scales, selection,
all 19 untouched slots, exact reconstructed loss and gradient decomposition,
numpy independent arithmetic for every saved block. Resume accepts completed jobs
only after source AND output SHA, and rejects source mismatch/corrupt output.
64 rows per block; input memory and full-size allocation/resource tested before
launch, 6GiB process and system reserve6GiB. Substantive obstruction stops for human.

## Inline Devil's Advocate checkpoint 1

PASS WITH LIMITS. Major alternative: GT-assisted independent parameters can lower
loss while a shared past-only head cannot realize them. Motion or diversity costs
can make full-set constructions undesirable; single-slot comparison isolates much
but not all of this. Original minibatch denominator geometry and head Jacobian are
not tested by output-space population gradients. Strict binary thresholds and one
future do not represent complete future multimodality. Preserve 3 unequal scenes,
CAL university-only, overlapping windows/tracks, F16 already seen, prior CAL selection,
future-completeness qualification bias; no causal root, intent, irreducibility, SOTA,
new reliable group CI/p-value or new formal major claim. Negative objective result
does not authorize changing loss; favorable objective result does not prove optimizer
failure. Retain old free-increment/time-query failures and FIT/CAL generalization gap.

## Completion and next node

Worker AND exit SUCCESS, all 450 done SHA, arithmetic/source/old preservation PASS,
resource budget and independent original-objective checks precede interpretation.
Save RESULT_ANALYSIS plus inline checkpoints2/3 and update controls to ANALYZED.
Then report and wait human; no automatic third formal train, event supervision,
new loss, decoder/scorer unfreeze, data expansion or goal completion.
