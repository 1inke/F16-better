# Registered local diagnostic design

Question: among cases lacking a same-path adequate candidate, which operational
response constraint first loses ALL K20 candidates: event presence, dominant type,
onset, direction, strength, or complete path accuracy? Does a candidate satisfying
all coarse response criteria still miss the complete trajectory?

The hierarchy is an accounting convention, NOT a ranking of causal bottlenecks.
Also report each component marginally and change the accounting order in a
prespecified sensitivity analysis. A failure can have multiple descriptive causes.

## Fixed data and labels

- Primary original FIT25912/CAL2879, no reshuffle/subset selection.
- Native candidates and logits from the existing original source caches. Original
  forecast-origin-relative coordinates; encoder past ends at zero. Saved future
  coordinates are labels/metric reference only, not inference/selection input.
- Reuse the unchanged original two-step velocity event rule (`common.events`):
  persistence over two adjacent steps; no-event onset sentinel99 remains missing,
  never interpreted as 39.6 seconds. Boundary-only uncertain cases are separate.
- Primary type is the rule's dominant FIRST event (turn/speed-change/start/stop).
  Multitype overlap is separately described. Earlier 387/469 coarse coverage used
  ANY shared event type, so the new dominant-type statistic is NOT the same metric.
- Onset tolerance +/-2 steps (.8s), fixed from prior diagnostic, no CAL/DEV tuning.
- Direction: turn left/right sign of angle at dominant event onset; speed-change
  acceleration/deceleration sign; start/stop sign follows type. No-event missing.
- Strength: mean of the two velocity observations at THAT candidate's own event
  onset. For turn, absolute angle vs observed reference heading, tolerance15deg;
  otherwise signed speed difference vs observed reference, tolerance max(.15m/s,
  .25*abs(GT strength)). These are new operational descriptive thresholds, not
  universal quality standards or optimized settings. Timing is evaluated separately.
- Same-path quality remains ADE<=.20m AND FDE<=.40m; .30/.50 sensitivity also fixed.
- Conditional onset/direction/strength errors ALWAYS include denominators and
  missing/mistyped candidate counts. Never combine features from different slots.
- Original logit argmax is only a posthoc selected-slot reference. With new
  time-response paths it is inherited/frozen scoring, not a newly trained selector.

## Accounting and outcomes

If a SAME candidate passes path quality, classify adequate-selected vs
adequate-present-but-selected-miss BEFORE event accounting, so small event-threshold
differences cannot override correct paths. Among clear-event cases with NO adequate
path: missing event / dominant type gap / onset gap / direction gap / strength gap /
coarse response covered but whole path inadequate. Nonclear and uncertain failures
are separate. Every row has one partition class plus independent multilabel masks.

Primary hierarchy event->type->timing->direction->strength. Prespecified alternate
order event->type->direction->strength->timing illustrates order dependence.
Report all stage rates, marginal type-timing/direction/strength coverage, same-slot
coincidence, speed-curve MAE, early/middle/late path errors, event-specific and
scene-family strata. GT-assisted joint oracle ADE+.5FDE is explicitly distinct
from original logit argmax and from independent minADE/minFDE.

Secondary ALL18 previously completed DEV outputs, including constrained/legacy/final;
no picking the best seed/policy or tuning on these outputs. Matched-ID rescue/lost
against native and paired block control; seeds are repeated IDs, not new people.
Do not apply formal-major training gates to this diagnostic or claim SOTA.

## Validation and continuation

Preflight: finite shapes/IDs, FIT/CAL disjointness, cached coordinate metrics,
synthetic events, same-slot coincidence, adequacy priority, candidate permutation,
original-source hashes and complete old-result SHA manifest. No model is instantiated
or mutated; old model SHA preservation is the relevant integrity check, not a fake
new all-weights replay. Independent saved-output arithmetic reconstructs stage masks,
partitions, marginal/group counts, and direct ADE/FDE for every job; records exact
scope (not all model replay or independent verification of event-label truth).

After SUCCESS and completion analysis, use FIT/CAL gap patterns and uncertainty
to specify the next candidate-specific response head. Architecture first retains
original losses/scorer/F16/data/guards; auxiliary event supervision or scoring are
separate interventions. No automatic training is embedded in this worker. Human
authorization for the sequence does not expand unfreezing/data/external-write scope.
