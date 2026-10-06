# F16 response-gap diagnosis — completion interpretation

Analyzed 2026-10-04. This is a saved-output diagnostic, not a new trained model or formal experiment.

## Outcome

The main descriptive gap is **coarse response coverage without an accurate whole trajectory**, not simply an absence of every change-like candidate. This supports prioritizing candidate-specific fine motion evolution, but does not prove the decoder is the unique causal bottleneck.

Worker and exit both SUCCESS, finished 2026-10-04T10:17:11.294612+08:00, elapsed 6m13.88s. All21 jobs and independent arithmetic audit PASS. Completed worker verifies19495 prior-file SHA and242 source SHA unchanged. Direct coordinate ADE/FDE recomputation error0 across21 jobs; independent scalar accounting also PASS. This is not model replay or independent certification of the event labels.

No saved halfway check record exists: runtime's first_progress_query_done remains false. This report does not retrospectively claim one. Completion review is user-triggered; no completed PID/process/task was polled.

## Primary CAL: denominator and quality definition

Among CAL2879,1360 have a clear persistent future motion-change event. A high-quality candidate here means **one and the same path** has ADE<=.20m AND FDE<=.40m; .30/.50 is a locked sensitivity threshold, not SOTA criteria.

- 353/1360 cases have at least one adequate candidate (25.96%).
- 1007/1360 have no adequate candidate (74.04%).
-Of the353 covered cases, the original logit-argmax selects an adequate path in30 and misses it in323. This selection gap cannot fix the1007 generation failures.

The following classification is ONLY among the1007 generation failures, using the registered event→type→timing→direction→strength hierarchy:

| First unresolved operational gap | Cases | Share of1007 |
|---|---:|---:|
| No change-like event or no matching dominant type |0|0.0%|
| No candidate with matching onset within ±2 steps |110|10.9%|
| No remaining candidate with matching direction |51|5.1%|
| No remaining candidate with matching onset-linked strength |115|11.4%|
| Same candidate matches all coarse criteria, but whole path remains inadequate |731|72.6%|

All coarse criteria must coincide in the same candidate; attributes from different slots are never pooled into a fictional path. Event type is dominant FIRST event, unlike the earlier ANY-type overlap statistic.

The strength criterion observes only two velocity samples at each path's own event onset: turn magnitude tolerance15deg; signed speed-change tolerance max(.15m/s,25% of GT strength). It does not certify response duration, recovery, sustained acceleration/curvature or complete speed evolution. Matching these coarse tolerances is therefore much weaker than accurate whole-path forecasting.

## Cross-checks and order sensitivity

- FIT:7085 clear-event generation failures;5275 (74.45%) match the coarse criteria but fail full-path quality.
- Secondary already-explored DEV:377/469 clear-event cases lack a good path;236/377 (62.60%) fall in the same coarse-covered/full-path-inadequate category.
- CAL looser .30/.50 threshold:615 generation failures;437/615 (71.06%) in this category. This is a descriptive sensitivity check, not a reason to alter standards.
- Reordering the hierarchy to event→type→direction→strength→timing changes CAL timing/direction/strength counts from110/51/115 to216/9/51. The731 final coarse-covered failures stay unchanged. These are accounting categories, not causal effect estimates or a uniquely ranked list of root causes.
- Coarse-stage CAL nested coverage is1359 type,1212 type+timing,1137 +direction,1003 +strength out of1360. Do not assume all353 adequate paths satisfy all coarse rules: only272 covered cases overlap the full coarse criterion;81 can be geometrically adequate without passing every event threshold.

For the GT-assisted same joint-oracle candidate, CAL early/middle/late mean position errors are .1374/.3006/.3839m; secondary DEV .1580/.3900/.5751m. This is consistent with errors accumulating across the forecast, not proof that late-response modeling alone explains them. The oracle uses GT for retrospective analysis and is not the deployed selector.

## Essential counterevidence: event coverage is not cue understanding

In the nonboundary no-clear-GT-event subgroup, every K20 bank contains some event-like candidate:

| Dataset | Cases | Any candidate event | Selected candidate event |
|---|---:|---:|---:|
| FIT |14980|14980|369|
| CAL |1420|1420|52|
| DEV, secondary |1000|1000|30|

Thus near-complete candidate event coverage can arise from generic multimodal diversity, including false event alternatives when GT has no clear change. It does NOT establish that observed cues triggered the right event hypothesis. Probe-readable information, candidate coverage and beneficial native use are three distinct statements. No irreducibility/intent/causal-utilization claim follows.

These supplementary counts and21 saved-job SHA were checked in this completion review; see completion_interpretation_audit.json. The full19495 prior-file comparison cited above is the completed worker's audit, not a fresh full-file replay here.

## Prior time-response trial: no meaningful repair

All18 prior outputs were inspected, without choosing a seed/checkpoint based on DEV. For constrained policies on the same469 clear-event IDs:

| Output | Adequate candidates exist, per seed | Coarse-covered/full-path failure, per seed |
|---|---|---|
| Native |92|236|
| Fresh block control |91 /92 /92|235 /236 /237|
| Stepwise time-response treatment |90 /91 /93|234 /241 /235|

The dominant failure pattern remains. Those are repeated IDs across seeds and policies, not18 independent validation populations. Increasing free per-step correction under the prior frozen-feature/original-loss setup did not solve the full-process problem.

## Implication for the already-authorized sequence

Prioritize a candidate-specific response parameterization that expresses **strength evolution and persistence/duration, coordinated speed/heading dynamics, and endpoint–path consistency**, rather than only a coarse event-label classifier, one global risk gate for allK, or another larger/free-output head. K20 should retain distinct plausible hypotheses; moving every candidate toward one GT is not the objective.

A next architecture contrast must first specify its isolated intervention and falsification criterion using FIT/CAL, with original F16/decoder/scorer, losses, guards and data held fixed. Event auxiliary supervision and final-path scoring remain separate interventions. This diagnostic is not itself a reason to add loss terms, unfreeze modules or assert a new baseline. No new training was launched in this result-report turn.

Selection also matters: CAL323 and secondary DEV87 cases contain an adequate candidate that the inherited original argmax misses. Address this later as a separate scorer-alignment experiment; do not conflate it with absence of good candidates.

## Limitations

All1360 CAL clear-event cases belong to the university scene family. FIT spans hotel42/university8704/zara1491 clear-event cases, and DEV hotel6/university383/zara80. These are highly unbalanced, correlated trajectory windows from data the original model has already seen; only three scene families. The result is not a held-test, cross-fold or reliable population inference.

Fixed coarse thresholds and unchanged automatic labels can misdescribe some paths; onset strength is deliberately local. Function/optimization/generalization, representation, observation ambiguity and physical bounds remain alternative contributors. Older GT-assisted three-anchor reachability and tiny-FIT/CAL overfit results are counterevidence against a decoder-only diagnosis.

Formal response-study counter remains1/3; diagnosis does not increment it. No major/SOTA/generalization or project-completion claim. All old scientific sources/models/results remain unchanged.

