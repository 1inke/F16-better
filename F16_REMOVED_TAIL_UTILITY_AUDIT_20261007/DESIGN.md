# Frozen protocol

## Inputs and unit

All scene_family/target_track/recording_time × seeds42,43,44 × outer/inner/original_CAL saved outputs, 52686 run-window rows. Same fixed bottom-native-logit4 replacement; no alternative deletion search. Original IDs, GT, native paths, loss-group labels, frozen logits and track/scene metadata must align exactly. GT is posthoc only. Native replay across three seeds is not three independent observations. Reports show each run and unique ID/track counts; never pool different outer populations as paired observations.

## A: lost support / winners

lost = native has a same-candidate ADE<=.2 AND FDE<=.4 path, replacement does not. Record every native good slot, native argmin J=max(ADE/.2,FDE/.4), native hard-Q argmin(ADE+.5FDE), separate minADE/minFDE slots. J winner must be good and removed on lost, ALL native-good slots must be removed; this is an exact consequence of unchanged common16, not a discovery of rarity. Q winner can be non-good/common. Record tie gaps, old-logit ranks, slot/rank histograms, removed joint utility and native individual utilities. Multiple good removed slots can make individual LOO coverage utility zero although the removed GROUP is necessary.

## B: geometry, not inferred semantics

For J-best removed good r*, compute pair distance D(r,c)=max(mean12 Euclidean separation/.10, endpoint separation/.20); d_redundancy=min common16 D. Store nearest common original slot and its meanpath/endpoint separation, plus continuous errors. D>1 is outside this fixed pair proximity threshold, not proof of a unique effective cluster or behavioral mode; D<=1 does not imply equivalent GT adequacy.

## C/D: per-case posthoc kinematics

Store errors ADE/FDE/J and same-J-best full path for r*, learned J-best, all four learned paths and GT. Record v_max, within-path v_p95 (12 intervals from last observed origin), arclength s, endpoint displacement, delta_s=predicted arclength-GT arclength. Record endpoint longitudinal overshoot and cross-track error along GT end direction; endpoint direction error undefined when either displacement <=.03m. Turning uses signed atan2 between adjacent velocities only where BOTH speed>.03m/s; store valid count, cumulative signed/absolute turn and error, not psychological/event labels or pointwise curvature at zero speed. DT=.4.

Fixed descriptive flags, not revised success gates: GT low average speed<=.1m/s; GT terminal stop = last3-interval mean speed<=.1m/s (report continuous values and original group too); direction error>30deg only valid displacement; endpoint longitudinal overshoot>.2m; arclength excess>.2m; terminal-stop not retained = GT terminal stop and predicted last3 speed>.1m/s; absolute total turning difference>30deg only with valid segments. Flags can overlap and must not be called causes. Lost/rescue alone are selection-conditioned; retained/neither and full-population GT prevalence supply context. No threshold sweep, no future teacher.

Speed attribution: per group record exact native20/replacement20 pooledALL-K time-step p95, common16/removed4/learned4 and native24 descriptive components, and count exceedances above ORIGINAL native p95. Replacement excess exceedances equal learned4 excess minus removed4 excess because common16 is fixed. This identity separates removals/additions; nonlinear quantile change is not additively allocated to causes. Report the original five-motion audit unchanged; do not substitute case-wise p95 for pooled p95.

## E: leave-one-out utility

For every native20 and replacement20 slot and every window save Ucoverage=good_j AND sum(good)==1, U_ADE=min other19 ADE-min20 ADE, likewise FDE and J. All quantities nonnegative; exact ties yield zero (no jitter). Independently enumerate all20 deletions on fixtures, actual128 and saved full outputs. Report native old-logit rank1..20 versus mean utility/nonzero rate/sum and removed marginal support; rank is NOT a learned/refined probability or a redundancy measure. No use of GT utility to construct a deployable policy. Native rank plots deduplicate repeated seeds within each G/split and show stable/transition/change as well as overall. No universal claim based on small lost counts.

## Verification and interpretation

Before launch: resource guard first; synthetic lost/rescue, disjoint minima, tie utilities, multi-good group-null-individual, geometry-close-but-coverage-fail, known speed/stop/direction fixtures; actual128 two implementations; all27 saved ID/native/group align and original job metrics lost/rescue EXACT; all inherited related SHA initial/final; completed-analysis SHA-skip bad-source/output reject. Benchmark eight saved128 batches, full input/output/temp estimate and disk; no models saved/trained. Completion: dual SUCCESS,27done output SHA, all saved utilities/cases/rank/tail summaries independent reconstruction, source initial/final and resource PASS. No complete old-model replay or external review claim.

ARS DA1 inline: PASS with limits. A's J winner-in-removed result is algebraically guaranteed; hard-Q is a separate diagnostic. Individual utility can miss group complementarity. Stop/turn flags depend on descriptive cutoffs and correlated reused windows; report continuous counterexamples, don't relabel change as CV failure. Utility computed with future GT is audit-only. Positive learned4 utility cannot establish new information or decide a safe future architecture. Full analysis and DA2/3 precede any future experiment proposal; no automatic training.
