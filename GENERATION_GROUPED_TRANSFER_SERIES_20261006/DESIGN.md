# Preregistered first contrast: target-track exposure

## Question and scope
Does new-head complete-candidate learning depend on access to other highly correlated windows from the target pedestrian recording, when evaluation windows and training budgets are matched? Only original FIT25912 is repartitioned. Frozen F16 already saw these data, so this is new-head adaptation transfer, not independently unseen whole-model generalization.

Primary complete quality remains one same candidate ADE<=.2m AND FDE<=.4m, J=minK max(ADE/.2,FDE/.4). GT-free complete-link .10 meanpath/.20 endpoint effective-good counts are secondary, not maximized; scorer is last. MinADE/minFDE, single-path early/middle/late errors, rescue/lost, all five pooled physical metrics dt.4 remain required. Original success gates and five-motion relative-native 1% protections are not loosened. Internal transfer effects do not upgrade F16 baseline or establish SOTA.

## Identity and splitter, fixed before quality outcomes
Canonical target-track key is sequence + pedestrian_id, never pedestrian_id alone. All original FIT windows remain recorded; ambiguous start times are reported, not silently deleted. If sequence or target identity itself is unavailable/ambiguous, STOP instead of treating windows as independent.

Fixed SHA256 ranking seed tag `20261006-grouped-transfer-v1`. Within each sequence, approximately 20% of target tracks are selected as outer held-target tracks by hash ranking. For each selected track, approximately 20% of its windows (at least one) are outer evaluation anchors by separate ID hash ranking. Those same anchors are evaluation for both arms. The random-window arm can train on other windows from held-target tracks; the grouped arm cannot. This deliberately tests correlated-target exposure. An ordinary random split with all held-track windows used for evaluation would leave no same-target contrast and is not used.

Inner calibration is shared by both arms: approximately 10% of the remaining target tracks by sequence/hash, all their windows. Their entire identities are excluded from both adaptation training arms. Outer anchors and inner calibration do not select model hyperparameters or split seeds. Primary outer evaluation is fixed epoch80; inner-calibrated constrained checkpoint is separately reported under original guards. Original CAL2879 never chooses these new head checkpoints.

Grouped training uses all remaining windows. Random arm is sampled without replacement from its eligible pool to match grouped training counts EXACT per sequence x original stable/transition/change loss-group. Sampling hashes never use candidate quality, ADE/FDE/J, future coordinates or model output. Existing loss-group labels are used only to match training composition, not outer identity/window selection or model input. Source exposure, representation/finite optimization and conditional distribution remain coupled; this is not pure shared-parameter causality.

Common training intersection supplies BOTH identical state32 mean/std (production float32, float64 accumulation, floor .001), physical native-group scales and original loss class weights. No outer/inner/CAL/DEV coordinates fit statistics. Fixed model is approved 91596 uniform dual rawpersistent experts with endpoint-preserving projection in forward/backward, float32 head/base -> float64 cumulative output, original quality/assignment/regularizers/protection+.1P/nativeK20/clip/LR/80epochs unchanged. Head weights initialized paired from scratch, never saved-head warmstart; differing train examples are the registered factor. Frozen old decoder weights/buffers exact.

## Leakage limitations and audit
Target disjointness does NOT establish disjoint neighboring pedestrians or scene/time context. Audit known20-frame interval overlap and report all ambiguous-time windows separately; do not claim zero overlap where start_frame=-1. Distinct person identities in the same recording may share context. A subsequent purged recording/time or scene-family study must explicitly address this, not infer from target split alone.

Hotel/university/zara are only3 families; scene counts are extremely imbalanced. Subsequent scene study needs common evaluation anchors with some other within-family windows available only to random control, matched training budgets, a separately registered split and no reliable 3-scene cluster significance claim. No scene split trained here until precise audit-supported protocol.

## Decision map
- Random exposure beats grouped: supports correlated-window dependence, but distribution/sample diversity/finite optimizer differences remain alternatives.
- Grouped improves same-path coverage stably with motion protection: supports transferable new-head response within this pool; CAL failure then motivates candidate-state/context/kinematic distribution checks.
- Grouped fails: prioritizes transfer/conditioning research, not proof that shared head interference caused failure; no automatic MoE/scorer architecture launch.
- Both fail or evaluation groups too small/unmapped: do not reinterpret as causal evidence; scientific gate STOP if contrast cannot answer the question.

Three paired model seeds are not three independent datasets. Confidence intervals, if reported, cluster evaluation target tracks and use paired same-ID outcomes; sequence/scene stratification and per-scene effects remain descriptive. The native network's prior exposure, window correlation, repeated CAL and3scene limitations are explicit.

## Inline counter-review1 (same assistant, not external reviewer)
Major correction: old split is ordered, not random. Major correction: held target identity differs from unseen social context. Major confound: group failure is not uniquely head interference. Major leakage risk: common statistics must use the common adaptation-train intersection, not full original FIT. These are incorporated. Identity or matching failure is a blocking condition, not a reason to drop difficult windows or relax quality/motion thresholds.

No new experiment result is asserted in this design. Source lock follows completed implementation, before preflight/launch; thereafter scientific edits require preserved provenance and revalidation, not silent digest replacement.
