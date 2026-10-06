# Experiment 2 preregistration: recording-time exposure
Status: DESIGN_IMPLEMENTATION_PENDING, not a running training worker. No production ETA yet.

## Question, fixed before any second-study quality output
Does exposing the adapting new head to other pedestrians/windows in the SAME recording-time interval improve complete trajectories, beyond what a purged-time head learns? Experiment1 removed target identities but retained almost all same-recording interval overlap; it cannot answer this question.

This is an exploratory newhead-only contrast on existing FIT25912. Frozen F16 previously saw these data. It is NOT unseen whole-network validation and is NOT a pure causal test of shared-head interference. Only training exposure differs. No new optimizer/loss/assignment/scorer/gating/unfreeze/data permission is inferred.

## Deterministic split, no selection by quality
Canonical sequence plus verified source-frame index from existing fingerprint metadata. Source metadata generator verifies each matched trajectory covers 20 consecutive source indices. Block width is FIXED50 source indices, not an inferred50 video frames or seconds. No width/seed/block search.
Hash tag: 20261006-grouped-time-v1. Rank strings with SHA256(tag|purpose|sequence:block).
For each sequence, candidate blocks are those containing at least one full20-index known-time FIT window. Choose max(1,round(.2*blockcount)) outer blocks by purpose outer-block; choose max(1,round(.1*remainingblockcount)) inner blocks by purpose inner-block. At least3 training blocks per sequence are required.
Outer evaluation: within each selected outer block, max(1,round(.2*N)) fully contained windows, ranked by tag|outer-anchor|sample_id. Same IDs in both arms. All full windows contained in inner blocks form shared inner calibration.
Grouped training: all uniquely mapped known-time windows whose [start,start+19] intersects neither ANY outer block interval nor ANY inner block interval, across ALL pedestrians in that sequence.
Random training pool: all uniquely mapped known-time windows not evaluation anchors and whose full interval intersects no inner block interval. Other outer-block windows may be trained.
Match random training to grouped count EXACT per sequence x original stable/transition/change loss group using hash tag|random-training|sample_id, without replacement. Common training intersection ONLY supplies both identical statistics/classweights/scales.
No outer/inner/original CAL quality, native errors or future coordinates choose blocks/windows/budgets/scalers. Existing loss group is used only for composition matching and loss, never prediction input.

## Unknown times and boundary windows
All original25912 IDs stay in registry with explicit reasons and arm inclusion flags. Unknown/ambiguous time rows cannot prove temporal disjointness: held out of this VERIFIED-TIME contrast, never declared unpredictable/difficult or deleted. Full trajectories crossing selected reserved interval boundaries are purged, not silently truncated. Report exact retained/unused/boundary/unknown counts and comparison eligibility; first study still covers broader pool. No claims about all25912 time-transfer from this eligible subset.

## Pre-training feasibility gates
No result-driven block change if these fail. Both arms training>=10000 windows, common training>=5000, outer>=500 and change>=100; shared inner has all3 loss groups, each sequence present and>=3 unreserved blocks. Each arm EXACT sequence x loss-group composition. Grouped outer full20-index same-sequence overlap must be ZERO among verified intervals; random overlap must be>0. Audit target exposure separately; repeated target identities may occur across disjoint time blocks, so temporal-disjoint is NOT target-disjoint. All IDs, frames, and interval reasons audited; strict mismatch or insufficient contrast STOP for human decision.
These are feasibility gates, not a weaker success definition or permission to discard hard trajectories.

## Model and scientific protocol
Use BOTH identical91596 uniformdual rawpersistent heads with approved endpoint-preserving acceleration projection in train forward/backward. Current state32/context/mode/observed-token attention, clocks, native clip, float64cumsum, originaltotal+.1P, hard winner, balancing/regularizers/protection, old LR/batch/order80epochs x3pairedseeds and nativeK20 slotorder unchanged. All oldF16/wholedecoder/scorer exactfrozen. Paired initialization from scratch, no warmstarts. Both fit identical common-train-only rawstate/scales/classweights. Inputs only past16/frozen8tokens/nativebase/speed; futureGT/groups/logits never predictionforward, original groups only loss.
Original CAL2879 descriptive posthoc only; no DEV/test/otherfold/externaldata. Primary fixedepoch80 on commonouter; inner-selected constrained/legacy secondary and never selected by outer quality.

## Analysis / alternatives
Coverage first: SAMEcandidate ADE<=.2 AND FDE<=.4; J=minKmax(ADE/.2,FDE/.4). GT-free ALLK20 complete-link .10meanpath/.20endpoint effectivegood next, not maximization/copy counts. Independent minADE/FDE cannot be stitched. Single Jclosest early/middle/late path, rescue-lost, effectivegood, distinct, and five pooledALLK motion dt.4; inheritednativeargmax is not new scoring probability.
Report arm/seed/scene/sequence and denominator; intervals50-wide, only6 recordings/3 highly imbalanced families. Exploratory hierarchical bootstrap seeds and outer sequence-block clusters, with target-track clustered sensitivity separately; no independent scene CI or multiple-CI decisive causal claim. Correlation between blocks and recurrent pedestrians remains.
Random better supports related-time exposure, not unique neighbor/social causality; training composition, sample diversity and finite optimization remain alternatives. Both arms weak does not warrant automatic newloss/MoE/scorer. Grouped credible improvements yet CALbad motivate distribution shift; not preselected.
All existing major,coverage/duplicate/hardphysics, original5%ADE/FDEchange and ALLfive-group-motion1% gates unchanged; .3/.5 sensitivityonly. No baseline/SOTA/psychological-intent/unique-decoder claims.
Inline counter-review1 (same assistant, not external): blockpurge can reduce training diversity; exact scene×group matching cannot match every conditional state. Time selection changes outer cohort vs experiment1; no direct paired between-study ranking. Unknown eligibility bias and frozen-F16 exposure preserved. Critical mapping/resource/contrast failure STOP, not revise block width or thresholds to pass.

## Implementation / resources / follow-up
All code/cache/log/models/results/failure only experiment2_recording_time; inherit prior digests unchanged then expand previous completed artifacts. Do not edit experiment1 locked science/results/models. No worker currently.
Actual ID/resource/source/paired nativeidentity/frozenfiniteK20 GPUgradient/both2epoch FITinner-subset paramsboolscalers optimizer allRNG history calibration EXACT/SHA-skip preflight mandatory. Resource process6GiB and system physical+commit each6GiBreserve with longterm~17GiBavailable; stats256, batch128. Bounded model/output/temp/disk estimate and throwaway4warm32measure benchmarks before actualETA.
Only allgatesPASS then unique0trigger OSworker/start_independent byte-lock, no active same task, one PID/parent/currentstage/stderr startup. One50%elapsed check then one actualcompletion node, no conversation polling. Every3formal fullreview; study1/3 completed, second not started yet.

