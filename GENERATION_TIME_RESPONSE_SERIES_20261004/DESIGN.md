# Pre-outcome local protocol: within-segment temporal response

## Question and counterevidence

Can observed-token-conditioned corrections improve difficult trajectories when allowed to vary within the intervals between steps4/8/12? Encoded event risk is readable, not proof of knowing correct event timing/direction/amplitude. Existing three-anchor GT-assisted constructions show that three anchors are not a universal hard bound. Tiny-set/dense-conditioning overfitting, failed time-query Top8 refinement and failed velocity-basis trials remain counterevidence. This trial tests a specific response parameterization, not decoder causality or autoregressive superiority.

## Matched construction

Both arms have frozen native context64+native-mode64 features, query128->32, key/value96->32, fixed observed/future sinusoidal time buffers, observed-token attention at each of twelve future times, and new192->128->64->2 GELU head. Last layer is zero. Expected43426 trainable parameters per arm; actual GPU count must confirm. The only arm-dependent operation is a deterministic temporal projection after common clipping: `zero` means four adjacent increments and repeats that mean; `mean_std` retains each increment. Both integrate corrections and add immutable nativeK20 trajectories. For identical weights, the projection preserves the correction sums at4/8/12 (including endpoint), to floating precision. Learned weights may diverge, so final endpoints are not experimentally matched. The control correction is piecewise linear, not the unchanged historical anchor-MLP architecture. Same nominal parameters do not equal function/effective capacity; projection changes optimization geometry.

All twenty original mode slots are retained, no early TopM pruning. Fixed-origin8-observed/12-future, dt.4s; not online acquisition or AR. Time queries never receive future positions. Prepared940 schema: focal past0:16, frozen8x96 tokens16:784, unused remainder zeros. Original decoder context/mode/geometry/scorer and entire F16 remain frozen. No new loss/event supervision, guard, clip, width, LR or epoch sweep. Candidate ranking cannot improve by a newly trained scorer in this trial.

## Data, loss and selection

Original FIT25912/CAL2879, original balanced hard joint winner, diversity/residual/smoothness, protection.1, constrained CAL selection, LR/order/budget and native velocity residual clip1.5*(speed+.03), seeds1/2/3 and80epochs remain unchanged. Existing cache reuse is legitimate because F16 is entirely frozen. Actual online replay and cached feature/path alignment must pass. Future coordinates only training loss/posthoc error/event labels. No heldtest, otherfold, external data or paid cloud. No GT groups/logits in forward. No old failed head/teacher warm-start.

## Verification before production

GPU8 actual FIT steps for each arm: paired initial full weights/buffers, exact native output identity, positive finite query/key/value/head gradients after zero-head activation, no old gradients/whole old decoder exact, finite128x20x12x2 output and common bounds, attention normalization/observed-input sensitivity, block sums/endpoints and segment-constant projection test, ID/split/online feature alignment/source/all-prior-result SHA checks. Both arms actual2epoch continuous/interrupted exact parameters/buffers/optimizer/all RNG/history/calibration and complete-seed SHA-verified skip test. Workload benchmark is throughput only, no outcome-based choice or saved benchmark model. All new artifacts and failures stay here.

## Outcomes and interpretation

Retain all major gates: formal change minADE/minFDE each>=5% improvement, protected overall/stable/transition<=1% harm, each seed improves, native and fresh-control CI exclude0, same-path joint control<=1% harm, coverage/duplicates/hard-motion safeguards, all groups speedp95/acceleration mean,p95/jerk mean,p95<=native1% harm. Existing saved prediction/group metrics independently recomputed; eighteen checkpoint SHA, explicit1516 IDs finiteK20, frozen/source/prior files unchanged and SUCCESS required. First all-gate signal requires scene/sample-correlation and three-seed sufficiency review. Repeated ETH DEV is exploratory, not independent generalization/SOTA.

Secondary descriptive outcomes (no checkpoint choice): same-candidate ADE/FDE, early/middle/late error, strict.20/.40 and sensitivity.30/.50 same-path adequacy/rescue/harm, persistent event type/onset/direction/speed-amplitude discrepancy using the frozen prior diagnosis rule. These do not prove causal cue use. Native argmax labels stay fixed; no scorer increment claim.

Public existing trajectory observations only; no new human recruitment, re-identification or fabricated blind judgments. Institutional approval/exemption is not claimed. Local protocol fixed before this trial's outcomes, not externally registered. Academic-research-suite evidence/alternative/control checks inform this design.

One independent production launch; one50%-estimated-time inspection, then adjusted completion deadline. Completion analysis followed by human discussion; no automatic follow-on. Every three new approved formal studies require a consolidated direction review; this study begins a separately recorded response series without rewriting unaudited historical cycle counters.
