# Response-head feedback and fixed tiny-FIT learning diagnosis

Human 2026-10-05 "开始你说的这个实验" approves the preceding proposal.
AI-assisted local computational diagnostic; not a new formal response trial.
Research question: does the original quality/regularizer/protection signal reach
the actual pulse head, and can a fresh shared past-only head fit fixed examples?
Distinguish output-space reward, parameter feedback, training-set fit and CAL
transfer. No unique decoder/optimizer/observability cause is assumed.

## Fixed populations and pre-registration

Keep original FIT25912/CAL2879 IDs and all-FIT weights/scales. Seed20261005 selects
128 of the 576 previously classified cue-first-compatible/coarse-response-covered
but whole-path-bad FIT windows. Choose 128 random other FIT windows (excluding the
entire targeted pool). Together the fixed tiny-FIT256 is an enriched diagnostic,
not a representative population or a causal sampling comparison. CAL contains all
137 first-compatible windows (including all77 coarse failures) plus 256 random
other CAL windows, total393. Targeted sampling uses existing posthoc labels solely
for diagnostic sampling; no labels, groups, GT or logits enter forward.
No DEV/test/otherfold/newdata. Windows/tracks can overlap; CAL is not claimed
independent, its original single-university and F16-seen limitations remain.

## Unchanged model and training objective

Import immutable original PulseResponseHead. Original F16/whole decoder/scorer
remain exact fixed, immutable nativeK20, past16+frozen observed8x96tokens only,
3anchor-nativebase/commonclip1.5*(speed+.03), original hardADE+.5FDE/balancing/
diversity/residual/smoothness/protection.1/guard.01 and allFIT scales unchanged.
Two original arms: fixed clock versus learned candidate-observation clock. Same
39590 nominal parameters, paired weights/buffers/zero-output native identity;
fixed clock rows have zero data-gradient, not equal effective capacity.
Six fresh tiny diagnostic runs seeds7/8/9, 120 fixed epochs, batch32, original
constant AdamW LR.0003/weightdecay.0001/gradientclip5. No sweep/extension based on
outcomes, no warmstart. Existing constrained seeds1/2/3 may be read for NO-OPTIMIZER
gradient snapshots only, never used to initialize training. No CAL selection:
record full history and original feasibility description, always retain final.
120 epochs is a registered memorization budget, not a formal80epoch comparison.

## Measurements and interpretation

Record epochs0/1/5/10/20/40/60/80/100/120: full tinyFIT/CAL objective, same-candidate
ADE<=.2 AND FDE<=.4 coverage/raw counts; .3/.5 sensitivity only; per-window paths
for all snapshots. Targeted/reference/first-compatible/coarse and regime groups
reported separately. Use allK20 GT-independent complete-link meanpath.10m and
endpoint.20m for effective good clusters; number isn't maximized. Save speed,
acceleration/jerk mean and windowp95 dt.4; these are not formal globalp95 gates.
Original quality/protection/regularizer parameter gradients and layer norms,
Q/K/value/hidden/output drift/amplitude/clock rows and code-gradient allocation at
epochs0/1/20/120. Gradients at zero initialization in hidden/clock may be zero:
lastzero output delays upstream, amplitudezero delays clock; key bias cancels
softmax analytically. Actual multi-step positive gradients must be observed.
Parameter gradients are real but batch/local snapshots, not whole-population
causal proof or head-Jacobian accessibility. Mixed-enriched batch geometry differs
from original fullFIT production. Native-output quality slot allocation does not
mean shared parameters never help other slots. Include existing failed softset/
energy/pathmixture/freeincrement/timequery and tinyFIT-good-CAL-bad evidence.
FIT failure can reflect objective tradeoffs, mapping/capacity or optimization;
FIT success with poor CAL supports a transfer gap, not irreducible ambiguity.
Tiny memorization/GT oracle is not intent, deployment benefit, SOTA or major.

## Validation, resources, launch and stop

New artifacts only this folder; all prior results/models/scientific sources and
failures full SHA locks before/after. Full input and temporary scale estimate,
actual peak RSS/private; approximately17GB available hostRAM persistent constraint.
6GiB process cap, physical and commit reserve each6GiB; internal resourceguard,
no OS/pagefile changes or user program termination. Actual GPU finiteK20, bounds,
whitelist/gradients, paired state/native identity, past-only raw online replay and
native-token alignment before launch. Both arms real2epoch continuous versus
interrupted parameter/allbuffers/bool/optimizer/allRNG/history/CAL EXACT, completed
run source/output SHA-skip and source/corruption rejection. Benchmark discarded
throwaway steps solely for ETA. One OS on-demand0trigger unique task, byte lock,
realPython/serviceparent/start/stage/stderr confirmation; one50%elapsed-time check,
then realistic completion node. No conversational polling.
Completion doubleSUCCESS, all saved prediction/metrics independently recomputed,
complete six histories, allold/allsource final SHA and frozen decoder exact.
Any substantive science/source/resource/permission obstruction STOP await human;
no blind retry or science change. Then analyze/report/wait; no next training.

## Inline Devil's Advocate checkpoint1 (same assistant, not external review)

PASS WITH LIMITS. Major risk: enriched tinyFIT can be memorized without useful
prediction, or fail without disproving expressivity. Controls isolate original
clock factor only under a small diagnostic budget, not a new architecture benefit.
Selection and three seeds share windows; report denominators, no reliable sceneCI
or independent generalization. CAL transfer cannot distinguish unseen motion from
feature insufficiency. No current Critical issue once claims stay diagnostic.
Checkpoint2/3 required after results. Human confirmation is the explicit approval
of this diagnostic proposal, not permission to change loss or add GT inference.
