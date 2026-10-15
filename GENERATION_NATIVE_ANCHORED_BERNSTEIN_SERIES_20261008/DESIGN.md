# Preregistered single-factor design

Status snapshot DESIGN_ONLY. AI-assisted local empirical research; inline ARS reviews are by the same assistant, not external peer review. Human instruction is specific continuation permission; no extra Phase-1 permission loop is needed within this scope.

## Evidence and research question

Previous native anchored arm: outer change 110/253 native ->111 for all three seeds, non-significant exploratory coverage interval; inner166/497->165, CAL243/700->243/244/245. All36 final native-arm group cells failed only pooled acceleration mean (+2.61 to3.05%). Small independent per-time position corrections did not ensure motion safety. This motivates, but does not prove, a time-structure mechanism.

RQ: Does bounded Bernstein temporal coupling, compared with pointwise bounded residuals under the identical native-anchor protocol, improve complete-path coverage with all five motion measures protected?

FINER F4/I4/N3/E5/R4. Narrow paired quantitative exploratory comparison using existing de-identified trajectory data, no new participants or collection; no claim of new IRB approval. Alternatives considered: add acceleration loss (outside current specific loss authority and prior physical-feedback failure); widen amplitude/epochs (local sweep, not selected); temporal coupling (selected identifiable output-form intervention). Reporting is reproducible computational experiment, not a clinical CONSORT/PRISMA study; local preregistration precedes outcomes, not public confirmatory registration.

## Identical parts

Original target-track grouped_train18909, outer1035 primary fixedepoch25, inner1856 and originalCAL2879 posthoc only. Seeds42/43/44. Frozen native20 cache/P2 mean8 observedtokens96D. Bottom4 old frozen logits stable original-slot ties, selected physical slots sorted ascending. Other16 exactly unchanged; K20 in both arms. Only shared120D input observed96+immutable selected path24, MLP120->128->64->24 GELU,25304 trainable parameters, final-layer zero. Train-only common float32 feature / float64 meanstd, std<1e-6 ->1, float32 scaler buffers. No futureGT/groups/IDs/scene/logits in head forward; logits only fixed slot mapping. Whole old F16/decoder/scorer not instantiated, trained or optimizer accessed, protected by relevant source/native-input SHA (not a whole-old-model GPU replay).

Original hard winner among FOUR refined paths: hardmin(ADE+.5FDE), plus six-pair mean squared hinge margin.3 weight.05. No fixed16 enters training assignment. AdamW lr.001 wd.0001 batch256 shuffle/drop_lastFalse25epochs/no gradclip. Inner/CAL do not select checkpoints. No protection/newphysical loss, new supervision, assignment, scorer or warmstart. Bound.20 and quadratic envelope unchanged, no tuning.

## Sole factor and mathematical contract

For raw output vectors z_i (i=0..11), u_i=tanh(z_i)/max(1,||tanh(z_i)||_2). At output t=1..12, s=(t-1)/11.

free12: w_t=u_(t-1).

bernstein12: w_t=sum_i C(11,i) s^i(1-s)^(11-i) u_i.

Both r_t=.20(t/12)^2 w_t; output=immutable native anchor+r, float32 head then float64 transform/addition. Both register the SAME float64 12x12 Bernstein matrix for paired buffers; control uses identity instead. Coefficients are nonnegative and sum to1. Hence ||r_t||<=.20(t/12)^2, endpoint<=.20, meanpath<=.0752314815. No altered evaluation epsilon. First and final output controls are identical for GIVEN raw weights/inputs (B first/last one-hot); learned final endpoints need not match between arms. Constant controls are an exact fixed point at the endpoints and numerically within1e-15 elsewhere.

This is a deterministic output-time coupling, NOT event knowledge, extra observations or changed candidate credit. Matrix is full rank on 12 sampled times (verify); unbounded linear expressive span alone need not change, but bounded control geometry, reachable trajectories and finite optimization do change. Same nominal25304 does NOT prove equal effective capacity/learning difficulty. Degree11 follows exactly the12 existing controls, no degree/kernel/width/coefficient search. Continuous Bernstein curve is smooth, but original native path plus envelope/boundary/learned endpoints means neither acceleration nor jerk protection follows analytically. Verify actual trajectories; never claim smoothness guarantees all five measures.

Save bounded controls and fixed matrix in every scientific prediction NPZ. Independently reconstruct residuals, endpoints, bounds and fixed16, alongside quality and motion. Report residual-only first/second/third finite differences dt.4 as descriptive mechanism checks; no outcome-based selection.

## Gates and outcomes

Before launch: exact inputs/stats/slotIDs; NumPy convex basis, constant/alternating controls/endpoints/bounds/zero fixtures; actual paired GPU8FIT256 lawful phased gradient connection/positive finite, canonical AST objective/all gradients exact, nativecache unchanged, old native-control nonzero output alignment exact; double parameter finite differences eps1e-4/1e-5 with winner switches recorded and original1e-7+1e-3|derivative| tolerance. Both REAL513FIT+12inner SUBSET2epoch continuous/interrupted ALLparams/bool/scaler/basisbuffers, optimizer, TorchCPU/CUDA/loader/Python/NumPyRNG/historyinnerQ EXACT. CompletedSHA skip and wrongsource/output/resume-contract refusal. This is not full18909 twoepochs or float32 quantization derivative proof. Saved actual outer128 full quality/complete-link/five-motion independent arithmetic and frozen newhead values. Initial/final relevant source SHA. 36throwaway4warm32measure eacharm; no saved benchmark model/outcome choice. process<=6GiB, physical+commit each reserve6GiB, bounded batch256/forward128/statistics; disk/input/output scale check. Any substantive failure STOP with preserved provenance, not threshold relaxation.

Completion: soleworker+exit SUCCESS; six25history/150rows, six checkpoint manifests bind sources/IDs/statistics/objective/optimizer+ALLRNG;18scientific jobs and all output SHA, full saved arithmetic independent full SciPy GT-free complete-link meanpath.10/endpoint.20 on ALLK20, samecandidate ADE<=.2 AND FDE<=.4/J<=1, rescue/lost/retained/ownzero, effectivegood/distinct, single Jclosest early/middle/late, original assignment stats. All five pooled motion per seed/group: speedp95,accelmean,p95,jerkmean,p95 dt.4.36 paired targettrack+3seed bootstrap records/144scalar CIs2000rng20261008. Report free12-native, bernstein12-native, bernstein12-free12. Free control may repeat prior native fresh run; actually check arrays after completion, do not count independent generalization. Saved arithmetic not all gradients/models repeated replay; CPU completion review only new folder.

Major all unchanged: changeADE andFDE each>=5% gain, overall/stable/transition<=1% harm,3seeds consistent, native/fresh ADE-FDE jointCI exclude0, samepathcontrol1%, coverage/duplicate protection AND everyseed/group allfive-motion<=native1.01. Sensitivity.3/.5 not replacement thresholds. Native coverage first, GT-free effectivegoods second not maximize/copies, scorer last. No baseline upgrade on execution PASS. F16seen, targetdisjoint not contextdisjoint,3scene/window/seed correlation, repeatedCAL and adaptive bottom4 limitations; not understanding movement/SOTA/independentwholemodel/decoderroot cause.

## ARS Devil's Advocate checkpoint1 — PASS with limits

Major: smoothing may simply suppress useful corrections or alter bounded capacity/optimization, not remove a proved cause. Retain control, same-weight endpoint and constant fixtures, full coverage/lost and allmotion gates; report non-improvement rather than choose a new kernel. Major: coverage with +1 window is weak, high loss reduction is not sufficient; use same complete candidate and all groups/CI. Minor: analytic matrix smoothness may not transfer to native path; actual physical gates unchanged. Strongest counterargument: modest protected improvement could be contraction back to native with negligible new support. Report residual norms, rescues/lost and original5% gates to test that. No critical protocol issue identified. No claim of external review. DA2/3 required at final analysis.

No new loss/scorer/oldmodule/data, no fourth local variant, no changing source after lock. After launch exit dialogue; unique half->completion COUNT1 updates. Every three formal completed experiments require full hypothesis/control/CI/coverage/motion/FITCAL/capacity/history review before a new direction.
