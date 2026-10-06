# Inline adversarial checkpoints 2 and 3

These are same-assistant counterargument checks, not independent external peer review. Checkpoint 1 is in the immutable DESIGN.md. This review follows the precommitted CLAIM_INTENT_MANIFEST.json and the completed saved-data audit.

## Checkpoint 2: challenge the result interpretation

1. **Could positive gradients establish useful motion understanding?** No. Gradients show a local learning path, not successful inference, event semantics, or complete trajectory generalization. Key bias cancels in softmax; its roundoff-sized gradient is excluded. Initial zero-head and zero-amplitude delayed upstream gradients are expected.
2. **Does FIT recovery prove adequate expressivity for every case?** No. Learned-clock recovery is only 54–58/128 targeted FIT windows. The remaining 70–74 failures still mix optimization, conditional mapping, and functional limitations. Earlier per-window GT-assisted witnesses do not prove a shared past-only head can learn those solutions.
3. **Could rescued CAL failures justify a gain claim?** Not at population level. Learned-clock CAL rescues 28/32/29 windows while losing 30/40/33. CAL total coverage, independent minADE and minFDE, and distinct-good totals worsen. Report both sides, not only rescue examples.
4. **Is the calibration subset a new generalization benchmark?** No. It is an enriched fixed diagnostic subset, from previously used FIT/CAL and a limited scene mixture. Three seeds reuse the same windows; overlapping tracks and F16's original exposure remain. No reliable scene CI, independent external replication, or SOTA claim.
5. **Can relative gradient norms identify the clock or protection as the cause?** No. Units, Jacobians and parameterization differ. Recorded quality-total gradient cosine stays high, but this is a fixed-batch diagnostic, not all samples or steps. Protection can conflict locally without globally blocking all learning.
6. **Can small coordinate error be equated with high-quality motion?** No. Final CAL mean acceleration rises roughly 33–36% for learned clock. Positional threshold success alone is not the full motion-protected quality goal. Saved mean window p95 is not formal pooled group p95.

Resolution: retain narrow conclusions—partial FIT learnability, local gradient connectivity, lack of stable CAL benefit under this diagnostic. No critical unresolved issue for those claims. Claims of a unique decoder root cause, fully solved expressivity, or useful intention understanding are rejected.

## Checkpoint 3: challenge the synthesis and next-step decision

- The primary count requires the **same candidate** to meet ADE≤0.2m and FDE≤0.4m. Independent minima are supplementary, not evidence of complete-path success.
- Native baseline, fixed-clock control and learned-clock treatment are separated. Learned clock improves targeted FIT counts versus fixed clock in all three seeds, but CAL targeted recovery is 11/12/11 versus 9/10/14: no consistent advantage.
- Candidate quantity is GT-independent clustered distinct-good count, not raw near-duplicates. CAL learned-clock distinct-good totals 229/230/230 versus native238 do not support quantity improvement.
- Original objective reference was independently rebuilt in NumPy; largest difference 7.392e-8 is bounded float32 rounding, not bitwise torch replay. All saved group aggregates and metric masks have zero difference. Worker full reclustering and frozen-state evidence are distinguished from the supplementary no-forward checks.
- Supplementary first failure is a review-only metadata misuse: actual-head cluster counts were attached to a hypothetical native prediction. Assertion rejected it; source and traceback were preserved, and native comparison now omits unavailable native cluster metadata. Science thresholds, production code and outputs were not edited.
- Resource success is actual measured peak private2.602GiB, not a future memory guarantee. Approximately17GB available remains a planning constraint; process6GiB and both system reserves6GiB remain unchanged.
- No automatic next training, new loss, supervision, data, or unfreezing is inferred from an analysis request. Response formal study stays2/3. Discussion should separate persistent FIT failure from CAL rescue/loss before choosing a new identifiable intervention; no blind epoch/LR/width sweep.

Resolution: PASS WITH LIMITS. Evidence supports the report's diagnostic interpretation, not a baseline upgrade or completed project goal.
