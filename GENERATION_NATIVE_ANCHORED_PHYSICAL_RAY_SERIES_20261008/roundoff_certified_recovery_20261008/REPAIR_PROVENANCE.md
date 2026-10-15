# Numerical recovery and technical decision — 2026-10-08

## Outcome
Routine floating-point handling has been resolved by the assistant, without asking the human to decide a tolerance. Fresh complete preflight and current-source technical review passed in roundoff_certified_recovery_20261008. No production task, worker, PID, formal training, scientific result or halfway node was started. Formal series remains 2/3.

The measured production ETA is 5–15 minutes (planning 8), plus an estimated 2 minutes of saved-results review. At 09:19:30HK the fixed six-hour window had only 10m48s remaining. Therefore a conservative launch cannot fit 09:30:18HK. Do not reduce ETA to force a launch or reset the window. This is a time-budget decision, NOT a wait for human numerical approval.

## Reproduced original failure
A fresh throwaway eight-step head, same seed/inputs, reproduced exactly the original maximum excess 3.864602421585557e-17. Two acceleration constraints:
- sample1513, selected candidate0/native slot4, difference step9: native norm1.1641532182693481e-8, actual1.1757947525586774e-8, excess2.106635790781469e-17, alpha2.7172480891932775e-6;
- sample1630, selected candidate3/native slot19, step5: native1.4551915228366852e-9, actual1.4697434767110762e-9, excess3.864602421585557e-17, alpha1.0650196394165768e-7.
No speed/jerk violations in this fixture; all five original pooled motion metrics passed literally in overall/stable/transition/change. Complete tensors/IDs/controls/alpha and report retained in the rejected recovery directory. These are preflight observations, not production benefit.

## Repair chosen
Keep the parent closed-root/interior projection forward EXACTLY; actual GPU comparison verified bitwise output and alpha equality. No native fallback, new loss, changed margin, derivative-baseline floor or coefficient sweep. Instead, the independent pointwise arithmetic audit uses a scale-dependent forward-error enclosure gamma_n=n*u/(1-n*u), u=eps64/2, n=8*order+16, times the absolute binomial finite-difference position scale / dt^order. Save raw exceedance counts/magnitudes and ratio to enclosure. Reject substantive excess and non-finite data. A fixed zero-native/nonzero-path defect1e-7 was actually rejected.

This certificate is a numerical claim, not literal bitwise pointwise<=1.01. The original five pooled research motion criteria still compare actual saved arrays literally against1.01native; no epsilon. No major coverage/error/seed/CI protection criterion changed. Finite precision and operation order are documented by [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html); adjacent representable spacing is documented by [NumPy spacing](https://numpy.org/doc/stable/reference/generated/numpy.spacing.html). Those sources do not validate this model's scientific efficacy.

## Gradient verification and corrected interpretation
The first16 original physical FD checks have analytic=FD=0: true but not informative nonzero verification. A supplemental actual nonzero-primary check used62 eligible windows, chose16 with greatest alpha only for numerical stability (no scientific sample exclusion or outcome selection), and used the final-bias gradient direction. Analytic derivative2.2661666067898317e-5:
- eps1e-4: FD-5.502818201108539e-6, error2.8164484269006857e-5, two ray-active constraint changes;
- eps1e-5: FD1.771449653631407e-5, error4.947169531584248e-6, one constraint change;
- eps1e-6: FD2.2661872378648695e-5, error2.063107503778405e-10, no constraint change;
- eps1e-7: FD2.2664092824697946e-5, error2.4267567996281536e-9, no constraint change.
No quality-winner switches at these steps. Thus decreasing-step convergence on a stable ray face supports the analytic derivative. Do NOT claim nonzero PASS at the two larger nominal steps or float32 quantization proof. No training hyperparameter was changed.

The attempted native fallback recovery is preserved but rejected for unnecessary forward-map change. Initial commentary attributed large-step FD problems to fallback; that causal attribution was incorrect: the unchanged parent projection also showed them. The measured active-face changes explain why the large-step central quotient was not a local derivative check. This correction is explicit; no source or failure evidence overwritten.

## Fresh gates actually completed
1166 relevant sources inherit all1151 parent digests unchanged (not all historical34k files). GPU8FIT256 both arms: paired initial weight/scaler/basis buffers, zero native identity, finite positive connected gradients, canonical AST scalar/all-gradient EXACT, independent NumPy scalar, unchanged native cache, actual128 full quality/original SciPy complete-link clustering/pooled motion independent max0. Both real513FIT+12inner subset2epoch continuous/interrupted ALLparameters/buffers/optimizer/TorchCPUCUDAloaderPythonNumPyRNG/history EXACT and doneSHA badsource/output rejection. Not full18909 twoepochs, oldF16 model replay or formal training. Final-source review repeats saved math and SHA, verifies substantive-defect rejection and nonzero FD convergence.

Benchmark no saved scientific model: FIT256 unprojected.0122495937s/physical.0220353250s; inner1856 full3port metrics3.1588123/3.1584022s; nominal compute5.15729min, not production guarantee. Preflight private2.710583GiB/RSS1.980984, minimum physical14.879967/commit24.285645GiB. Original process6GiB and both system reserves6GiB remain. No OS/pagefile changes or user-process termination.

## Inline adversarial reviews (same assistant, not external)
DA2: ordinary roundoff is not a scientific obstruction needing human choice. Certificate must not hide meaningful violations, make literal pointwise guarantees, or imply major success; retain raw counts and test original pooled gates. Native fallback changes reachable support and is not needed.
DA3: tiny alpha and many zero-native derivative budgets may render this strong ray constraint nearly identity. Positive gradient/finite-difference convergence proves numerical connectivity, not useful correction, complete coverage, transfer, or learnability. The formal contrast is not run; no3/3 claim or fullthree review. Before a new launch, use a new explicitly valid time window/current source/resource check, then soleworker and exactly one halfway/completion. No coefficient/degree/width/LR sweep.

## Ownership rule
Latest human criticism is accepted: the assistant owns ordinary floating-point, rounding, scale, meaningful-tolerance and credibility judgments, must investigate/repair/document/revalidate autonomously. Ask only when changing actual scientific objective, data/module scope, permissions, or confronting a real unresolved resource/scientific obstruction. Do not escalate a tiny raw number without its scale or numerical analysis.

