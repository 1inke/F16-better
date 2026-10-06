# Prelaunch field interpretation — centered FIT normalization
Single-assistant local engineering/scientific review, not independent external review.

- BOTH arms use the same active observed-centered state32. zero applies original raw FIT mean/std; mean_std applies centered FIT mean/std only to first30. Last2 acceleration/heading features AND raw scalers remain exactly unchanged.
- Registered scaler buffers state_mean/state_std first30 intentionally differ. All trainable weights and nonscaler buffers are fully paired. Do not claim all buffers identical.
- Centered features are immutable native predictions minus last-observed-velocity reference30; no new future information, GT/group/logit inputs, rotation, new velocity window, width/LR/floor/epoch selection.
- Fixed positive affine normalization with first-layer bias preserves unbounded function span, but finite optimization, initialization and weight-decay geometry change. Neither success nor failure identifies psychological intent or decoder root cause.
- Control sameweights/nonzero synthetic last layer matches prior ObservedDiscrepancy treatment exactly using only immutable prior class AST. No prior module writes, failed teacher or warmstart. Production repetition must be checked, not presumed independent.
- Both arms retain persistent sigmoid profiles and learned clocks. Inherited profile_sum_error is abs(sum(sigmoid)), intentionally nonzero, not a failed zero-sum test. Saved drift is total endpoint correction including pulse.
- Gradients staged after zero-last head updates; key bias cancels analytically in softmax, roundoff is not useful data gradient.
- Strict samecandidate ADE<=.2 AND FDE<=.4 coverage first; GT-free complete-link .10meanpath/.20endpoint effectivegoods second, not maximize/duplicate; old scorer last. Inherited native argmax is old slot, not rescored new probability.
- Quality normalizedadequacyclosest versus temporal ADE+.5FDE jointoracle use different single complete slots. Never stitch early/mid/late oracles. onset99=noevent; direction turn-only conditional denominator; clocks not semantic event onset/duration.
- Whole old F16/decoder/scorer fixed, immutable K20 and old loss/guards/FIT25912/CAL2879/LRorder80budget; no data/heldtest/newloss/eventsupervision/unfreeze/scorer changes.
- About17GB available host RAM planning; 6GiB process and each6GiB physical+commit reserve, chunk256 statistics and batch reference; no OS/pagefile/userprogram modification.
- Every3newformal requires fullreview; substantive source/ID/resource/science/permission obstruction STOP human. No fourth local normalization/profile/coeff/epoch variant.
