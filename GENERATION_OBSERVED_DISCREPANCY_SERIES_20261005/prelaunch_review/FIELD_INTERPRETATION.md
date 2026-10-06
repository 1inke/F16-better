# Prelaunch field interpretation — observed discrepancy

Single-assistant local review, not independent external review.

- Generic zero is ACTIVE raw nativecandidate-state32; mean_std is raw state32 minus last-observed velocity reference in its first30coordinates. BOTH state branches and learned sigmoid clocks are active. Neither control is masked or Gaussian.
- Common original rawFIT mean/std floor.001 is used for BOTH. Last2 acceleration magnitude and original native first/last signed heading unchanged. Reference is conditional translation, not rotation or new real-world/future information. Optimization/input distribution differ; nominal parameters do not imply matched effective capacity.
- README/DESIGN DESIGN_ONLY is the preregistration snapshot. Current control runtime and actual worker determine running status; no production started before all real gates.
- profile_sum_error is inherited field measuring abs(sum(sigmoidprofile)), not a failed zero-sum test. Sigmoid persistent profile is intentionally not zero-sum; drift field in saved pulse arrays is total endpoint correction incl pulse, not drift-only.
- Raw control sameweights is compared by executing only old immutable class AST without old module aliases/writes. Freshpaired initialization/nonzero synthetic last layer, not failed head teacher/warmstart. Production architecture/seeds repeat prioractual; exact results must be checked, not presumed or counted independent.
- State/scalers only immutable nativeprediction and FIT25912*20 points, observed past[:16], frozen tokens16:784/rest0. No futureGT/groups/logits input, no new loss/event supervision/scorer/unfreeze/data.
- Gradient staging respects zero last head: inspect positive upstream after actual updates. Keybias cancels analytically under softmax; zero/roundoff not effective signal.
- Quality closest normalizedadequacy slot vs temporal ADE+.5FDE jointoracle slot differ; no temporal segment splice. Inherited nativeargmax is old slot, not new probability/scorer/intention decision.
- Samecomplete ADE≤.2 AND FDE≤.4 coverage first; GT-free complete-link .10meanpath/.20endpoint effectivegood second, not maximize or duplicate; .3/.5 sensitivity only. All strict motion1%/major/seed/nativefreshCI conditions unchanged.
- ~17GBavailable hostRAM persistent planning; process6GiB and systemphysical+commit each6GiBreserve. Reference computed perbatch, no unbounded allfeature copies/OS changes/userprogram termination.
- 3scene/windows dependence/F16seen/CALuniversity/repeatedETHDEV and prior failures retained; no SOTA/independentgeneralization/intent/unique decoder cause claim.
