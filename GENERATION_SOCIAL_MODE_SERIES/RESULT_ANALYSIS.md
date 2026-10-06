# SOCIAL_MODE result analysis
Normal worker SUCCESS2026-10-02T05:01:02.403988+08:00, actual17m45. No active oldworker; OS taskReady. SixNEW80epoch trainings,18policies not18independenttrainings.
Independent18SHA/18alloriginaldecoderfrozen/18explicitIDs1516/finite1516x20x12x2/allstoredmetrics recompute PASS. Production frozen audit and SUCCESS PASS. First independent diagnostic device=str invocation failed, corrected torch.device CPU PASS; not production failure.
## Constrained results
| Arm | Group | minADE | minFDE | Samepathjoint | accelmean |
|---|---|---:|---:|---:|---:|
| static | overall | .217975 | .399963 | .431590 | .101673 |
| static | stable | .174672 | .315526 | .343046 | .104584 |
| static | transition | .275066 | .510721 | .546310 | .093805 |
| static | change | .367311 | .691721 | .738985 | .095717 |
| conditional | overall | .218010 | .399479 | .431430 | .101120 |
| conditional | stable | .174613 | .314808 | .342637 | .104002 |
| conditional | transition | .274810 | .511180 | .546019 | .093326 |
| conditional | change | .368086 | .691406 | .740150 | .095234 |
Native change ADE/FDE .372175/.696326: gains1.10%/.71%, below5%. Overall accelerationmean .101120 vs native .099469, +1.66%, exceeds motion protection. No breakthrough/newbaseline.
## Change deltas and exploratory CI
| Reference | metric | delta | CI low | CI high |
|---|---|---:|---:|---:|
| native | ADE | -.004089 | -.006693 | -.002126 |
| native | FDE | -.004921 | -.012003 | -.000328 |
| native | joint | -.006481 | -.012274 | -.002652 |
| fresh static | ADE | +.000774 | -.000438 | +.002221 |
| fresh static | FDE | -.000315 | -.002612 | +.002266 |
| fresh static | joint | +.001166 | -.001127 | +.004041 |
Native CI improvements are exploratory on repeatedly-inspected ETHdev, not independent generalization. All freshstatic CI crosszero, so explicit candidate-conditioned socialgate has no proved increment; cannot conclude significant harm either.
5%/nativeANDcontrolCI/motiondistribution FAIL. Protectedgroups/eachseednativeimproves/feasible/coverage/duplicates/hardmotion/samepath PASS. Frozen scorerweights don't imply ranking unchanged with changedpaths; no rankingclaim.
## Decision
Do not promote final/unconstrained/dev-selected models, loosen guards, or rerun completedseeds. Cycle5completed2/3. Next fixed FIT-only residual social representation singlefactor, no gate, documented separately. After third mandatory consolidated cycle5review and bigdirection decision.
