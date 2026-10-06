# Fixed FIT learning diagnosis and next direction

Memorization worker SUCCESS21:40:23HK, end-to-end86.65s (training/final audits48.67s), not estimated3–6min. One fixed seed7 FIT256/CAL256 descriptive diagnosis, not a formal trial. Independent final objective and all reported group metrics recomputed within1e-6 from checkpoint; source locks/all old10314 files/entire original F16 and decoder frozen PASS. No DEV/test. Initial/final saved results unmodified.

| Split/group | Initial ADE/FDE | Final ADE/FDE | Interpretation |
|---|---|---|---|
| FIT overall256 | .176169/.300121 | .149176/.236403 | gains15.32%/21.23% |
| FIT change61 | .237616/.409982 | .209776/.348914 | gains11.72%/14.90% |
| CAL overall256 | .196514/.327677 | .200516/.331918 | harm2.04%/1.29% |
| CAL stable131 | .142599/.234761 | .155832/.256222 | harm9.28%/9.14% |
| CAL transition69 | .232958/.409937 | .229856/.396808 | gains1.33%/3.20% |
| CAL change56 | .277731/.443678 | .268890/.429040 | gains3.18%/3.30% |

FIT objective .356945 -> .291595; CAL .418465 -> .448556. The head learns, but does not nearly memorize these256 observations and does not transfer overall improvement to the disjoint CAL256. Training curves still decline at120; cannot declare convergence/optimization solved or capacity sufficient. This is the original regularized/protected objective, not unrestricted interpolation. Different split group composition and only one seed limit comparisons. No major/generalization/intent claim.

No all-slot/winner velocity clipping at recorded learning-curve points, and final raw/limit maximum .20113FIT/.17791CAL; does not cover every intervening training step or every fullFIT sample. Thus clipping is not active at these measured points, but cannot globally dismiss its effects. All head layers have finite nonzero final gradients.

Read-only gradient decomposition, same256 subsets, no optimizer: final FIT winner/regularizer/protection/total gradient norms .10594/.00004260/.06340/.12602; winner vs protection cosine .04753, winner vs regularizer -.23936 but negligible regularizer magnitude. CAL winner/protection cos .48177. These points do not support blanket protection-loss cancellation; cannot prove it never occurs elsewhere. Stable/change winner-gradient cos initialFIT-.32923, finalFIT-.43456, finalCAL-.27438; transition/change finalFIT+.10107, CAL-.33946. Local group conflict is a hypothesis, not a proven global cause, and GT-derived groups must never be inference routing inputs.

Next major-direction hypothesis: replace global shared nonlinear MLP correction with an observation-similarity basis and a trainable linear anchor readout, keeping original loss/guard/nativebase/clip. Compare fixed Gaussian neighborhood basis to matched-size random Fourier feature basis, all statistics and centers determined by FIT observed features only. This changes feature function class and update coupling, not merely a loss coefficient/module freeze. Could overfit or dilute motion-change effects; no promised benefit. Detailed preregistration in new GENERATION_LOCAL_BASIS_SERIES/DESIGN.md. No formal training launched yet; paired GPU preflight and exact optimizer/RNG/calibration resume remain mandatory.
