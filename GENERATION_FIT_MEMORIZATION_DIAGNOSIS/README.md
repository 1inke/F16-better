# Fixed FIT memorization diagnosis

Not a formal cycle7 experiment. One fresh seed7 head (25158 trainable parameters), original F16/native decoder/scorer completely frozen. Native K20 + three-anchor correction + velocity clip1.5*(observed speed+.03). Original balanced joint-winner, diversity/residual/smoothness/protection.1 loss unchanged. All FIT25912 class weights/protection scales maintained.

Pre-registered FIT256 and disjoint CAL256 sampled independently of outcomes with RNG seed20261002. Fixed diagnostic budget120 epochs, batch32, original LR/weight decay; no sweep or checkpoint selection. CAL only describes generalization, no dev/test access. Full online observed-history feature extraction cached under frozen encoder: after four learning steps cached/online output exact equality verified. Features contain no future GT/groups/logits. GT used only original training loss and diagnostic errors.

Report epochs0/1/5/10/20/40/60/80/100/120: objective, independent minADE/minFDE, same-path joint, all-slot and joint-winner clip saturation, head layer gradient norms. Objective includes original regularizers, so this is not pure unconstrained memorization. A small-set improvement is not intent understanding, reliable generalization, a new baseline or a major result. Failure to fit alone cannot prove capacity insufficiency; learning curves and gradients must be interpreted jointly.

All new scripts/checkpoints/log/results stay here. Prior10314 files locked and verified before/after; source hashes locked. Worker saves epoch-boundary optimizer/RNG/parameter checkpoints and resumes next epoch. No assertion of production-level exact-resume audit: this is a single diagnostic, not the next formal paired experiment. Formal experiments still require both-arm exact-resume preflight.

On-demand Windows task Trajectory-FitMemorization-Diagnosis uses independent worker and exclusive byte lock, no recurring trigger. ETA3–6min/planning5min, one60%-time query then revise actual remainder. Machine must remain awake/logged in. Codex automatic analysis requires application availability.
