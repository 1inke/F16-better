# Generator residual-capacity controlled pilot

Hypothesis: the correction velocity bound 0.5*(observed speed+0.03) limits correction of abrupt-motion candidates. Only increase the bound to 1.5; physical residual/smoothness penalty coefficients, initialization, batches, 35 epochs, all loss coefficients, K20, fit/cal split and constrained calibration selection stay unchanged. Bound also changes optimization gradients, so this tests the bound setting, not a pure isolated representational capacity proof.

Three fresh large-bound correction trainings vs hash-verified same-seed small-bound balanced_protected controls from the preceding completed experiment. Reused controls are not independent replications. No dev-selected epochs/coefficients, no future/group at inference, no scorer/backbone training, no test. All18 policy checkpoint hashes locked before dev. Primary constrained, legacy/final secondary diagnostics only. Fallback epoch0 is explicitly infeasible.

Same source trainer includes every-epoch optimizer/RNG resume, previously tested exact interruption recovery. Controls must reproduce prior predictions. Safety checks include identity, finite GPU backward, bound scaling, permutation equivariance, source/control hashes, unchanged fit scalers/config. Metrics, three-seed paired crossed bootstrap and 5%change/1%protected-group/coverage/kinematic gates retained. Even a positive result is exploratory repeated-dev evidence.

Run start_experiment.ps1 once. Expected 5-10 minutes; reserve15 minutes silent wait before automatic review. Keep computer awake and app running.
