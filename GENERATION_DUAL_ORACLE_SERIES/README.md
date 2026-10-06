# Joint/separate oracle supervision ablation

Hypothesis: joint ADE+0.5FDE assignment may leave the best endpoint candidate unsupervised. This is not proof that the hypothesis explains previous failures.

One factor: replace weighted joint-winner mean by a fixed 50/50 blend of joint-winner and separate ADE/FDE-winner means. Keep K20, original candidates, 36,184-parameter shared correction, velocity bound1.5, losses/regularizers/protection, seeds1/2/3, 35epochs, original fit25912/cal2879, and constrained calibration checkpoint selection unchanged. No top3/family auxiliary, no proposals/pruning, no backbone/scorer training, no test access.

Controls: verified original residual-capacity large-bound checkpoints, reused not independent replication. 18 models locked before repeated ETH dev1516. Primary constrained; legacy/final diagnostic only. Alpha0 recovers original loss. GT only training labels/offline evaluation, never model input. Separate minima may use different paths: report same-path joint quality, ADE at FDE winner, FDE at ADE winner, and disagreement. Added same-path joint guard <=1% harm versus learned control in each group. Original major gates remain changeADE/FDE>=5% gain, protected groups<=1% harm, each seed improves, paired CI against native/control, coverage/duplicate/kinematic guards. Any apparent signal requires correlation/three-seed uncertainty review; repeated dev is not independent generalization.

Preflight locks scientific sources/data/control hashes; real GPU gradient/identity/permutation tests and fit/cal assignment diagnostics. Resume smoke verifies interrupted vs continuous optimizer/RNG/parameter/calibration agreement. Production saves each epoch and resume; completed seeds skipped only after hash verification. Post-evaluation audits 18 hashes, explicit IDs, finite K20 predictions and recomputed metrics.

Run in PowerShell: `& "C:\Users\linke\Documents\New project\GENERATION_DUAL_ORACLE_SERIES\start_experiment.ps1"`

If interrupted, run the same Python entry with `--action run --device cuda:0` after checking no existing process. Preserve failure/log evidence. Do not rerun launcher over existing logs or overwrite old experiments. Expected 40–60min; reserve90min silent wait after startup confirmation.
