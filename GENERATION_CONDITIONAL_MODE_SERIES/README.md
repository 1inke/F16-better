# Cycle2 experiment3: observation-conditional mode calibration

## Hypothesis and evidence

Fit-only native-candidate diagnosis found late-horizon error dominates change cases (first/middle/last thirds: 0.119/0.262/0.361m). Left/right coarse family coverage is 100%, yet joint-selected ADE is 0.304/0.307m. Interpolation using actual GT anchors at steps4/8/12 has only 0.045/0.046m ADE. These GT anchors are an unavailable inference diagnostic, not a deployable predictor, proven approximation floor, or evidence of intent understanding. This motivates testing conditional anchor placement before adding knots or smoothing.

## Single factor

Both arms train original anchor templates, proposal scale/delta and refiner, plus a zero-initialized Linear64-to64 modulation. Static arm applies tanh modulation to a constant ones vector; conditional arm uses the original observation-derived context. Mode embedding is multiplied by (1+gate), then passed through the frozen original mode MLP. Original context/query/modeMLP/F16/scorer weights remain fixed. Both still have native observation-dependent geometry, K20, and the original residual clipping/integration.

Both have 31,314 trainable parameters and exactly paired initialization, but the static arm has lower effective functional capacity. This is a conditional-calibration intervention, not an equal-effective-capacity architecture proof. No GT, group or logits enter forward. Frozen scorer weights do not imply unchanged ranking when trajectories change; primary claims are generation oracle metrics only.

## Protocol

Six fresh trainings: static/conditional x seeds1/2/3, 80epochs each. Original fit25912/cal2879, balanced joint winner, diversity/residual/smoothness, protection0.1, learning budget and constrained calibration stay fixed. Final checkpoints are diagnostic, never selected using dev. No heldout test, cross-fold, external data or F16 unfreezing.

GPU preflight passed native identity, gradient reachability, frozen tensors, paired initialization, split and source hashes. Both-arm continuous versus interrupted two-epoch optimizer/RNG/calibration resume is exact. Completed production seeds are hash-verified and skipped on recovery.

## Running and audit

Independent Windows on-demand task `Trajectory-ConditionalMode-Cycle2` runs `independent_worker.py`; no recurrence triggers. Use `start_independent.ps1` for recovery, not a second direct Python worker. Computer must remain awake and logged in. Codex can close without owning the worker, but automated analysis needs the app available.

Started 2026-10-01 21:45:58 HK, PID16352, Windows-service parent2728. Estimate15–25min, planning20min; one progress check21:57:59, initial analysis22:05:59. No polling before the progress deadline; revise from actual remaining work at that check.

Require all18 policy model hashes, explicit IDs, finite K20 predictions, recomputed metrics, frozen audits and successful exit. Major gates remain changeADE/FDE each>=5% improvement, protected groups<=1%harm, three-seed consistency, nonzero-crossing paired CIs against native and fresh static control, same-path/coverage/duplicates/hard-motion guards, and all-group speedp95/accelerationmean,p95/jerkmean,p95<=native1%harm (dt0.4s). First major signal needs sample/scene correlation and three-seed uncertainty review. Repeated ETH dev is exploratory, not independent generalization.

This is cycle2 experiment3. On completion, write the consolidated three-experiment review before any fourth experiment or major-direction adjustment.
