# Candidate prior isolation pilot

Follow-up to the directional candidate pilot. The fixed eight maneuver proposals improved oracle coverage but were almost never selected. This experiment isolates whether their low synthetic score prior contributes to that failure.

Four matched arms, ETH seed 1, 30 epochs each:

| Arm | Candidate set | Score |
|---|---|---|
| low_prior_aug28 | original20 + fixed8 | attenuated native + learned residual; proposals median minus .5 std |
| equal_prior_aug28 | identical 28 | attenuated native + learned residual; proposals native median |
| utility_base20 | original20 | learned utility only |
| utility_aug28 | original20 + fixed8 | learned utility only |

The utility arms remove native logits from both additive scores and candidate features, and remove the explicit/native provenance flag. They retain the same observed motion/social information, frozen seven-state posterior, candidate trajectories and maneuver descriptors. All four heads restart from the same random seed; training batches, targets, budget, and calibration split match.

Calibration uses the last 10% of the existing ETH train cache. Development is evaluated after all models are locked. The underlying generator has seen train/calibration samples; the development set has been inspected in earlier research. This is a diagnostic pilot, conditional on one generator and one head seed. No held-out test, fold expansion, or generator retraining is included.

Outputs: models, model_lock.json, per-sample NPZ, metrics.csv, paired_bootstrap.csv, summary.json, REPORT.md. Training logs are stdout.log/stderr.log. A failed process writes results/failure.txt.

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\CANDIDATE_PRIOR_ISOLATION_SERIES\run_prior_isolation.py" --action preflight --device "cuda:0"
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\CANDIDATE_PRIOR_ISOLATION_SERIES\run_prior_isolation.py" --action run --device "cuda:0"
```

Run the second command to resume only if the existing process has stopped. Completed arm checkpoints are reused.
