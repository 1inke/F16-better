# EXP-187D Social Feasibility Audit

Zero-training evaluation of the final Joint-V2 checkpoints.  The experiment
compares E0, E1, E2 and F16 on all five ETH/UCY folds with the project's fixed
distance thresholds:

- collision: distance below 0.20 m;
- near collision: distance below 0.50 m.

The native score selects the focal agent's Top-1 candidate. Because the locked
training protocol supervises only `prediction[:, :1]`, social feasibility is
measured between this focal prediction and the ground-truth future paths of
valid neighbours. The ground-truth focal-neighbour statistics are the reference.
Unsupervised non-focal model outputs are deliberately excluded.

No model is trained and no threshold is tuned.

Metric protocol: `focal_prediction_vs_gt_neighbors_v2`.

Formal run:

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\SOCIAL_FEASIBILITY_AUDIT_SERIES\run_exp187d_social_feasibility.py" --device cuda:0 --folds all --models all --seed 1 --batch-size 64
```

The run is resumable. Use `--force` only to recompute completed model/fold
profiles.
