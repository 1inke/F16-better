# EXP-197 F16 Real Sparse Execution

This is a zero-training execution audit of the locked formal F16 checkpoint.
It is numbered EXP-197 because EXP-196 is already used by the AV2 scaling
series.  The adapter computes the original eight relation channels and the
original relation encoder only on selected graph edges, then applies edge-list
temporal-social attention with scatter softmax/aggregation.

The gate order is strict: smoke -> numerical equivalence -> graph density ->
real benchmark -> synthetic scaling.  No checkpoint is written and no model
parameter is added or changed.

Quick correctness check:

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\SPARSE_EXECUTION_SERIES\run_exp197_sparse_execution.py" --stage smoke --device cuda:0
```

Formal zero-training run:

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\SPARSE_EXECUTION_SERIES\run_exp197_sparse_execution.py" --stage all --device cuda:0 --batches 20 --batch-size 16 --density-limit 2000 --warmup 50 --iterations 200 --cycles 10
```

Use `--density-limit 0` for a complete ETH/UCY density census.  Results are
written to `EXP_197_RESULTS`.  The all-stage runner stops immediately if the
numerical equivalence gate fails.

The included density stage deliberately covers the five native ETH/UCY arrays.
SDD and AV2 must be evaluated with their own configuration, coordinate scale,
maximum-agent setting and matched checkpoint; applying the ETH graph radius to
those arrays would produce a misleading density result.
