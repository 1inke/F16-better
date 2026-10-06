# EXP-198 F16 Fixed-K Sparse Execution

This is the final zero-training runtime experiment after EXP-197.  It compares
the formal masked-dense F16, dynamic edge-list execution, and GPU-regular
Fixed-K execution.  EXP-198 is used because EXP-197 already denotes the prior
edge-list experiment.

The formal graph is `adaptive_kinematic`: `TOP_K=4` is not its operative bound;
it uses `K_MAX=6`. Consequently, the exact regular representation has width
seven (`social_bound=6`, plus self). The synthetic protocol explicitly uses
four social neighbours plus self, for width five. Invalid slots are masked,
and the main relation/attention path uses gather plus
contiguous softmax/reduction without dynamic edge grouping or scatter.

Quick gate:

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\FIXED_K_SPARSE_EXECUTION_SERIES\run_exp198_fixed_k.py" --stage equivalence --device cuda:0 --batches 20 --batch-size 16
```

Formal run:

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\FIXED_K_SPARSE_EXECUTION_SERIES\run_exp198_fixed_k.py" --stage all --device cuda:0 --batches 20 --batch-size 16 --warmup 100 --iterations 500 --cycles 15
```

Add `--include-256` only if an N=256 stress point is desired.  The all-stage
runner stops before benchmarking if graph or numerical equivalence fails.
