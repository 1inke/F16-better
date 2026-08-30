# EXP-151 ETSE-E1

E1a tests a learned, time-aligned relation bias in F16 sparse temporal-social attention. The E0 checkpoint and CV-free 20-mode / three-anchor decoder are loaded exactly, then frozen. Only a zero-initialised `relation(8) -> 16 -> heads` projection is trained.

Run the initial screen:

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\EDGE_TEMPORAL_SPARSE_INTERACTION_SERIES\run_etse_e1_batch.py" --device cuda:0 --seeds 1
```

It runs both General and Challenge, each for 12 epochs. Results are stored under `EXP_151_ETSE_E1/results/`; the compact summary is `EXP_151_ETSE_E1/summary.json`.

Promotion requires at least one split to improve oracle minADE or minFDE by 0.5% or more versus its matched E0 checkpoint, no material regression on the other split, and a subsequent mechanism audit before E1b.

## Decision

**EXP-151 rejected. Do not proceed to E1b.** The learned time-aligned
relation bias was active, but its oracle gains were below the screening
threshold and General same-path FDE regressed. The machine-readable decision
is recorded in `EXP_151_ETSE_E1/summary.json`.
