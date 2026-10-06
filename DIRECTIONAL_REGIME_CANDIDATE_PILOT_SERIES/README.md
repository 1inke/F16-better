# Directional Regime + Explicit Candidate Pilot

This is a bounded ETH validation pilot for the failure mode exposed by the curated visualizations: the selected trajectory remains close to constant-velocity continuation even when a better turning/stopping candidate exists.

The pilot isolates three questions:

1. Does an observation-only seven-state maneuver posterior help candidate-specific ranking compared with the native scorer and the previous scalar D1 gate?
2. Does entropy-aware attenuation of the native score reduce continuation inertia?
3. Do eight deterministic observation-only maneuver proposals improve candidate coverage before any generator-weight training?

Arms:

- `native_base20`
- `scalar_d1_reference_base20` (existing formal seed-1 D1 checkpoint; read-only reference)
- `directional_base20`
- `directional_attenuated_base20`
- `directional_attenuated_aug28`

The augmented arm appends stop, decelerate, accelerate, mild-left/right, sharp-left/right, and reverse proposals. They are generated only from normalized observation history. Future trajectories are used only for train/calibration labels and offline metrics.

Protocol safeguards:

- exact formal ETH train/dev sample IDs and frozen fold-local V2 candidates;
- last 10% of formal train cache is internal calibration;
- formal dev is loaded for alignment but is not used to select classifier/reranker checkpoints;
- no held-out test path and no generator-weight training;
- one seed only, so results are a pilot rather than formal evidence.

Run:

```powershell
& "C:\Users\linke\Documents\New project\DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES\run_experiment.ps1" -Phase All -Device "cuda:0"
```

Resume after interruption with the same command. Completed checkpoints are reused.
