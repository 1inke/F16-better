# F16 response-gap diagnosis — 2026-10-04

Human authorized the next sequence on 2026-10-04: response-gap diagnosis,
candidate-specific response architecture, then separately test supervision and scoring.
This folder implements the FIRST diagnostic only. No generator/probe training,
optimizer, model forward, unfreezing, new data, or formal-trial counter increment.
All new artifacts stay here; old sources, checkpoints and results remain immutable.

Primary: original FIT 25,912 and CAL 2,879 native K20 saved candidates.
Secondary: the already-explored DEV 1,516 native candidates and all 18 saved
time-response policy outputs. Secondary DEV is descriptive, NOT used to select
thresholds, checkpoints, architecture hyperparameters, or claim generalization.

`DESIGN.md` and `protocol.json` fix operational criteria before this new analysis.
`diagnosis.py --preflight` checks synthetic cases, input alignment and old SHA locks.
`start_independent.ps1` launches one on-demand Windows task with a byte lock.
`check_once.ps1` is for the single half-time check and the completion deadline.
Saved job SHA records allow a verified completed job to be skipped after a failure;
this is deterministic analysis resumption, not a production optimizer/RNG audit.

Outputs: per-candidate `jobs/*.npz`, aggregate `RESULT.json`, `RESULT_ANALYSIS.md`,
`independent_audit.json`, progress/worker/exit state and preserved failure provenance.
CPU only: GPU gradients, paired training and exact optimizer resume are not applicable.

Planning ETA will use actual preflight timing. Half-time and completion follow-up
are saved in runtime_state.json and the existing heartbeat. After launch return
to the user; do not poll until the registered check. No automatic next training
before interpreting this diagnosis and specifying a locked single-factor contrast.

No diagnostic classification proves causal decoder failure, intent, unknowability,
or that one realized GT invalidates every other plausible future. Different event
criteria may yield different operational gaps. Three scene families and correlated
windows do not support reliable scene-cluster inference; report descriptive strata.

Research design and code assisted by Codex using Academic Research Suite principles;
no external independent peer-review or public preregistration claim.
