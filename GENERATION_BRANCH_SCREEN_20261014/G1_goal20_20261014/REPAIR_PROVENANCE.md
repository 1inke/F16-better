# Preflight checkpoint-device repair

Before any production launch or source lock, the first real two-epoch interrupted-resume preflight stopped at strict optimizer-state comparison. `torch.load(map_location=cuda)` had moved AdamW's CPU step counter to CUDA; `torch.equal` rejected mixed devices. This was not a completed exact-resume PASS.

The original model/preflight sources, failure traceback, statistics, split IDs and smoke checkpoint are retained in `failures/preflight1/`. The restore reader now loads the checkpoint on CPU; `model.load_state_dict` and `optimizer.load_state_dict` perform their own registered device transfers while AdamW's non-capturable step counter remains CPU. No model, objective, RNG, split, coefficient or tolerance changed. Full preflight must be rerun and pass before source lock and sole production launch.

All repairs and reviews are by the same assistant, not external peer review. No old model/result/scientific source was changed.

## Prelaunch accounting review

Before any task/worker creation, manual final review found that the saved auditor asserted 1875 steps per seed, whereas the registered 18909/batch256 actually gives ceil=74 and 25×74=1850. Production epoch code and throughput benchmark already used the correct ceil and were not changed. The assertion now derives its count from the train size and checks 1850. The earlier lock, source, auditor, preflight/statistics/smoke files are retained under `failures/prelaunch1_step_count_schema`; the complete preflight and source lock must be regenerated explicitly, not silently replacing old locked digests or continuing on an incompatible source. This is a reporting-schema repair, not a training budget or scientific-rule change.
