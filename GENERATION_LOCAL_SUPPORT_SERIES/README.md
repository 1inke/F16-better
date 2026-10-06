# Fixed-20 candidate generation with local support

First generation-only pilot for the agreed objective: a plausible trajectory set should provide several good candidates near the realized future while retaining alternatives. This pilot tests a trainable local-support objective; it does not yet establish learned intention inference or adaptive allocation among intention clusters.

All arms have exactly 20 trajectories:

- `native`: existing frozen fold-local ETH F16 V2 trajectories.
- `oracle_only`: learned trajectory residual with best-of-20 quality loss.
- `local_support`: identical residual network and losses, plus the mean quality of the three closest candidates.

The residual network reads observation-only motion/social features and each existing candidate trajectory. It predicts bounded per-step velocity corrections, integrated to produce corrected trajectories. The frozen backbone is reused from its feature cache. This is a trained generator-side correction module, not appended hand-designed trajectories. Mode logits remain fixed, and selected ADE is diagnostic only because changing candidate geometry can make old logits stale.

Both learned arms use the same seed, batch order, architecture, epoch budget, fit/calibration split, residual/smoothness losses, and modest anti-collapse constraint. The anti-collapse term maintains only a bounded fraction of existing pairwise separation; it does not reward arbitrarily large spread. All 20 trajectories remain available at inference, which has no future/GT input.

Scope: ETH seed1 pilot, fit/internal calibration from the previously used train cache, diagnostic development only after all models lock. The generator has seen fit/calibration examples, and development has been reviewed previously. The pilot is not independent held-out-test evidence. No test or automatic seed/fold expansion.

Main diagnostics: independent minADE20/minFDE20; mean ADE of nearest three; counts within .3/.5 m; at-least-three hit rate; pairwise duplicate fraction; candidate class coverage; endpoint spread; trajectory speed/acceleration; stable/transition/change groups. Compare best and final checkpoints. Advancement requires oracle and support gains, diversity/kinematic guards, and final-checkpoint consistency, followed by human review.

Training labels may use GT. Inference, proposal correction, normalization fit, and candidate generation never use it. Every metric preserves sample_id. Source/cache/raw-data hashes are checked before training; completed model hashes are locked before development evaluation.

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\GENERATION_LOCAL_SUPPORT_SERIES\run_generation_support.py" --action preflight --device "cuda:0"
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\GENERATION_LOCAL_SUPPORT_SERIES\run_generation_support.py" --action run --device "cuda:0"
```

Logs: stdout.log / stderr.log. Results: summary.json, metrics.csv, paired_bootstrap.csv, REPORT.md, per-sample NPZ, best/final model checkpoints. Failures are recorded in results/failure.txt.
