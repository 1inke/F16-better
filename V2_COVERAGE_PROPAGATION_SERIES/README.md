# V2 Coverage Propagation Series

## Purpose

This experiment asks whether the H3 oracle-coverage advantage survives the final V2 correction stage.

Only one new model is trained:

| Condition | H3 diversity | H3 coverage | V2 diversity | V2 coverage | Status |
|---|---:|---:|---:|---:|---|
| Coverage-On reference | 0.05 | 0.20 | 0.05 | 0.20 | Reuse EXP-173 C00 |
| Coverage-Off source | 0.05 | 0.00 | 0.05 | 0.20 | One new V2 training |

The V2 loss is identical in both conditions. The only treatment is the H3 source checkpoint's oracle-coverage weight.

## Safety and scope

- ETH, generator seed 1.
- Fixed 5% scene-tail validation split with 1,516 exact sample IDs.
- The new command does not receive test-data paths.
- The analyzer reads only the training file and the frozen validation IDs.
- The existing historical C00 artifact contains an old test evaluation, but this workflow does not read or report it.
- Best checkpoint is primary; epoch-20 final checkpoint is the sensitivity view.
- Paired bootstrap measures validation-sample uncertainty conditional on the two trained models, not training-seed uncertainty.

## One-command run

```powershell
& "C:\Users\linke\Documents\New project\V2_COVERAGE_PROPAGATION_SERIES\run_experiment.ps1"
```

The new V2 training normally takes about 7–12 minutes on the current RTX 5070 Laptop GPU. Analysis generally takes under one minute.

## Preflight and dry run

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\V2_COVERAGE_PROPAGATION_SERIES\run_v2_coverage_propagation.py" --stage preflight
```

```powershell
& "C:\Users\linke\Documents\New project\V2_COVERAGE_PROPAGATION_SERIES\run_experiment.ps1" -DryRun
```

## Resume after interruption

The runner skips a completed condition when `metrics.json` exists. If training stopped before that file was written, rerunning starts that single condition again.

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\V2_COVERAGE_PROPAGATION_SERIES\run_v2_coverage_propagation.py" --stage v2_off
```

Do not use `--force` unless the existing treatment result is intentionally being replaced.

## Run analysis separately

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\V2_COVERAGE_PROPAGATION_SERIES\analyze_v2_coverage_propagation.py" --device cuda:0 --batch-size 128 --bootstrap-repetitions 10000 --bootstrap-seed 20260922
```

## Outputs

- `results/v2_from_h3_coverage_off/.../metrics.json`: new training metrics.
- `results/analysis/condition_summary.csv`: H3/V2 best/final summaries.
- `results/analysis/paired_bootstrap.csv`: paired effects and confidence intervals.
- `results/analysis/effect_summary.json`: gate inputs.
- `results/analysis/verdict.json`: machine-readable decision.
- `results/analysis/V2_COVERAGE_PROPAGATION_REPORT.md`: final report.

## Advancement gate

The H3 coverage effect advances only when best and epoch-20 final views have the same favorable direction and jointly satisfy:

- V2 `J=minADE+0.5*minFDE` improves by at least 2%, or one oracle metric improves by at least 2% while the other degrades by no more than 0.5%;
- Support@1m increases by at least 0.5 candidate/sample;
- Selected ADE does not degrade by more than 0.5%.

If both checkpoint views have less than 1% absolute J change and less than 0.5 Support@1m change, the H3 gain is treated as absorbed by V2 and generator-loss tuning stops.
