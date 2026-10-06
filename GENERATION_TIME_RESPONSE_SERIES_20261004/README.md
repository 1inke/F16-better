# Observed-cue temporal response experiment

Human-authorized 2026-10-04, new folder only. All old results, scientific sources and models are immutable. This is one new paired experiment, not a completion claim for the historical unaudited cycle11 record.

`zero`: common time-query head, followed by four-step block mean and repeat. `mean_std`: identical head, retains all twelve increments. Both use frozen original context/mode features and eight observed tokens. No future observations or GT enter forward. K20/native base, existing loss, regularizers, protection, calibration and physical clip remain fixed. Neither arm trains F16, original decoder or scorer.

Read DESIGN.md/protocol.json first. Commands: `run_native.py --action preflight --device cuda:0`, `test_resume.py`, `benchmark.py`; only then `start_independent.ps1`. Six fresh 80-epoch trainings, seeds1/2/3, eighteen saved checkpoint policies. No old treatment/teacher warm-start.

The independent worker is launched once through an on-demand Windows task without triggers, protected by a byte lock. One halfway-time check updates the completion estimate; completion requires all output/source/old-result/frozen audits. Then discuss the result with the human; do not automatically launch another trial.
