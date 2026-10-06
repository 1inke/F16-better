# Preflight 1 implementation mismatch

The first full preflight stopped in `prepare_statistics.py` before model initialization or GPU work. Its assertion expected 12,544 common-statistics IDs, copied from a prior study. This experiment's preregistered metadata scale and freshly reconstructed identity split both specify 12,183. The source trace is preserved beside this note.

Only the stale row-count assertion was changed to 12,183. The split, sample IDs, statistics formulas, scale rules, model, objective, thresholds, and training protocol were not changed. The corrected source must pass the complete source, GPU, resume, resource, and pipeline preflight before any worker can launch.
