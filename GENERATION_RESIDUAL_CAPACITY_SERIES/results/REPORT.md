# Residual capacity experiment

CAPACITY_GATE_FAIL

ETH repeatedly used dev; K20; frozen backbone/scorer; reused verified small-bound controls; three paired correction seeds, no independent data or test

| Bound | Policy | Group | minADE | minFDE |
|---|---|---|---:|---:|
| small | constrained | overall | 0.217592 | 0.399445 |
| small | constrained | change | 0.367054 | 0.692020 |
| small | legacy | overall | 0.216302 | 0.398476 |
| small | legacy | change | 0.365691 | 0.695207 |
| small | final | overall | 0.216724 | 0.399966 |
| small | final | change | 0.371634 | 0.709482 |
| large | constrained | overall | 0.217429 | 0.399570 |
| large | constrained | change | 0.366540 | 0.692019 |
| large | legacy | overall | 0.216541 | 0.399641 |
| large | legacy | change | 0.367847 | 0.699003 |
| large | final | overall | 0.216680 | 0.402176 |
| large | final | change | 0.375132 | 0.722238 |

{
  "change_ade_fde_gain_5pct": false,
  "protected_groups_guard": true,
  "each_seed_change_improves": true,
  "native_and_small_ci": false,
  "all_seed_feasible": true,
  "coverage_guard": true,
  "duplicate_guard": true,
  "kinematic_guard": true
}