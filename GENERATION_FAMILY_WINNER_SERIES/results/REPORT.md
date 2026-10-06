# Single family winner experiment

FAMILY_WINNER_GATE_FAIL

ETH reused dev; K20; three paired correction seeds; verified reused controls; family GT labels loss-only; no test/scorer/backbone training

| Arm | Policy | Group | minADE | minFDE |
|---|---|---|---:|---:|
| control | constrained | overall | 0.217429 | 0.399570 |
| control | constrained | change | 0.366540 | 0.692019 |
| control | legacy | overall | 0.216541 | 0.399641 |
| control | legacy | change | 0.367847 | 0.699003 |
| control | final | overall | 0.216680 | 0.402176 |
| control | final | change | 0.375132 | 0.722238 |
| family | constrained | overall | 0.217496 | 0.399912 |
| family | constrained | change | 0.367746 | 0.695281 |
| family | legacy | overall | 0.216967 | 0.401365 |
| family | legacy | change | 0.368794 | 0.708112 |
| family | final | overall | 0.217452 | 0.403277 |
| family | final | change | 0.378625 | 0.731513 |

{
  "change_ade_fde_gain_5pct": false,
  "protected_groups_guard": true,
  "each_seed_change_improves": false,
  "native_and_control_ci": false,
  "all_seed_feasible": true,
  "coverage_guard": true,
  "duplicate_guard": true,
  "kinematic_guard": true
}