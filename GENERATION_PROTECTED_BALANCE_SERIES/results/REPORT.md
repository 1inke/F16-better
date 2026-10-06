# Protected balance generation experiment

PROTECTED_BALANCE_GATE_FAIL

ETH reused dev; correction seeds1/2/3; K20; no selector training or test; infeasible trained arms fall back to epoch0 native

| Arm | Policy | Group | minADE | minFDE |
|---|---|---|---:|---:|
| uniform | constrained | overall | 0.219596 | 0.403698 |
| uniform | constrained | change | 0.371948 | 0.701125 |
| uniform | legacy | overall | 0.218849 | 0.406598 |
| uniform | legacy | change | 0.374725 | 0.728023 |
| uniform | final | overall | 0.218029 | 0.405888 |
| uniform | final | change | 0.376321 | 0.737514 |
| balanced | constrained | overall | 0.219121 | 0.401078 |
| balanced | constrained | change | 0.367625 | 0.686214 |
| balanced | legacy | overall | 0.219021 | 0.401707 |
| balanced | legacy | change | 0.366796 | 0.687346 |
| balanced | final | overall | 0.217886 | 0.405780 |
| balanced | final | change | 0.358561 | 0.697876 |
| uniform_protected | constrained | overall | 0.217704 | 0.400216 |
| uniform_protected | constrained | change | 0.369927 | 0.697846 |
| uniform_protected | legacy | overall | 0.217112 | 0.401399 |
| uniform_protected | legacy | change | 0.373415 | 0.712162 |
| uniform_protected | final | overall | 0.216718 | 0.401698 |
| uniform_protected | final | change | 0.372699 | 0.719169 |
| balanced_protected | constrained | overall | 0.217592 | 0.399445 |
| balanced_protected | constrained | change | 0.367054 | 0.692020 |
| balanced_protected | legacy | overall | 0.216302 | 0.398476 |
| balanced_protected | legacy | change | 0.365691 | 0.695207 |
| balanced_protected | final | overall | 0.216724 | 0.399966 |
| balanced_protected | final | change | 0.371634 | 0.709482 |

{
  "change_ade_fde_gain_5pct": false,
  "protected_groups_guard": true,
  "each_seed_change_improves": true,
  "native_and_balanced_ci": false,
  "all_seed_feasible": true,
  "coverage_guard": true,
  "duplicate_guard": true,
  "kinematic_guard": true
}