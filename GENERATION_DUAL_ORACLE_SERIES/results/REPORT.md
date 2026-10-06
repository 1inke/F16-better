# Joint/separate oracle supervision

DUAL_ORACLE_GATE_FAIL

Repeated ETH dev; K20 native paths; shared128/bound1.5; loss blend .5 joint+.5 independent ADE/FDE minima; original regularizers/protection; frozen backbone/scorer; reused control not independent replication; GT loss-only; no test. Generic NPZ/CSV family label denotes dual_oracle, not family loss.

| Arm | Policy | Group | minADE | minFDE | same-path Joint |
|---|---|---|---:|---:|---:|
| control | constrained | overall | 0.217429 | 0.399570 | 0.430452 |
| control | constrained | change | 0.366540 | 0.692019 | 0.737693 |
| control | legacy | overall | 0.216541 | 0.399641 | 0.429670 |
| control | legacy | change | 0.367847 | 0.699003 | 0.742439 |
| control | final | overall | 0.216680 | 0.402176 | 0.430593 |
| control | final | change | 0.375132 | 0.722238 | 0.761337 |
| dual_oracle | constrained | overall | 0.217296 | 0.399454 | 0.430434 |
| dual_oracle | constrained | change | 0.366404 | 0.692141 | 0.738310 |
| dual_oracle | legacy | overall | 0.216457 | 0.399370 | 0.429347 |
| dual_oracle | legacy | change | 0.366126 | 0.696916 | 0.738779 |
| dual_oracle | final | overall | 0.216996 | 0.401676 | 0.430765 |
| dual_oracle | final | change | 0.379349 | 0.728821 | 0.768884 |

{
  "change_ade_fde_gain_5pct": false,
  "protected_groups_guard": true,
  "each_seed_change_improves": true,
  "native_and_control_ci": false,
  "all_seed_feasible": true,
  "coverage_guard": true,
  "duplicate_guard": true,
  "kinematic_guard": true,
  "same_path_joint_quality_guard": true
}