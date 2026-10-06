# Candidate set mean/std context

SET_CONTEXT_GATE_FAIL

ETH repeatedly inspected dev; fixed nativeK20; six fresh matched paired-seed trainings; mean/std context vs zero inputs, same42328parameter architecture/initialization; no new loss/GT inference/test/backbone/scorer; zero-context48 weights receive no data gradient, count matching not equal effective capacity

| Arm | Policy | Group | minADE | minFDE | Joint |
|---|---|---|---:|---:|---:|
| zero | constrained | overall | 0.219785 | 0.402926 | 0.434472 |
| zero | constrained | change | 0.376833 | 0.721488 | 0.763083 |
| zero | legacy | overall | 0.219806 | 0.402253 | 0.434144 |
| zero | legacy | change | 0.376640 | 0.717603 | 0.761280 |
| zero | final | overall | 0.225114 | 0.408240 | 0.443109 |
| zero | final | change | 0.393550 | 0.754081 | 0.797162 |
| mean_std | constrained | overall | 0.219022 | 0.400997 | 0.433154 |
| mean_std | constrained | change | 0.372979 | 0.707470 | 0.753092 |
| mean_std | legacy | overall | 0.219523 | 0.401961 | 0.433977 |
| mean_std | legacy | change | 0.374951 | 0.711509 | 0.756645 |
| mean_std | final | overall | 0.225793 | 0.408933 | 0.443477 |
| mean_std | final | change | 0.394045 | 0.742365 | 0.791068 |

{
  "change_ade_fde_gain_5pct": false,
  "protected_groups_guard": false,
  "each_seed_change_improves": false,
  "native_and_zero_ci": false,
  "all_seed_feasible": true,
  "coverage_guard": true,
  "duplicate_guard": true,
  "kinematic_guard": true,
  "same_path_joint_guard": true
}