# Candidate set mean/std context

SET_CONTEXT_GATE_FAIL

ETH repeatedly inspected dev; fixed nativeK20; six fresh matched paired-seed trainings; mean/std context vs zero inputs, same42328parameter architecture/initialization; no new loss/GT inference/test/backbone/scorer; zero-context48 weights receive no data gradient, count matching not equal effective capacity

| Arm | Policy | Group | minADE | minFDE | Joint |
|---|---|---|---:|---:|---:|
| zero | constrained | overall | 0.217287 | 0.399621 | 0.430337 |
| zero | constrained | change | 0.366895 | 0.692607 | 0.738193 |
| zero | legacy | overall | 0.216724 | 0.400318 | 0.429998 |
| zero | legacy | change | 0.368670 | 0.707919 | 0.746854 |
| zero | final | overall | 0.216481 | 0.400894 | 0.429759 |
| zero | final | change | 0.374134 | 0.721209 | 0.760334 |
| mean_std | constrained | overall | 0.217411 | 0.399647 | 0.430539 |
| mean_std | constrained | change | 0.367473 | 0.693328 | 0.739439 |
| mean_std | legacy | overall | 0.216572 | 0.401443 | 0.430408 |
| mean_std | legacy | change | 0.368657 | 0.710070 | 0.748795 |
| mean_std | final | overall | 0.216600 | 0.402377 | 0.431242 |
| mean_std | final | change | 0.373490 | 0.717344 | 0.759419 |

{
  "change_ade_fde_gain_5pct": false,
  "protected_groups_guard": true,
  "each_seed_change_improves": true,
  "native_and_zero_ci": false,
  "all_seed_feasible": true,
  "coverage_guard": true,
  "duplicate_guard": true,
  "kinematic_guard": true,
  "same_path_joint_guard": true
}