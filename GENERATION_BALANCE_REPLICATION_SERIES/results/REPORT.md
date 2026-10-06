# Balance-only replication

REPLICATION_GATE_FAIL_REVIEW_REQUIRED

ETH frozen generator seed1; correction seeds1/2/3; seed1 discovery reused, seeds2/3 replication; repeatedly inspected dev, no test; not trajectory ensemble

| Seed | Arm | Checkpoint | Group | minADE | minFDE |
|---|---|---|---|---:|---:|
| 1 | native_uniform | best | overall | 0.218538 | 0.405486 |
| 1 | native_uniform | best | change | 0.377452 | 0.734561 |
| 1 | native_uniform | final | overall | 0.217375 | 0.403855 |
| 1 | native_uniform | final | change | 0.379417 | 0.740980 |
| 1 | native_balanced | best | overall | 0.219207 | 0.401388 |
| 1 | native_balanced | best | change | 0.366805 | 0.684849 |
| 1 | native_balanced | final | overall | 0.219830 | 0.410567 |
| 1 | native_balanced | final | change | 0.364113 | 0.709926 |
| 2 | native_uniform | best | overall | 0.218213 | 0.404329 |
| 2 | native_uniform | best | change | 0.370691 | 0.718301 |
| 2 | native_uniform | final | overall | 0.218881 | 0.408988 |
| 2 | native_uniform | final | change | 0.373799 | 0.731660 |
| 2 | native_balanced | best | overall | 0.218925 | 0.402064 |
| 2 | native_balanced | best | change | 0.368602 | 0.694831 |
| 2 | native_balanced | final | overall | 0.216300 | 0.403254 |
| 2 | native_balanced | final | change | 0.352030 | 0.676076 |
| 3 | native_uniform | best | overall | 0.219797 | 0.409979 |
| 3 | native_uniform | best | change | 0.376032 | 0.731207 |
| 3 | native_uniform | final | overall | 0.217830 | 0.404821 |
| 3 | native_uniform | final | change | 0.375746 | 0.739903 |
| 3 | native_balanced | best | overall | 0.218931 | 0.401669 |
| 3 | native_balanced | best | change | 0.364981 | 0.682357 |
| 3 | native_balanced | final | overall | 0.217527 | 0.403520 |
| 3 | native_balanced | final | change | 0.359542 | 0.707627 |

{
  "both_new_seeds_change_ade_fde_improve_native": true,
  "new_mean_change_ade_gain_1pct": true,
  "balanced_vs_uniform_change_CI": false,
  "overall_stable_transition_oracle_guard_1pct": false,
  "final_change_ade_fde_no_harm": true
}

Only three correction seeds (two new); uncertainty estimates remain exploratory. No selector improvement or intention understanding claimed.