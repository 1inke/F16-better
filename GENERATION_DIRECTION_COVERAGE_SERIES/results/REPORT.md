# Fixed-budget direction coverage pilot

PILOT_GATE_FAIL_REVIEW_REQUIRED

ETH seed1 reused dev; K20; frozen backbone and no scorer training; no test

| Arm | Group | minADE | minFDE | Top3 ADE |
|---|---|---:|---:|---:|
| native | overall | 0.219154 | 0.399725 | 0.326190 |
| native | stable | 0.175177 | 0.313804 | 0.268952 |
| native | transition | 0.275791 | 0.512714 | 0.403145 |
| native | change | 0.372175 | 0.696326 | 0.522072 |
| replacement_untrained | overall | 0.231615 | 0.416771 | 0.320907 |
| replacement_untrained | stable | 0.168162 | 0.292520 | 0.247637 |
| replacement_untrained | transition | 0.302453 | 0.551775 | 0.413127 |
| replacement_untrained | change | 0.463428 | 0.874458 | 0.578030 |
| native_uniform_best | overall | 0.218538 | 0.405486 | 0.320498 |
| native_uniform_best | stable | 0.173229 | 0.312261 | 0.261597 |
| native_uniform_best | transition | 0.275648 | 0.520921 | 0.397302 |
| native_uniform_best | change | 0.377452 | 0.734561 | 0.524489 |
| native_uniform_final | overall | 0.217375 | 0.403855 | 0.317879 |
| native_uniform_final | stable | 0.172094 | 0.309955 | 0.257972 |
| native_uniform_final | transition | 0.271264 | 0.514531 | 0.394784 |
| native_uniform_final | change | 0.379417 | 0.740980 | 0.526579 |
| native_balanced_best | overall | 0.219207 | 0.401388 | 0.326710 |
| native_balanced_best | stable | 0.175824 | 0.317323 | 0.270673 |
| native_balanced_best | transition | 0.278391 | 0.518579 | 0.403570 |
| native_balanced_best | change | 0.366805 | 0.684849 | 0.516938 |
| native_balanced_final | overall | 0.219830 | 0.410567 | 0.323962 |
| native_balanced_final | stable | 0.177610 | 0.324045 | 0.268130 |
| native_balanced_final | transition | 0.276798 | 0.523674 | 0.400000 |
| native_balanced_final | change | 0.364113 | 0.709926 | 0.514041 |
| replacement_uniform_best | overall | 0.225915 | 0.411041 | 0.312287 |
| replacement_uniform_best | stable | 0.170280 | 0.299234 | 0.245190 |
| replacement_uniform_best | transition | 0.283126 | 0.514943 | 0.394047 |
| replacement_uniform_best | change | 0.434132 | 0.840709 | 0.550468 |
| replacement_uniform_final | overall | 0.225915 | 0.411041 | 0.312287 |
| replacement_uniform_final | stable | 0.170280 | 0.299234 | 0.245190 |
| replacement_uniform_final | transition | 0.283126 | 0.514943 | 0.394047 |
| replacement_uniform_final | change | 0.434132 | 0.840709 | 0.550468 |
| replacement_balanced_best | overall | 0.227666 | 0.417404 | 0.318916 |
| replacement_balanced_best | stable | 0.177923 | 0.316315 | 0.258010 |
| replacement_balanced_best | transition | 0.275629 | 0.510562 | 0.395739 |
| replacement_balanced_best | change | 0.417068 | 0.806675 | 0.532483 |
| replacement_balanced_final | overall | 0.227666 | 0.417404 | 0.318916 |
| replacement_balanced_final | stable | 0.177923 | 0.316315 | 0.258010 |
| replacement_balanced_final | transition | 0.275629 | 0.510562 | 0.395739 |
| replacement_balanced_final | change | 0.417068 | 0.806675 | 0.532483 |

{
  "overall_oracle_no_harm": false,
  "change_ade_gain_2pct": false,
  "change_fde_no_harm": false,
  "stable_ade_guard_1pct": false,
  "transition_ade_guard_1pct": true,
  "duplicate_guard": true,
  "kinematic_guard": true,
  "change_final_consistency": false
}

Best selected on common internal calibration. Final is diagnostic. Explicit motion anchors are heuristics, not understood intentions. Replacement slots have no trained scores; selection improvement is not evaluated.