# Mode-conditioned generation support pilot

PILOT_GATE_FAIL_REVIEW_REQUIRED

ETH seed1 reused dev diagnostic; fixed K20; frozen backbone/scorer; no held-out test

| Arm | Group | minADE | minFDE | Top3 ADE | Classes | GT family coverage |
|---|---|---:|---:|---:|---:|---:|
| native | overall | 0.219154 | 0.399725 | 0.326190 | 4.640 | 0.984 |
| native | transition | 0.275791 | 0.512714 | 0.403145 | 4.796 | 0.982 |
| native | change | 0.372175 | 0.696326 | 0.522072 | 5.457 | 0.910 |
| mode_oracle_best | overall | 0.218538 | 0.405486 | 0.320498 | 4.557 | 0.981 |
| mode_oracle_best | transition | 0.275648 | 0.520921 | 0.397302 | 4.712 | 0.978 |
| mode_oracle_best | change | 0.377452 | 0.734561 | 0.524489 | 5.399 | 0.892 |
| mode_oracle_final | overall | 0.217375 | 0.403855 | 0.317879 | 4.532 | 0.983 |
| mode_oracle_final | transition | 0.271264 | 0.514531 | 0.394784 | 4.695 | 0.978 |
| mode_oracle_final | change | 0.379417 | 0.740980 | 0.526579 | 5.457 | 0.915 |
| mode_global_best | overall | 0.219739 | 0.406355 | 0.320997 | 4.490 | 0.982 |
| mode_global_best | transition | 0.276826 | 0.516883 | 0.398503 | 4.664 | 0.973 |
| mode_global_best | change | 0.376325 | 0.720614 | 0.518507 | 5.368 | 0.901 |
| mode_global_final | overall | 0.217845 | 0.400226 | 0.310822 | 4.220 | 0.972 |
| mode_global_final | transition | 0.272883 | 0.518890 | 0.388986 | 4.438 | 0.960 |
| mode_global_final | change | 0.381225 | 0.742017 | 0.519473 | 5.372 | 0.897 |
| mode_family_best | overall | 0.219544 | 0.403714 | 0.318163 | 4.493 | 0.984 |
| mode_family_best | transition | 0.278533 | 0.516124 | 0.395653 | 4.642 | 0.978 |
| mode_family_best | change | 0.383153 | 0.737453 | 0.523517 | 5.354 | 0.910 |
| mode_family_final | overall | 0.217549 | 0.401785 | 0.306683 | 4.166 | 0.970 |
| mode_family_final | transition | 0.282283 | 0.537262 | 0.389679 | 4.425 | 0.960 |
| mode_family_final | change | 0.399581 | 0.796334 | 0.534461 | 5.296 | 0.888 |

Heuristic motion families are not learned intentions. Missing base families receive winner loss only. Final checkpoints are diagnostic, not dev-selected replacements. Native logits are unchanged and can be stale after generation changes.

{
  "oracle_gain_1pct": false,
  "support_gain_2pct": true,
  "change_no_harm": false,
  "transition_no_harm": false,
  "family_coverage_guard": true,
  "class_count_guard": true,
  "duplicate_guard": true,
  "acceleration_guard": false
}