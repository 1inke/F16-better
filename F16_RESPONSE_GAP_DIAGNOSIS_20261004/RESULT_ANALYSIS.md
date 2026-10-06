# Response-gap diagnosis: saved-output completion

Operational diagnosis only. Read DESIGN.md and independent_audit.json before interpreting.
AI-assisted design/code; no external independent peer review/public preregistration.

## Primary original FIT/CAL

### FIT_native — clear events N=10237

{
  "group": "clear_event",
  "n": 10237,
  "independent_min_ADE": 0.25548768043518066,
  "independent_min_FDE": 0.43411409854888916,
  "selected_ADE": 0.6961372494697571,
  "selected_FDE": 1.5584447383880615,
  "joint_oracle_ADE": 0.26656225323677063,
  "joint_oracle_FDE": 0.4493652284145355,
  "joint_oracle_early": 0.12943656742572784,
  "joint_oracle_middle": 0.2898437976837158,
  "joint_oracle_late": 0.3804064393043518,
  "joint_oracle_speed_curve_mae": 0.11706580966711044,
  "joint_oracle_angle_curve_mae": 17.683263778686523,
  "clear_event_n": 10237,
  "threshold_0": {
    "ADE": 0.2,
    "FDE": 0.4,
    "samepath_adequate_n": 3152,
    "selected_adequate_n": 406,
    "partition_counts": {
      "adequate_selected": 406,
      "adequate_present_selection_miss": 2746,
      "no_candidate_event": 0,
      "dominant_type_gap": 41,
      "onset_gap": 850,
      "direction_gap": 297,
      "strength_gap": 622,
      "coarse_response_covered_path_inadequate": 5275,
      "nonclear_path_inadequate": 0,
      "boundary_uncertain_path_inadequate": 0
    },
    "alternate_order_partition_counts": {
      "adequate_selected": 406,
      "adequate_present_selection_miss": 2746,
      "no_candidate_event": 0,
      "dominant_type_gap": 41,
      "onset_gap": 1424,
      "direction_gap": 60,
      "strength_gap": 285,
      "coarse_response_covered_path_inadequate": 5275,
      "nonclear_path_inadequate": 0,
      "boundary_uncertain_path_inadequate": 0
    }
  },
  "threshold_1": {
    "ADE": 0.3,
    "FDE": 0.5,
    "samepath_adequate_n": 5831,
    "selected_adequate_n": 854,
    "partition_counts": {
      "adequate_selected": 854,
      "adequate_present_selection_miss": 4977,
      "no_candidate_event": 0,
      "dominant_type_gap": 16,
      "onset_gap": 502,
      "direction_gap": 189,
      "strength_gap": 441,
      "coarse_response_covered_path_inadequate": 3258,
      "nonclear_path_inadequate": 0,
      "boundary_uncertain_path_inadequate": 0
    },
    "alternate_order_partition_counts": {
      "adequate_selected": 854,
      "adequate_present_selection_miss": 4977,
      "no_candidate_event": 0,
      "dominant_type_gap": 16,
      "onset_gap": 882,
      "direction_gap": 45,
      "strength_gap": 205,
      "coarse_response_covered_path_inadequate": 3258,
      "nonclear_path_inadequate": 0,
      "boundary_uncertain_path_inadequate": 0
    }
  },
  "event_accounting_denominator": 10237,
  "nested_coverage_counts": {
    "event": 10237,
    "type": 10165,
    "timing": 8900,
    "direction": 8479,
    "strength": 7697
  },
  "marginal_same_candidate_coverage_counts": {
    "any_event": 10237,
    "dominant_type": 10165,
    "any_multitype_overlap": 10236,
    "type_direction": 10083,
    "type_strength": 9884,
    "type_timing": 8900,
    "type_direction_strength": 9730
  },
  "best_onset_error_given_any_same_type": {
    "n": 10165,
    "mean_steps": 1.1064436792916872,
    "missing_same_type_n": 72
  }
}

### CAL_native — clear events N=1360

{
  "group": "clear_event",
  "n": 1360,
  "independent_min_ADE": 0.26279935240745544,
  "independent_min_FDE": 0.43356430530548096,
  "selected_ADE": 0.7141484618186951,
  "selected_FDE": 1.584069848060608,
  "joint_oracle_ADE": 0.2739778459072113,
  "joint_oracle_FDE": 0.4521631598472595,
  "joint_oracle_early": 0.13735602796077728,
  "joint_oracle_middle": 0.3006404638290405,
  "joint_oracle_late": 0.3839370608329773,
  "joint_oracle_speed_curve_mae": 0.12095451354980469,
  "joint_oracle_angle_curve_mae": 16.626670837402344,
  "clear_event_n": 1360,
  "threshold_0": {
    "ADE": 0.2,
    "FDE": 0.4,
    "samepath_adequate_n": 353,
    "selected_adequate_n": 30,
    "partition_counts": {
      "adequate_selected": 30,
      "adequate_present_selection_miss": 323,
      "no_candidate_event": 0,
      "dominant_type_gap": 0,
      "onset_gap": 110,
      "direction_gap": 51,
      "strength_gap": 115,
      "coarse_response_covered_path_inadequate": 731,
      "nonclear_path_inadequate": 0,
      "boundary_uncertain_path_inadequate": 0
    },
    "alternate_order_partition_counts": {
      "adequate_selected": 30,
      "adequate_present_selection_miss": 323,
      "no_candidate_event": 0,
      "dominant_type_gap": 0,
      "onset_gap": 216,
      "direction_gap": 9,
      "strength_gap": 51,
      "coarse_response_covered_path_inadequate": 731,
      "nonclear_path_inadequate": 0,
      "boundary_uncertain_path_inadequate": 0
    }
  },
  "threshold_1": {
    "ADE": 0.3,
    "FDE": 0.5,
    "samepath_adequate_n": 745,
    "selected_adequate_n": 79,
    "partition_counts": {
      "adequate_selected": 79,
      "adequate_present_selection_miss": 666,
      "no_candidate_event": 0,
      "dominant_type_gap": 0,
      "onset_gap": 56,
      "direction_gap": 36,
      "strength_gap": 86,
      "coarse_response_covered_path_inadequate": 437,
      "nonclear_path_inadequate": 0,
      "boundary_uncertain_path_inadequate": 0
    },
    "alternate_order_partition_counts": {
      "adequate_selected": 79,
      "adequate_present_selection_miss": 666,
      "no_candidate_event": 0,
      "dominant_type_gap": 0,
      "onset_gap": 134,
      "direction_gap": 8,
      "strength_gap": 36,
      "coarse_response_covered_path_inadequate": 437,
      "nonclear_path_inadequate": 0,
      "boundary_uncertain_path_inadequate": 0
    }
  },
  "event_accounting_denominator": 1360,
  "nested_coverage_counts": {
    "event": 1360,
    "type": 1359,
    "timing": 1212,
    "direction": 1137,
    "strength": 1003
  },
  "marginal_same_candidate_coverage_counts": {
    "any_event": 1360,
    "dominant_type": 1359,
    "any_multitype_overlap": 1360,
    "type_direction": 1347,
    "type_strength": 1310,
    "type_timing": 1212,
    "type_direction_strength": 1292
  },
  "best_onset_error_given_any_same_type": {
    "n": 1359,
    "mean_steps": 1.0625459896983076,
    "missing_same_type_n": 1
  }
}

## Interpretation pending

Completion follow-up must compare primary FIT/CAL strata and all fixed secondary outputs,
consider marginal and alternate-order gaps, then specify one architecture hypothesis.
Do not infer that hierarchical counts identify causes, or launch further training automatically.