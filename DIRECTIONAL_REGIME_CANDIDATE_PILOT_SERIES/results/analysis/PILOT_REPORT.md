# Directional regime + explicit candidate pilot result

- Verdict: `PILOT_REVIEW_REQUIRED__NO_DIRECTIONAL_SELECTION_SIGNAL__EXPLICIT_CANDIDATE_COVERAGE_SIGNAL`
- Fold/seed: ETH / 1
- Held-out test access: none
- Intent dev accuracy / macro-F1: 0.6781 / 0.4256

| Arm | Selected ADE | Selected FDE | minADE | ADE regret | Native scale |
|---|---:|---:|---:|---:|---:|
| native_base20 | 0.542432 | 1.197316 | 0.219154 | 0.323278 | 1.0000 |
| scalar_d1_reference_base20 | 0.544690 | 1.195676 | 0.219154 | 0.325536 | 1.0000 |
| directional_base20 | 0.555387 | 1.220090 | 0.219154 | 0.336233 | 1.0000 |
| directional_attenuated_base20 | 0.548930 | 1.208279 | 0.219154 | 0.329775 | 0.6639 |
| directional_attenuated_aug28 | 0.549086 | 1.216418 | 0.206997 | 0.342089 | 0.6639 |

## Gates

- directional_selection_gain: False
- attenuation_beats_plain_directional: True
- attenuated_fde_guard: False
- augmented_oracle_gain: True
- augmented_selection_gain: False

## Explicit-candidate diagnostic

- Explicit proposal is oracle for 312/1516 samples (20.58%).
- The selector chooses an explicit proposal for 1/1516 samples.
- Oracle ADE gain: 0.012157 (5.55%).

## Paired bootstrap deltas (treatment - reference)

- directional_vs_native_ade: +0.012955, 95% CI [+0.008648, +0.017305]
- directional_vs_native_fde: +0.022775, 95% CI [+0.013462, +0.032320]
- attenuated_vs_native_ade: +0.006498, 95% CI [+0.001223, +0.011768]
- attenuated_vs_native_fde: +0.010963, 95% CI [-0.000973, +0.022669]
- augmented_vs_attenuated_ade: +0.000156, 95% CI [-0.005348, +0.005515]
- augmented_vs_attenuated_fde: +0.008140, 95% CI [-0.004415, +0.020579]
- augmented_vs_base_oracle_ade: -0.012157, 95% CI [-0.014498, -0.009929]

This pilot cannot authorize test evaluation or seed/fold expansion without review.
