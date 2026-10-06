# Candidate prior isolation pilot

ETH seed1 development diagnosis; generator-seen fit/calibration; reused frozen intent classifier; no test

| Arm | Selected ADE | Selected FDE | minADE |
|---|---:|---:|---:|
| native | 0.542432 | 1.197316 | 0.219154 |
| low_prior_aug28 | 0.546406 | 1.209731 | 0.206997 |
| equal_prior_aug28 | 0.547796 | 1.212718 | 0.206997 |
| utility_base20 | 0.552635 | 1.222136 | 0.219154 |
| utility_aug28 | 0.544372 | 1.205818 | 0.206997 |

## Explicit proposals

```json
{
  "low_prior_aug28": {
    "selected_n": 2,
    "oracle_n": 312,
    "selected_extra_ade": 0.7478624433279037
  },
  "equal_prior_aug28": {
    "selected_n": 1,
    "oracle_n": 312,
    "selected_extra_ade": 0.4519229829311371
  },
  "utility_base20": {
    "selected_n": 0,
    "oracle_n": 0,
    "selected_extra_ade": null
  },
  "utility_aug28": {
    "selected_n": 76,
    "oracle_n": 312,
    "selected_extra_ade": 0.13191800808386975
  }
}
```

## Paired bootstrap (conditional on one trained seed)

- equal_prior_aug28 vs low_prior_aug28, selected_ade: +0.001390 [-0.001266, +0.003708]
- equal_prior_aug28 vs low_prior_aug28, selected_fde: +0.002987 [-0.003377, +0.008575]
- equal_prior_aug28 vs low_prior_aug28, oracle_ade: +0.000000 [+0.000000, +0.000000]
- utility_aug28 vs utility_base20, selected_ade: -0.008263 [-0.017313, +0.000958]
- utility_aug28 vs utility_base20, selected_fde: -0.016318 [-0.037643, +0.005128]
- utility_aug28 vs utility_base20, oracle_ade: -0.012157 [-0.014606, -0.009893]
- utility_base20 vs native, selected_ade: +0.010203 [+0.000031, +0.020969]
- utility_base20 vs native, selected_fde: +0.024820 [+0.000537, +0.049390]
- utility_base20 vs native, oracle_ade: +0.000000 [+0.000000, +0.000000]
- utility_aug28 vs native, selected_ade: +0.001940 [-0.007499, +0.011314]
- utility_aug28 vs native, selected_fde: +0.008502 [-0.014497, +0.031487]
- utility_aug28 vs native, oracle_ade: -0.012157 [-0.014592, -0.009831]
- equal_prior_aug28 vs native, selected_ade: +0.005365 [+0.000642, +0.010104]
- equal_prior_aug28 vs native, selected_fde: +0.015402 [+0.004479, +0.026546]
- equal_prior_aug28 vs native, oracle_ade: -0.012157 [-0.014563, -0.009855]