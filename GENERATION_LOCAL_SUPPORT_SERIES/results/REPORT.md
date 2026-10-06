# Generation local support pilot

LOCAL_SUPPORT_PILOT_GATE_FAIL_REVIEW_REQUIRED

ETH seed1 reused development diagnosis; no held-out test; K=20 for every arm

| Arm | minADE20 | minFDE20 | Top3 ADE | Near .3m count | Duplicate fraction |
|---|---:|---:|---:|---:|---:|
| native | 0.219154 | 0.399725 | 0.326190 | 2.230 | 0.0020 |
| oracle_only_best | 0.219040 | 0.407475 | 0.320150 | 2.410 | 0.0020 |
| oracle_only_final | 0.217590 | 0.402118 | 0.318612 | 2.435 | 0.0020 |
| local_support_best | 0.220011 | 0.407114 | 0.319803 | 2.473 | 0.0020 |
| local_support_final | 0.218529 | 0.399027 | 0.309694 | 2.752 | 0.0021 |

## Gates

- oracle_gain: False
- support_gain: False
- duplicate_guard: True
- acceleration_guard: False
- final_consistency: True

Top3 improvement alone is not proof of intention understanding. Fixed-score selected error is diagnostic because scorer logits were not retrained.