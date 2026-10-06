# PIF1 validation report

Decision: **STOP_PIF1_AFTER_SEED1**

| Condition | minADE | minFDE | Selected ADE | Recall@1 | Mean oracle rank | Duplicate sample rate |
|---|---:|---:|---:|---:|---:|---:|
| c00 | 0.219566 | 0.396031 | 0.538869 | 20.18% | 6.320 | 12.47% |
| pif1_best | 0.244400 | 0.416886 | 0.572566 | 19.53% | 6.772 | 17.08% |
| pif1_final | 0.244400 | 0.416886 | 0.572566 | 19.53% | 6.772 | 17.08% |

The pilot is validation-only. Best and final checkpoints must agree before any seed expansion; no test path is used.
