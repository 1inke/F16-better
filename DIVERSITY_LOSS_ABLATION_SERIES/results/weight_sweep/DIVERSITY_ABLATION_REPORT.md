# F16 Diversity Loss Ablation Report

## Protocol

- EXP-173 V2 / F16 / ETH validation only
- Same H3 initialization, same scene-tail split, same seed and training schedule
- Only diversity weight/margin are changed
- Baseline: weight=0.05, margin=0.4

## Baseline reproduction check

- selected_ade: current=0.538868845, archived=0.538868845, abs_delta=0.000e+00
- minade: current=0.219565526, archived=0.219565526, abs_delta=0.000e+00
- minfde: current=0.396031290, archived=0.396031290, abs_delta=0.000e+00

## Condition summary

| condition | w | margin | Selected ADE | minADE | minFDE | regret | spread | support@1m | ADE<1m count | ΔADE1,2 median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| divw_0p000_m_0p400 (s1) | 0.000 | 0.400 | 0.5378 | 0.2179 | 0.3972 | 0.3199 | 2.430 | 4.48 | 12.23 | 0.085 |
| divw_0p010_m_0p400 (s1) | 0.010 | 0.400 | 0.5380 | 0.2188 | 0.3986 | 0.3191 | 2.432 | 4.50 | 12.20 | 0.085 |
| divw_0p025_m_0p400 (s1) | 0.025 | 0.400 | 0.5382 | 0.2188 | 0.3964 | 0.3194 | 2.437 | 4.48 | 12.18 | 0.084 |
| divw_0p050_m_0p400 (s1) | 0.050 | 0.400 | 0.5389 | 0.2196 | 0.3960 | 0.3193 | 2.430 | 4.47 | 12.20 | 0.082 |

## Interpretation rule

The diversity-loss hypothesis is supported if reducing diversity consistently decreases endpoint spread and increases GT endpoint support / reduces oracle isolation, while minADE/minFDE do not materially deteriorate. A single improved scalar is insufficient; use the paired bootstrap table to inspect direction and uncertainty.

Do not inspect test results until a setting is selected on validation and then confirmed across seeds.