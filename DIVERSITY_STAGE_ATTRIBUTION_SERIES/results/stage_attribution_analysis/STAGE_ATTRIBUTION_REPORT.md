# F16 H3→V2 Diversity Stage Attribution

- Phase: `final`
- Scope: ETH, seed 1, the frozen 5% scene-tail validation split.
- Test policy: no test path is used by this analysis or by the new training runner.
- Bootstrap: paired over identical validation sample IDs; descriptive validation evidence, not independent confirmation.
- Verdict: **CASE_B_NOT_MAIN_CAUSE** — H3-NoDiv does not cross any preregistered geometry threshold.

## Stage attribution summary

| Condition | H3 w | V2 w | Selected ADE | minADE | minFDE | Regret | Spread | Support@1m | ΔADE1,2 median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C00 | 0.05 | 0.05 | 0.5389 | 0.2196 | 0.3960 | 0.3193 | 2.430 | 4.47 | 0.082 |
| C01 | 0.05 | 0.00 | 0.5378 | 0.2179 | 0.3972 | 0.3199 | 2.430 | 4.48 | 0.085 |
| C10 | 0.00 | 0.05 | 0.5378 | 0.2205 | 0.4008 | 0.3173 | 2.439 | 4.32 | 0.088 |
| C11 | 0.00 | 0.00 | 0.5446 | 0.2205 | 0.3990 | 0.3241 | 2.488 | 4.04 | 0.093 |

## H3-only formation check

| H3 condition | Spread | Support@0.5 | Support@1.0 | ADE<0.5 | ADE<1.0 | ΔADE1,2 median | minADE | minFDE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H3 w=.05 | 2.725 | 1.02 | 3.36 | 3.88 | 10.57 | 0.097 | 0.2254 | 0.4136 |
| H3 w=0 | 2.703 | 1.02 | 3.31 | 4.04 | 10.51 | 0.096 | 0.2259 | 0.4136 |

## Diversity activity at a hypothetical 0.4 m margin

The activity metrics are calculated for every condition, including weight=0 runs.

| Condition | Active pair rate | Active sample rate | Mean penalty | Active pairs/sample mean | median | Active distance mean |
|---|---:|---:|---:|---:|---:|---:|
| H3 w=.05 | 0.00092 | 0.1715 | 0.00028 | 0.18 | 0.0 | 0.1012 |
| H3 w=0 | 0.00091 | 0.1708 | 0.00028 | 0.17 | 0.0 | 0.0944 |
| C00 | 0.01288 | 0.1979 | 0.00184 | 2.45 | 0.0 | 0.2573 |
| C01 | 0.01292 | 0.1959 | 0.00204 | 2.46 | 0.0 | 0.2422 |
| C10 | 0.00913 | 0.1761 | 0.00137 | 1.74 | 0.0 | 0.2503 |
| C11 | 0.00232 | 0.1689 | 0.00070 | 0.44 | 0.0 | 0.0977 |

## Fixed C00 A-cohort

The sample IDs are frozen from C00 where regret ≥0.5 m and minADE <0.5 m.

| Condition | N | Spread | Support@1m | ADE<1m | ΔADE1,2 median | minADE | Selected ADE | Regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C00 | 285 | 2.711 | 1.90 | 8.08 | 0.182 | 0.2778 | 0.9792 | 0.7014 |
| C01 | 285 | 2.691 | 1.91 | 8.20 | 0.182 | 0.2792 | 0.9685 | 0.6893 |
| C10 | 285 | 2.677 | 1.99 | 8.14 | 0.170 | 0.2874 | 0.9553 | 0.6679 |
| C11 | 285 | 2.700 | 1.99 | 8.02 | 0.178 | 0.2855 | 0.9633 | 0.6778 |

## Paired bootstrap versus C00

Positive delta means the named condition minus C00; the CSV contains all ten preregistered metrics and 95% CIs.

| Condition | Metric | Mean delta | 95% CI |
|---|---|---:|---:|
| C01 | endpoint_spread | -0.00019 | [-0.00405, 0.00361] |
| C01 | support_0p5 | -0.02309 | [-0.04485, -0.00198] |
| C01 | support_1p0 | 0.00989 | [-0.01913, 0.03892] |
| C01 | ade_lt_0p5 | 0.12599 | [0.08575, 0.16557] |
| C01 | ade_lt_1p0 | 0.03034 | [-0.00264, 0.06398] |
| C01 | delta_ade_1_2 | 0.00256 | [0.00010, 0.00503] |
| C01 | minade | -0.00164 | [-0.00322, -0.00008] |
| C01 | minfde | 0.00116 | [-0.00266, 0.00495] |
| C01 | selected_ade | -0.00109 | [-0.00412, 0.00183] |
| C01 | regret | 0.00055 | [-0.00276, 0.00374] |
| C10 | endpoint_spread | 0.00865 | [-0.00147, 0.01897] |
| C10 | support_0p5 | -0.08377 | [-0.11675, -0.04947] |
| C10 | support_1p0 | -0.15106 | [-0.19855, -0.10290] |
| C10 | ade_lt_0p5 | -0.03694 | [-0.09433, 0.01913] |
| C10 | ade_lt_1p0 | -0.15633 | [-0.21900, -0.09299] |
| C10 | delta_ade_1_2 | 0.00403 | [-0.00048, 0.00850] |
| C10 | minade | 0.00093 | [-0.00227, 0.00427] |
| C10 | minfde | 0.00481 | [-0.00314, 0.01266] |
| C10 | selected_ade | -0.00112 | [-0.00790, 0.00558] |
| C10 | regret | -0.00204 | [-0.00877, 0.00461] |
| C11 | endpoint_spread | 0.05778 | [0.04727, 0.06839] |
| C11 | support_0p5 | -0.65831 | [-0.74472, -0.57520] |
| C11 | support_1p0 | -0.42810 | [-0.48615, -0.37137] |
| C11 | ade_lt_0p5 | -0.47625 | [-0.53760, -0.41557] |
| C11 | ade_lt_1p0 | -0.52441 | [-0.59235, -0.45515] |
| C11 | delta_ade_1_2 | 0.00936 | [0.00461, 0.01418] |
| C11 | minade | 0.00098 | [-0.00257, 0.00449] |
| C11 | minfde | 0.00295 | [-0.00580, 0.01151] |
| C11 | selected_ade | 0.00577 | [-0.00131, 0.01312] |
| C11 | regret | 0.00479 | [-0.00238, 0.01192] |

## Preregistered engineering scales

- Endpoint spread: |Δ| ≥ 0.10 m.
- Support@1m: |Δ| ≥ 0.5 candidate/sample.
- ΔADE1,2: |Δ| ≥ 0.02 m.
- minADE/minFDE: relative change ≥ 2% is a meaningful cost or gain.

## Provenance

```json
{
  "phase": "final",
  "fold": "eth",
  "seed": 1,
  "split": "validation_only_scene_tail_5pct",
  "validation_sample_count": 1516,
  "validation_sample_id_sha256": "dfd29348a6eb3d3ac601c3fae60250d86762a85fb94ea326d050d582621e870d",
  "test_accessed_by_this_analysis": false,
  "conditions": {
    "h3_w005": {
      "checkpoint": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "checkpoint_sha256": "8fee24d0bd7b009e6e6ee280a0eee4e0e6c345d750733754741192f3adf0bb57",
      "split_indices": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\split_indices.npz",
      "split_indices_sha256": "1f53e20b1d2462df27c6fe9d2032973fd92304f7616f7f0dd6f80da01de8197e",
      "h3_source": null,
      "h3_source_sha256": null
    },
    "h3_w000": {
      "checkpoint": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\h3_w000\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "checkpoint_sha256": "fb7a2a7744f85be965e3ab58053b9056999e323246c3ec82a35ad42b465c4e68",
      "split_indices": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\h3_w000\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\split_indices.npz",
      "split_indices_sha256": "1f53e20b1d2462df27c6fe9d2032973fd92304f7616f7f0dd6f80da01de8197e",
      "h3_source": null,
      "h3_source_sha256": null
    },
    "c00": {
      "checkpoint": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\v2\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "checkpoint_sha256": "691e9f9ff6b7dc1df1c7d5f4901d97ea9885fac891099c84044bf56dd98d2341",
      "split_indices": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\v2\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\split_indices.npz",
      "split_indices_sha256": "1f53e20b1d2462df27c6fe9d2032973fd92304f7616f7f0dd6f80da01de8197e",
      "h3_source": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "h3_source_sha256": "8fee24d0bd7b009e6e6ee280a0eee4e0e6c345d750733754741192f3adf0bb57"
    },
    "c01": {
      "checkpoint": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_LOSS_ABLATION_SERIES\\results\\weight_sweep\\divw_0p000_m_0p400\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "checkpoint_sha256": "2cf652b93b1569d5714ac7aadbc28caebf1b2596dc29db9b4ef969e07f773aa4",
      "split_indices": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_LOSS_ABLATION_SERIES\\results\\weight_sweep\\divw_0p000_m_0p400\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\split_indices.npz",
      "split_indices_sha256": "1f53e20b1d2462df27c6fe9d2032973fd92304f7616f7f0dd6f80da01de8197e",
      "h3_source": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "h3_source_sha256": "8fee24d0bd7b009e6e6ee280a0eee4e0e6c345d750733754741192f3adf0bb57"
    },
    "c10": {
      "checkpoint": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\c10_h3w000_v2w005\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "checkpoint_sha256": "85c5d4b8b5269d4ee28363a4b9dfc0883ff46730cb09eb0725beef8e8959633b",
      "split_indices": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\c10_h3w000_v2w005\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\split_indices.npz",
      "split_indices_sha256": "1f53e20b1d2462df27c6fe9d2032973fd92304f7616f7f0dd6f80da01de8197e",
      "h3_source": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\h3_w000\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "h3_source_sha256": "fb7a2a7744f85be965e3ab58053b9056999e323246c3ec82a35ad42b465c4e68"
    },
    "c11": {
      "checkpoint": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\c11_h3w000_v2w000\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "checkpoint_sha256": "89fd8f780c89cc90f57ba18e774176c235d713d6f872a0793ce935f57de32620",
      "split_indices": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\c11_h3w000_v2w000\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\split_indices.npz",
      "split_indices_sha256": "1f53e20b1d2462df27c6fe9d2032973fd92304f7616f7f0dd6f80da01de8197e",
      "h3_source": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\h3_w000\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "h3_source_sha256": "fb7a2a7744f85be965e3ab58053b9056999e323246c3ec82a35ad42b465c4e68"
    }
  }
}
```
