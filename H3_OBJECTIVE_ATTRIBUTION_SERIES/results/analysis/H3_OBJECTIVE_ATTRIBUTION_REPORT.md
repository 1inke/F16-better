# H3 Diversity × Oracle-Coverage Objective Attribution

- Scope: ETH, seed 1, fixed 5% scene-tail validation split; no test access.
- Primary: each condition's oracle-combo best checkpoint.
- Sensitivity: all conditions at matched epoch 30 (`final_model.pth`).
- Paired bootstrap conditions on the trained models and does not measure training-seed uncertainty.
- Verdict: **COVERAGE_IS_USEFUL_ALLOCATOR** — Coverage materially improves GT-local allocation and/or oracle quality in both checkpoint views.

## Best checkpoint summary

| Condition | Diversity | Coverage | Selected ADE | minADE | minFDE | Spread | Support@0.5 | Support@1m | ADE<1m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1_O1 | 0.05 | 0.20 | 0.5404 | 0.2254 | 0.4136 | 2.725 | 1.02 | 3.36 | 10.57 |
| D0_O1 | 0.00 | 0.20 | 0.5433 | 0.2259 | 0.4136 | 2.703 | 1.02 | 3.31 | 10.51 |
| D1_O0 | 0.05 | 0.00 | 0.5468 | 0.2330 | 0.4261 | 2.860 | 0.92 | 2.74 | 9.74 |
| D0_O0 | 0.00 | 0.00 | 0.5457 | 0.2288 | 0.4210 | 2.832 | 0.96 | 2.76 | 9.69 |

## Final checkpoint summary

| Condition | Diversity | Coverage | Selected ADE | minADE | minFDE | Spread | Support@0.5 | Support@1m | ADE<1m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1_O1 | 0.05 | 0.20 | 0.5404 | 0.2254 | 0.4136 | 2.725 | 1.02 | 3.36 | 10.57 |
| D0_O1 | 0.00 | 0.20 | 0.5433 | 0.2259 | 0.4136 | 2.703 | 1.02 | 3.31 | 10.51 |
| D1_O0 | 0.05 | 0.00 | 0.5468 | 0.2330 | 0.4261 | 2.860 | 0.92 | 2.74 | 9.74 |
| D0_O0 | 0.00 | 0.00 | 0.5419 | 0.2288 | 0.4222 | 2.839 | 0.97 | 2.78 | 9.73 |

## Coverage main effect

Positive delta means coverage=.20 minus coverage=0. For ADE/FDE, negative is better; for support, positive is better.

| Checkpoint | Metric | Delta | 95% CI |
|---|---|---:|---:|
| best | endpoint_spread | -0.13222 | [-0.14091, -0.12341] |
| best | support_0p5 | 0.07520 | [0.05342, 0.09664] |
| best | support_1p0 | 0.58674 | [0.53166, 0.64182] |
| best | ade_lt_1p0 | 0.82652 | [0.75561, 0.89678] |
| best | minade | -0.00525 | [-0.00777, -0.00274] |
| best | minfde | -0.00993 | [-0.01611, -0.00377] |
| best | selected_ade | -0.00445 | [-0.00866, -0.00027] |
| final | endpoint_spread | -0.13568 | [-0.14438, -0.12724] |
| final | support_0p5 | 0.07355 | [0.05211, 0.09532] |
| final | support_1p0 | 0.57454 | [0.52144, 0.62863] |
| final | ade_lt_1p0 | 0.80310 | [0.73285, 0.87764] |
| final | minade | -0.00527 | [-0.00774, -0.00285] |
| final | minfde | -0.01051 | [-0.01677, -0.00445] |
| final | selected_ade | -0.00252 | [-0.00664, 0.00162] |

## Decision scales

- Endpoint spread: |Δ| ≥ 0.10 m.
- Support@1m or ADE<1m count: |Δ| ≥ 0.5 candidate/sample.
- minADE/minFDE: relative |Δ| ≥ 2%.
- A mechanism claim requires the same direction at best checkpoint and matched epoch 30.

## Provenance

```json
{
  "fold": "eth",
  "seed": 1,
  "validation_only": true,
  "test_accessed_by_this_analysis": false,
  "validation_sample_count": 1516,
  "validation_id_sha256": "dfd29348a6eb3d3ac601c3fae60250d86762a85fb94ea326d050d582621e870d",
  "preflight": {
    "scope": {
      "fold": "eth",
      "seed": 1,
      "split": "validation_only"
    },
    "test_policy": "test paths are intentionally not passed to new training commands",
    "validation_identity": {
      "sample_count": 1516,
      "sample_id_sha256": "dfd29348a6eb3d3ac601c3fae60250d86762a85fb94ea326d050d582621e870d",
      "existing_h3_c00_c01_exact_match": true
    },
    "config_audit": {
      "official_sha256": "d0a826f632b728110dd22c16a7449d97dd153ba60f219a01548d1482da4cdfd3",
      "div005_sha256": "d0a826f632b728110dd22c16a7449d97dd153ba60f219a01548d1482da4cdfd3",
      "div000_sha256": "70665528d903f7a68d8c908c6ec3ecc80784e83eb63e9e6db7c92085441e8b9d",
      "allowed_difference": "MODEL.DECODER.DIVERSITY_WEIGHT",
      "official_value": 0.05,
      "div000_value": 0.0
    },
    "inputs": {
      "data": "C:\\Users\\linke\\Desktop\\DRSE\\ASTRA_ACCURACY_Q20\\datasets\\eth_ucy\\processed_data_diverse\\eth_data_train.npy",
      "num": "C:\\Users\\linke\\Desktop\\DRSE\\ASTRA_ACCURACY_Q20\\datasets\\eth_ucy\\processed_data_diverse\\eth_num_train.npy",
      "h3_new": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\h3_w000\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "h3_new_split": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\h3_w000\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\split_indices.npz",
      "h3_baseline": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "c00": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\v2\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "c01": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_LOSS_ABLATION_SERIES\\results\\weight_sweep\\divw_0p000_m_0p400\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth"
    },
    "input_hashes": {
      "h3_new": "fb7a2a7744f85be965e3ab58053b9056999e323246c3ec82a35ad42b465c4e68",
      "h3_new_split": "1f53e20b1d2462df27c6fe9d2032973fd92304f7616f7f0dd6f80da01de8197e",
      "h3_baseline": "8fee24d0bd7b009e6e6ee280a0eee4e0e6c345d750733754741192f3adf0bb57",
      "c00": "691e9f9ff6b7dc1df1c7d5f4901d97ea9885fac891099c84044bf56dd98d2341",
      "c01": "2cf652b93b1569d5714ac7aadbc28caebf1b2596dc29db9b4ef969e07f773aa4"
    },
    "objective_attribution": {
      "design": "2x2 diversity weight {.05,0} x oracle coverage weight {.20,0}",
      "new_training_cells": [
        "d1_o0",
        "d0_o0"
      ],
      "reused_cells": [
        "d1_o1",
        "d0_o1"
      ],
      "primary_checkpoint": "best_selected.pth",
      "matched_epoch_sensitivity": "final_model.pth at epoch 30",
      "validation_id_sha256": "dfd29348a6eb3d3ac601c3fae60250d86762a85fb94ea326d050d582621e870d",
      "test_policy": "no test paths are passed"
    }
  },
  "conditions": {
    "d1_o1": {
      "best": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "sha256": "8fee24d0bd7b009e6e6ee280a0eee4e0e6c345d750733754741192f3adf0bb57"
      },
      "final": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\final_model.pth",
        "sha256": "f2fbe842647a1f1a5c2a5f7f044cf1d8722be3d9304fc56013daa9a77ef8f758"
      }
    },
    "d0_o1": {
      "best": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\h3_w000\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "sha256": "fb7a2a7744f85be965e3ab58053b9056999e323246c3ec82a35ad42b465c4e68"
      },
      "final": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\DIVERSITY_STAGE_ATTRIBUTION_SERIES\\results\\h3_w000\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\final_model.pth",
        "sha256": "6d8d6dcc2603af091ac3cd3734ca87e165faf6c0f2f110a710339ec161eb0bba"
      }
    },
    "d1_o0": {
      "best": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\H3_OBJECTIVE_ATTRIBUTION_SERIES\\results\\d1_o0\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "sha256": "6d2bfd0b4fec2d4b2272b18cb9ba78976e48ad859a15afad4b5aab33e7868c3d"
      },
      "final": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\H3_OBJECTIVE_ATTRIBUTION_SERIES\\results\\d1_o0\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\final_model.pth",
        "sha256": "7062240f9496ff5a5678416f4091917ec6ada408e43ae04c3ecffd9d144dc542"
      }
    },
    "d0_o0": {
      "best": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\H3_OBJECTIVE_ATTRIBUTION_SERIES\\results\\d0_o0\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "sha256": "8380d6195f5c05c81b3911dfa25a272ef14cb2ee77d492d81705f4fe6b17963f"
      },
      "final": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\H3_OBJECTIVE_ATTRIBUTION_SERIES\\results\\d0_o0\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\final_model.pth",
        "sha256": "001f422f12b8b4ec256f8083890aa3e23411c6bced931ff158a9f6218ef005a8"
      }
    }
  }
}
```
