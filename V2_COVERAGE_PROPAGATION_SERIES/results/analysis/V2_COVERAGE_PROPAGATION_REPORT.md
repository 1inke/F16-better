# H3 Coverage → V2 Propagation Experiment

- Scope: ETH, seed 1, fixed 5% scene-tail validation split (N=1,516).
- New training: one V2 run from H3 D=.05/O=0; V2 itself remains D=.05/O=.20.
- Reference: existing EXP-173 C00 from H3 D=.05/O=.20; this analysis reads validation data only.
- Sensitivity: oracle-combo best checkpoint and matched epoch-20 final model.
- Paired bootstrap is conditional on the trained models and does not estimate training-seed uncertainty.
- Verdict: **MIXED_OR_CHECKPOINT_SENSITIVE** — The propagation evidence does not jointly pass quality, local-support, selection-guard, and checkpoint-consistency gates.

## Best checkpoint summary

| Condition | Stage | H3 coverage | V2 coverage | J | minADE | minFDE | Same-path FDE | Selected ADE | Spread | Support@1m | ADE<1m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H3 O=.20 | h3 | 0.20 | NA | 0.4322 | 0.2254 | 0.4136 | 0.4788 | 0.5404 | 2.725 | 3.36 | 10.57 |
| H3 O=0 | h3 | 0.00 | NA | 0.4461 | 0.2330 | 0.4261 | 0.4891 | 0.5468 | 2.860 | 2.74 | 9.74 |
| H3 O=.20 -> V2 O=.20 | v2 | 0.20 | 0.20 | 0.4176 | 0.2196 | 0.3960 | 0.4648 | 0.5389 | 2.430 | 4.47 | 12.20 |
| H3 O=0 -> V2 O=.20 | v2 | 0.00 | 0.20 | 0.4215 | 0.2212 | 0.4007 | 0.4664 | 0.5397 | 2.442 | 4.36 | 11.92 |

## Final checkpoint summary

| Condition | Stage | H3 coverage | V2 coverage | J | minADE | minFDE | Same-path FDE | Selected ADE | Spread | Support@1m | ADE<1m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H3 O=.20 | h3 | 0.20 | NA | 0.4322 | 0.2254 | 0.4136 | 0.4788 | 0.5404 | 2.725 | 3.36 | 10.57 |
| H3 O=0 | h3 | 0.00 | NA | 0.4461 | 0.2330 | 0.4261 | 0.4891 | 0.5468 | 2.860 | 2.74 | 9.74 |
| H3 O=.20 -> V2 O=.20 | v2 | 0.20 | 0.20 | 0.4176 | 0.2196 | 0.3960 | 0.4648 | 0.5389 | 2.430 | 4.47 | 12.20 |
| H3 O=0 -> V2 O=.20 | v2 | 0.00 | 0.20 | 0.4226 | 0.2220 | 0.4012 | 0.4665 | 0.5388 | 2.364 | 4.81 | 12.40 |

## Coverage-source effect before and after V2

Delta is coverage-on source minus coverage-off source. Negative error delta and positive support delta are improvements.

| Checkpoint | Stage | Metric | Delta | Relative | 95% CI |
|---|---|---|---:|---:|---:|
| best | h3 | j | -0.01384 | -3.10% | [-0.02022, -0.00750] |
| best | h3 | minade | -0.00763 | -3.27% | [-0.01081, -0.00445] |
| best | h3 | minfde | -0.01243 | -2.92% | [-0.02043, -0.00465] |
| best | h3 | samepath_fde | -0.01027 | -2.10% | [-0.02169, 0.00112] |
| best | h3 | selected_ade | -0.00643 | -1.18% | [-0.01145, -0.00140] |
| best | h3 | endpoint_spread | -0.13496 | -4.72% | [-0.14447, -0.12527] |
| best | h3 | support_1p0 | 0.62269 | 22.74% | [0.56464, 0.68272] |
| best | h3 | ade_lt_1p0 | 0.82586 | 8.48% | [0.74868, 0.90633] |
| best | v2 | j | -0.00395 | -0.94% | [-0.01007, 0.00223] |
| best | v2 | minade | -0.00164 | -0.74% | [-0.00460, 0.00135] |
| best | v2 | minfde | -0.00462 | -1.15% | [-0.01231, 0.00322] |
| best | v2 | samepath_fde | -0.00164 | -0.35% | [-0.01286, 0.00940] |
| best | v2 | selected_ade | -0.00080 | -0.15% | [-0.00761, 0.00628] |
| best | v2 | endpoint_spread | -0.01169 | -0.48% | [-0.02216, -0.00128] |
| best | v2 | support_1p0 | 0.10884 | 2.50% | [0.06135, 0.15633] |
| best | v2 | ade_lt_1p0 | 0.28298 | 2.37% | [0.21834, 0.34830] |
| final | h3 | j | -0.01384 | -3.10% | [-0.02034, -0.00759] |
| final | h3 | minade | -0.00763 | -3.27% | [-0.01076, -0.00450] |
| final | h3 | minfde | -0.01243 | -2.92% | [-0.02017, -0.00472] |
| final | h3 | samepath_fde | -0.01027 | -2.10% | [-0.02154, 0.00102] |
| final | h3 | selected_ade | -0.00643 | -1.18% | [-0.01152, -0.00141] |
| final | h3 | endpoint_spread | -0.13496 | -4.72% | [-0.14468, -0.12544] |
| final | h3 | support_1p0 | 0.62269 | 22.74% | [0.56464, 0.68008] |
| final | h3 | ade_lt_1p0 | 0.82586 | 8.48% | [0.74736, 0.90501] |
| final | v2 | j | -0.00499 | -1.18% | [-0.01125, 0.00132] |
| final | v2 | minade | -0.00241 | -1.09% | [-0.00540, 0.00063] |
| final | v2 | minfde | -0.00515 | -1.28% | [-0.01292, 0.00265] |
| final | v2 | samepath_fde | -0.00172 | -0.37% | [-0.01324, 0.00986] |
| final | v2 | selected_ade | 0.00010 | 0.02% | [-0.00653, 0.00678] |
| final | v2 | endpoint_spread | 0.06590 | 2.79% | [0.05425, 0.07740] |
| final | v2 | support_1p0 | -0.34499 | -7.17% | [-0.41031, -0.28034] |
| final | v2 | ade_lt_1p0 | -0.19921 | -1.61% | [-0.26649, -0.12995] |

## Preregistered advancement gates

- Same direction at best and matched epoch 20.
- V2 J improves by at least 2%, or minADE/minFDE has one ≥2% gain while the other degrades by no more than 0.5%.
- Support@1m increases by at least 0.5 candidate/sample.
- Selected ADE does not degrade by more than 0.5%.
- If |J change| <1% and |Support@1m change| <0.5 at both checkpoints, treat the H3 gain as absorbed by V2.

| Checkpoint | V2 J rel. | V2 minADE rel. | V2 minFDE rel. | Support@1m delta | Selected ADE rel. |
|---|---:|---:|---:|---:|---:|
| best | -0.94% | -0.74% | -1.15% | 0.109 | -0.15% |
| final | -1.18% | -1.09% | -1.28% | -0.345 | 0.02% |

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
      "split": "validation_only_scene_tail_5pct"
    },
    "design": {
      "reference": "H3 D=.05/O=.20 -> V2 D=.05/O=.20 (existing C00)",
      "treatment": "H3 D=.05/O=0 -> V2 D=.05/O=.20 (one new training)",
      "single_changed_factor": "H3 oracle coverage weight in the source checkpoint",
      "new_training_count": 1,
      "primary_checkpoint": "best_selected.pth",
      "sensitivity_checkpoint": "final_model.pth at epoch 20"
    },
    "test_policy": {
      "new_training_receives_test_paths": false,
      "analysis_reads_test_data": false,
      "existing_reference_note": "The historical C00 artifact contains an old test evaluation, but this runner and analyzer never load or report it."
    },
    "validation_identity": {
      "sample_count": 1516,
      "sample_id_sha256": "dfd29348a6eb3d3ac601c3fae60250d86762a85fb94ea326d050d582621e870d",
      "all_required_runs_exact_match": true
    },
    "sources": {
      "h3_on": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "h3_off": "C:\\Users\\linke\\Documents\\New project\\H3_OBJECTIVE_ATTRIBUTION_SERIES\\results\\d1_o0\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "v2_on": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\v2\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
      "v2_off": "C:\\Users\\linke\\Documents\\New project\\V2_COVERAGE_PROPAGATION_SERIES\\results\\v2_from_h3_coverage_off\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth"
    },
    "source_hashes": {
      "h3_on": "8fee24d0bd7b009e6e6ee280a0eee4e0e6c345d750733754741192f3adf0bb57",
      "h3_off": "6d2bfd0b4fec2d4b2272b18cb9ba78976e48ad859a15afad4b5aab33e7868c3d",
      "v2_on": "691e9f9ff6b7dc1df1c7d5f4901d97ea9885fac891099c84044bf56dd98d2341",
      "v2_off": "91442c22cd6b87c357da0348e17675024c28c46069a2e3d0952b596e9072b810"
    },
    "inherited_preflight": {
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
      }
    }
  },
  "conditions": {
    "h3_on": {
      "best": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "sha256": "8fee24d0bd7b009e6e6ee280a0eee4e0e6c345d750733754741192f3adf0bb57",
        "h3_source": null,
        "h3_source_sha256": null
      },
      "final": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\final_model.pth",
        "sha256": "f2fbe842647a1f1a5c2a5f7f044cf1d8722be3d9304fc56013daa9a77ef8f758",
        "h3_source": null,
        "h3_source_sha256": null
      }
    },
    "h3_off": {
      "best": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\H3_OBJECTIVE_ATTRIBUTION_SERIES\\results\\d1_o0\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "sha256": "6d2bfd0b4fec2d4b2272b18cb9ba78976e48ad859a15afad4b5aab33e7868c3d",
        "h3_source": null,
        "h3_source_sha256": null
      },
      "final": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\H3_OBJECTIVE_ATTRIBUTION_SERIES\\results\\d1_o0\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\final_model.pth",
        "sha256": "7062240f9496ff5a5678416f4091917ec6ada408e43ae04c3ecffd9d144dc542",
        "h3_source": null,
        "h3_source_sha256": null
      }
    },
    "v2_on": {
      "best": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\v2\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "sha256": "691e9f9ff6b7dc1df1c7d5f4901d97ea9885fac891099c84044bf56dd98d2341",
        "h3_source": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "h3_source_sha256": "8fee24d0bd7b009e6e6ee280a0eee4e0e6c345d750733754741192f3adf0bb57"
      },
      "final": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\v2\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\final_model.pth",
        "sha256": "f537fe2bd4969f98dd1d1f023948a2a36ae543a76c584ccf9fca3e1855735574",
        "h3_source": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\h3\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "h3_source_sha256": "8fee24d0bd7b009e6e6ee280a0eee4e0e6c345d750733754741192f3adf0bb57"
      }
    },
    "v2_off": {
      "best": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\V2_COVERAGE_PROPAGATION_SERIES\\results\\v2_from_h3_coverage_off\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "sha256": "91442c22cd6b87c357da0348e17675024c28c46069a2e3d0952b596e9072b810",
        "h3_source": "C:\\Users\\linke\\Documents\\New project\\H3_OBJECTIVE_ATTRIBUTION_SERIES\\results\\d1_o0\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "h3_source_sha256": "6d2bfd0b4fec2d4b2272b18cb9ba78976e48ad859a15afad4b5aab33e7868c3d"
      },
      "final": {
        "checkpoint": "C:\\Users\\linke\\Documents\\New project\\V2_COVERAGE_PROPAGATION_SERIES\\results\\v2_from_h3_coverage_off\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\final_model.pth",
        "sha256": "636f09ccb2d1a38cb45adf1217dcbe4b3dbbadc69aa5f19b325925da2d608230",
        "h3_source": "C:\\Users\\linke\\Documents\\New project\\H3_OBJECTIVE_ATTRIBUTION_SERIES\\results\\d1_o0\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
        "h3_source_sha256": "6d2bfd0b4fec2d4b2272b18cb9ba78976e48ad859a15afad4b5aab33e7868c3d"
      }
    }
  }
}
```
