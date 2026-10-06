# F16 H3→V2 Diversity Stage Attribution

- Phase: `h3`
- Scope: ETH, seed 1, the frozen 5% scene-tail validation split.
- Test policy: no test path is used by this analysis or by the new training runner.
- Bootstrap: paired over identical validation sample IDs; descriptive validation evidence, not independent confirmation.
- Verdict: **CASE_B_NOT_MAIN_CAUSE** — H3-NoDiv does not cross any preregistered geometry threshold.

## Stage attribution summary

| Condition | H3 w | V2 w | Selected ADE | minADE | minFDE | Regret | Spread | Support@1m | ΔADE1,2 median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

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

## Preregistered engineering scales

- Endpoint spread: |Δ| ≥ 0.10 m.
- Support@1m: |Δ| ≥ 0.5 candidate/sample.
- ΔADE1,2: |Δ| ≥ 0.02 m.
- minADE/minFDE: relative change ≥ 2% is a meaningful cost or gain.

## Provenance

```json
{
  "phase": "h3",
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
    }
  }
}
```
