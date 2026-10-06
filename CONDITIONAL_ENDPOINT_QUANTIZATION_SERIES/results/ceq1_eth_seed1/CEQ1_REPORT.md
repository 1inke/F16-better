# CEQ1 条件端点分布量化实验

- 范围：ETH seed 1，固定 validation IDs，不读取测试集。
- 候选族：从训练折相似历史对应的真实未来构成条件经验分布，再按概率质量量化为 20 条候选。
- 分配原则：weighted Lloyd 最小化条件分布的量化误差；没有候选间排斥项、固定方向模板或硬配额。
- 评分：每个候选的 cluster probability mass；最高分与候选生成来自同一分布。
- 校准锁定参数：`{"neighbours": 64, "temperature": 0.5, "components": 20, "lloyd_iterations": 8}`。
- 判定：**STOP_CEQ_ROUTE** — The conditional empirical future family does not clear the geometry gate or collapses locally; do not build a neural distillation head.

## Validation 指标

| Family | Subset | N | J | minADE | minFDE | Selected ADE | Regret | Spread | Support@1m | R@1 | R@3 | Duplicate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c00 | all | 1516 | 0.4176 | 0.2196 | 0.3960 | 0.5389 | 0.3193 | 2.4301 | 4.468 | 0.202 | 0.361 | 0.0019 |
| c00 | high_motion | 270 | 0.5307 | 0.2833 | 0.4948 | 0.7031 | 0.4198 | 2.6246 | 3.370 | 0.104 | 0.274 | 0.0007 |
| c00 | non_high_motion | 1246 | 0.3931 | 0.2057 | 0.3746 | 0.5033 | 0.2975 | 2.3880 | 4.705 | 0.223 | 0.380 | 0.0022 |
| ceq1 | all | 1516 | 0.5576 | 0.2954 | 0.5244 | 0.8192 | 0.5237 | 1.8787 | 5.635 | 0.078 | 0.227 | 0.0690 |
| ceq1 | high_motion | 270 | 0.6999 | 0.3701 | 0.6595 | 1.0212 | 0.6511 | 2.2303 | 3.759 | 0.070 | 0.200 | 0.0157 |
| ceq1 | non_high_motion | 1246 | 0.5268 | 0.2793 | 0.4951 | 0.7754 | 0.4962 | 1.8025 | 6.042 | 0.080 | 0.233 | 0.0805 |

## CEQ1 minus C00 paired bootstrap

| Subset | Metric | Delta | Relative | 95% CI |
|---|---|---:|---:|---:|
| all | j | 0.14006 | 33.54% | [0.12256, 0.15808] |
| all | minade | 0.07588 | 34.56% | [0.06731, 0.08476] |
| all | minfde | 0.12837 | 32.41% | [0.10855, 0.14866] |
| all | selected_ade | 0.28032 | 52.02% | [0.25826, 0.30296] |
| all | regret | 0.20445 | 64.03% | [0.18264, 0.22624] |
| all | endpoint_spread | -0.55144 | -22.69% | [-0.58517, -0.51720] |
| all | support_1p0 | 1.16755 | 26.13% | [0.98549, 1.35092] |
| all | oracle_recall_1 | -0.12335 | -61.11% | [-0.14644, -0.09960] |
| all | oracle_recall_3 | -0.13391 | -37.11% | [-0.16491, -0.10290] |
| all | duplicate_pair_rate | 0.06703 | 3466.07% | [0.05805, 0.07635] |
| high_motion | j | 0.16916 | 31.87% | [0.12305, 0.21669] |
| high_motion | minade | 0.08678 | 30.63% | [0.06393, 0.11032] |
| high_motion | minfde | 0.16476 | 33.30% | [0.11308, 0.21973] |
| high_motion | selected_ade | 0.31809 | 45.24% | [0.25769, 0.38049] |
| high_motion | regret | 0.23131 | 55.10% | [0.16921, 0.29475] |
| high_motion | endpoint_spread | -0.39431 | -15.02% | [-0.47076, -0.31606] |
| high_motion | support_1p0 | 0.38889 | 11.54% | [0.07407, 0.71481] |
| high_motion | oracle_recall_1 | -0.03333 | -32.14% | [-0.08148, 0.01111] |
| high_motion | oracle_recall_3 | -0.07407 | -27.03% | [-0.14074, -0.00741] |
| high_motion | duplicate_pair_rate | 0.01501 | 2081.08% | [0.00957, 0.02105] |
| non_high_motion | j | 0.13376 | 34.03% | [0.11515, 0.15284] |
| non_high_motion | minade | 0.07351 | 35.73% | [0.06428, 0.08293] |
| non_high_motion | minfde | 0.12049 | 32.16% | [0.09912, 0.14248] |
| non_high_motion | selected_ade | 0.27214 | 54.07% | [0.24870, 0.29680] |
| non_high_motion | regret | 0.19862 | 66.76% | [0.17551, 0.22191] |
| non_high_motion | endpoint_spread | -0.58549 | -24.52% | [-0.62328, -0.54811] |
| non_high_motion | support_1p0 | 1.33628 | 28.40% | [1.12520, 1.54414] |
| non_high_motion | oracle_recall_1 | -0.14286 | -64.03% | [-0.16934, -0.11637] |
| non_high_motion | oracle_recall_3 | -0.14687 | -38.69% | [-0.18138, -0.11316] |
| non_high_motion | duplicate_pair_rate | 0.07830 | 3564.62% | [0.06755, 0.08941] |

## Train-only calibration

| K neighbours | Temperature | Objective | minADE | minFDE | Selected ADE |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.5 | 0.461544 | 0.1655 | 0.2732 | 0.6377 |
| 64 | 1.0 | 0.470906 | 0.1682 | 0.2770 | 0.6569 |
| 64 | 2.0 | 0.472986 | 0.1692 | 0.2791 | 0.6568 |
| 128 | 0.5 | 0.472396 | 0.1689 | 0.2751 | 0.6638 |
| 128 | 1.0 | 0.480775 | 0.1731 | 0.2836 | 0.6635 |
| 128 | 2.0 | 0.482175 | 0.1738 | 0.2848 | 0.6639 |
| 256 | 0.5 | 0.475064 | 0.1714 | 0.2832 | 0.6481 |
| 256 | 1.0 | 0.484980 | 0.1752 | 0.2919 | 0.6553 |
| 256 | 2.0 | 0.486409 | 0.1763 | 0.2935 | 0.6533 |

## 判定门槛

- `generation_quality`: False
- `selected_ade_improves_2pct`: False
- `support_1m_increases_0p25`: True
- `high_motion_j_not_worse_1pct`: False
- `duplicate_pair_rate_below_10pct`: True

## Provenance

```json
{
  "experiment": "CEQ1_CONDITIONAL_EMPIRICAL_FUTURE_QUANTIZATION",
  "scope": "ETH seed 1 validation-only feasibility pilot",
  "new_neural_training": false,
  "test_paths_read_or_passed": false,
  "candidate_family": "training-only conditional empirical future distribution; weighted full-trajectory Lloyd components; component mass is score",
  "not_used": [
    "pairwise repulsion",
    "fixed global endpoint prototypes",
    "hard candidate-family quota",
    "validation future in retrieval features"
  ],
  "split": {
    "train_count": 28791,
    "validation_count": 1516,
    "validation_id_sha256": "dfd29348a6eb3d3ac601c3fae60250d86762a85fb94ea326d050d582621e870d",
    "split_file_sha256": "1f53e20b1d2462df27c6fe9d2032973fd92304f7616f7f0dd6f80da01de8197e"
  },
  "baseline": {
    "checkpoint": "C:\\Users\\linke\\Documents\\New project\\E2_F16_END_TO_END_COMPARISON_SERIES\\EXP_173_E2_F16_END_TO_END\\results\\eth\\v2\\e3\\f16_temporal_social_cross\\seed_1\\val_scene_-1\\best_selected.pth",
    "checkpoint_sha256": "691e9f9ff6b7dc1df1c7d5f4901d97ea9885fac891099c84044bf56dd98d2341",
    "contract": {
      "decoder_class": "JointTrainableV2Decoder",
      "decoder_fingerprint": "4bec38da2b5907d9",
      "num_modes": 20,
      "decoder_experiment": "v2_joint_wide_endpoint_correction",
      "encoder_experiment": "none",
      "classification_weight": 0.5,
      "loss_class": "ModeQueryLoss",
      "fusion_alpha": 0.3,
      "oracle_coverage_weight": 0.2,
      "oracle_coverage_temperature": 0.1,
      "oracle_coverage_fde_weight": 0.25,
      "oracle_coverage_warmup_epochs": 0.0,
      "oracle_coverage_normalized": false,
      "scene_sampling": "none",
      "motion_difficulty_sampling": false,
      "motion_difficulty_strength": 0.8,
      "ema_decay": 0.995,
      "augmentation": {
        "rotation": false,
        "reflection": true,
        "speed_min": 0.9,
        "speed_max": 1.1,
        "probability": 0.5,
        "curvature_max": 0.0
      },
      "future_difficulty_weight": 0.0,
      "focal_heading_normalize": true,
      "velocity_loss_weight": 0.0,
      "acceleration_loss_weight": 0.0,
      "masked_history_weight": 0.0,
      "masked_history_probability": 0.15,
      "equivariance_weight": 0.0,
      "training_only_auxiliary_parameters": 0,
      "checkpoint_criterion": "oracle_combo",
      "hierarchical_decoder": {
        "cv_reference": false,
        "anchor_steps": [
          4,
          8,
          12
        ],
        "top_m": 20,
        "use_refiner": true
      },
      "parameter_counts": {
        "total": 321988,
        "trainable": 321411,
        "decoder": 54829
      }
    }
  },
  "smoke": false,
  "device": "cuda:0",
  "feature_dimension": 42,
  "calibration": {
    "fit_count": 28023,
    "calibration_count": 768,
    "seed": 20260922,
    "objective": "mean(J) + 0.25 * mean(selected_ADE)",
    "selected": {
      "neighbours": 64,
      "temperature": 0.5,
      "components": 20,
      "lloyd_iterations": 8
    }
  },
  "validation_count_evaluated": 1516,
  "motion_threshold_train_q75": 0.1424953192472458,
  "verdict": {
    "code": "STOP_CEQ_ROUTE",
    "explanation": "The conditional empirical future family does not clear the geometry gate or collapses locally; do not build a neural distillation head.",
    "gates": {
      "generation_quality": false,
      "selected_ade_improves_2pct": false,
      "support_1m_increases_0p25": true,
      "high_motion_j_not_worse_1pct": false,
      "duplicate_pair_rate_below_10pct": true
    }
  }
}
```
