# Causal history encoder

HISTORY_ENCODER_GATE_FAIL

Repeated ETH dev; nativeK20; flat-historyMLP43742 vs GRU43736 parameters,6difference(.014%); shared decoder identical paired initialization, encoders distinct; six fresh35epoch paired-seed runs; 8observed velocity increments/observedspeed+.03,32historyfeatures; original observation/path/family also retained; originalbalancedjointwinner+regularizers+protection.1/bound1.5/fitcal/constrainedcal; no pooledsetcontext/basis/newloss/proposals/test/backbone/scorer. Local auxiliary observation encoder is trainable; original F16 backbone frozen. Generic zero/mean_std labels in NPZ/CSV represent flat/gru only.

| Arm | Policy | Group | minADE | minFDE | Joint |
|---|---|---|---:|---:|---:|
| flat_mlp | constrained | overall | 0.217313 | 0.399521 | 0.430280 |
| flat_mlp | constrained | change | 0.365747 | 0.690354 | 0.736113 |
| flat_mlp | legacy | overall | 0.216833 | 0.399733 | 0.429781 |
| flat_mlp | legacy | change | 0.365571 | 0.693888 | 0.736591 |
| flat_mlp | final | overall | 0.217578 | 0.400922 | 0.430899 |
| flat_mlp | final | change | 0.376000 | 0.716978 | 0.760469 |
| gru | constrained | overall | 0.217284 | 0.399363 | 0.430168 |
| gru | constrained | change | 0.365707 | 0.689777 | 0.735843 |
| gru | legacy | overall | 0.216892 | 0.399945 | 0.429901 |
| gru | legacy | change | 0.366582 | 0.696670 | 0.738673 |
| gru | final | overall | 0.216757 | 0.399306 | 0.429484 |
| gru | final | change | 0.370982 | 0.704231 | 0.748934 |

{
  "change_ade_fde_gain_5pct": false,
  "protected_groups_guard": true,
  "each_seed_change_improves": true,
  "native_and_zero_ci": false,
  "all_seed_feasible": true,
  "coverage_guard": true,
  "duplicate_guard": true,
  "kinematic_guard": true,
  "same_path_joint_guard": true
}