# EXP-187D Social Feasibility Audit

> **INVALIDATED PROTOCOL (v1): DO NOT CITE.** The locked training and official
> evaluation supervise only the focal agent (`prediction[:, :1]`), while this
> result computed pairwise collisions among all model outputs. Non-focal model
> trajectories are not valid benchmark predictions. The corrected v2 protocol
> evaluates the supervised focal Top-1 trajectory against ground-truth futures
> of valid neighbours. Re-run the experiment with the current script; its cache
> version check will replace all v1 profiles.

Decision: **ACCURACY_SUPPORTED_SOCIAL_FEASIBILITY_NOT_SUPPORTED**

| Model | minADE20 | minFDE20 | Pair collision | Pair near collision | Mean min clearance |
|---|---:|---:|---:|---:|---:|
| E0 | 0.230288 | 0.377202 | 1.141% | 4.949% | 3.0018 m |
| E1 | 0.224059 | 0.365715 | 0.570% | 4.578% | 2.9894 m |
| E2 | 0.227909 | 0.370242 | 0.909% | 5.015% | 2.9721 m |
| F16 | 0.218105 | 0.353215 | 0.867% | 5.348% | 2.9864 m |
