# Protected balance: supervision and checkpoint isolation

Four matched arms: uniform, balanced, uniform_protected, balanced_protected, correction seeds1/2/3; all use existing K20 trajectories, frozen backbone/scorer, same architecture and original35epoch/LR/batch budgets. No score-based pruning or top3 support loss.

Protection: fit samples classified stable or transition use separate hinges for candidate-set minADE/minFDE above native errors ×1.01, divided by fit-native mean minADE/minFDE. Fixed coefficient0.1, no dev coefficient sweep. Groups and future enter losses only, never generator forward. This is a stronger per-sample training constraint than the group-level evaluation tolerance and may restrict useful redistribution; controls isolate that possibility.

All arms constrained calibration selection: overall/stable/transition minADE and minFDE<=native×1.01, then minimize normalized change ADE+FDE; overall normalized errors tie-break. If no trained epoch feasible, epoch0/native fallback is explicitly marked infeasible. Legacy calibration key and final checkpoint are secondary policies on the SAME trained models, not dev-selected alternatives.

Every epoch state saved; atomic resume includes optimizer, Python/NumPy/Torch/CUDA RNG and loader generator. Completion manifests preserve outputs. Original models untouched. Source/fit/cache/raw/model hashes locked; internal prior best/final diagnosis in prior_internal_calibration.csv. No held-out test access. Reused ETH dev and only3correctionseeds remain exploratory, not independent confirmation.

Prespecified primary balanced_protected/constrained: change ADE/FDE>=5%gain, protected groups<=1%harm, every seed direction consistent, bootstrap supports native and balanced-control comparisons, candidate family coverage/class count/duplicates/absolute kinematic guards. Gates do not declare general intention understanding.

Outputs results/summary.json, REPORT.md, per_seed_metrics.csv, aggregate_metrics.csv, bootstrap.csv, checkpoint/model-lock/exit manifests and explicit-ID predictions. Resume command accepts completed validated arms and epoch-level unfinished state; experiment sources cannot change under original lock.
