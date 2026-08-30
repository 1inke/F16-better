# EXP-156 Fold-local official ETH/UCY validation

This experiment first audits the five processed leave-one-out folds, then reproduces the current decoder without cross-fold checkpoints: H3 three-anchor proposal, V2 bounded wide correction, and M2 anchored mode template. Only M2 receives the held-out test archive. The protocol is 8 observed frames and 12 predicted frames at 2.5 Hz (3.2 s / 4.8 s).
