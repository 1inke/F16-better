# EXP-157 fold-stage attribution

- EXP-157A evaluates the already selected H3, V2 and M2 checkpoints on each held-out fold without retraining.
- EXP-157B trains the old kinematic K20 decoder under the exact EXP-156 data split, 4.8 s horizon, optimizer, schedule, augmentation and checkpoint protocol.

Test archives do not participate in checkpoint selection.
