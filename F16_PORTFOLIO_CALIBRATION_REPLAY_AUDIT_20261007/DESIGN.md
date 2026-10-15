# Frozen portfolio and calibration replay audit — 2026-10-07

Question: do the latest compact-portfolio and selective-interface findings survive a reproducible preprocessing chain and an equal-K native control?

This is an exploratory nonformal diagnosis of existing checkpoints, no parameter updates. Three existing grouped-FIT splits, seeds42/43/44, original FIT only. No DEV/test/new data. Original whole F16 is already exposed to the pool; outer cohorts are reused and overlap across protocols. No independent generalization or information-theoretic ceiling claim.

Freeze existing intent, BestOf4, R2 and M1 weights/buffers. AST-isolate definitions without executing legacy top-level code. Save all statistics and IDs with the new outputs. Same complete-candidate ADE<=.2 AND FDE<=.4, native top1/native top3/native20/C1/C2/C3/full24 on identical IDs. GT-free complete-link .10 meanpath/.20 endpoint effective clusters. All five pooled motion measures dt=.4 retain the original1% guard; larger diversity is not automatically better. Existing original major criteria remain unchanged; this diagnosis cannot upgrade baseline.

Preprocessing comparison, all checkpoint weights fixed:
A: reproduce M3 outer candidate/set/calibrator standardization and outer entropy median/p90; label transductive.
B: reproduce M1 train-subset5000 candidate/set standardization, calibrator train-subset standardization; threshold median/p90 from the same train subset. Threshold constants .6/.5/.9 inherited, no outer tuning.
C: feed R2 its full-training candidate/set statistics (its actual training contract), then derive calibrator train-subset statistics from those scores. Existing calibrator was trained under B, so C is a frozen input-contract stress test, not proof a repaired calibrator works.

The three-way softmax target softmax(-2J+3good) is quality utility, not future occurrence probability. Report original utility cross-entropy/squared error with actual uniform baselines, selected-good binary ECE/Brier/NLL and probability-weighted ADE/FDE separately. Multiple good candidates are not exclusive events. Report all-windows success, retained-only success, retained coverage, hypotheses count and native controls on the identical retained windows. No safety/collision avoidance claims.

Save per-window candidate errors/trajectories/scores/probabilities/tiers and reconstruct arithmetic from saved arrays. All relevant input/source/checkpoint/report SHA before/after; do not claim every historic file globally revalidated. Fixed CPU geometry and batch128 GPU model inference; process6GiB, physical+commit reserve6GiB. Actual first-job timing and measured resources before OS launch and ETA. Source mismatch or substantive ID/resource/numeric failure stops and preserves evidence. No old workers/PIDs queried. One current startup, one half-elapsed check, one completion check, quiet between.

Decision after audit: if same-K learned portfolio does not reliably outperform native3 in coverage/J/motion, retain native3 as compact reference and quantify the learned proposals' conditional rescue. If input-contract failures dominate, save a precise corrected calibration design before any authorized training. If utility confidence is poorly aligned with empirical goodness, do not call it calibrated occurrence probability. Every three new formal experiments still requires full review; this audit does not count as training.

Counter-review1 (same assistant, not external): comparing3 to1 confounds output budget; absence of successful tested Top1 gates is not an intrinsic information ceiling; reused outer cohorts support exploratory checks only. The proposed audit resolves measurable pipeline and comparator issues before model selection.
