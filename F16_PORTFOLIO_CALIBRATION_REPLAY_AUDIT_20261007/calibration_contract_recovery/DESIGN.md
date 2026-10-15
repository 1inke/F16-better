# Frozen calibration contract recovery (2026-10-07)

This child implements the same registered frozen A/B/C audit. No optimizer, parameter update, new loss or data. Preserve parent sources/results/review unchanged.

Parent saved-only review passes arithmetic, but source review identified its classifier helper returning logits while the original M1/M3 explicitly applies softmax. Static candidate class ordering is invariant, so equal-K portfolios remain valid. Parent probability/calibration/interface numbers are NOT faithful contract replays and must not be used for conclusions. This is our audit implementation error, not a user model-training failure.

Only repair: IntentClassifier inference applies the original torch.softmax on CUDA per batch, checks seven probabilities finite, in [0,1], sum approximately one. Other helpers, A/B/C statistics definitions, models, IDs, thresholds, metrics and resource limits unchanged. Freeze all existing weights/buffers and preserve parent and upstream relevant SHA. Full relevant SHA is not all historical old files.

Actual first128 GPU smoke plus independent numeric softmax fixture required before sole zero-trigger worker. Nine full jobs, saved-only arithmetic review. Recover B original M1 ECE/NLL/Brier within numerical tolerance; do not call C a newly trained repaired calibrator. Compare same-budget portfolios and same-retained controls. Failures retain source evidence; no automatic repeated restarts.

After recovery completion, decide next direction from the faithful results. Existing quality utility is not future occurrence probability; original major and five-motion 1% constraints remain. Avoid expanding scorer/new quality assignment/data permissions or declaring intrinsic unpredictability/safety.
