# EXP-186A Candidate Selection Feature Audit

This experiment freezes the official five-fold EXP-173 seed-1
`F16 + H3 + Joint-V2 + original score + K20` checkpoints. It audits a fixed,
pre-registered bank of 25 candidate-observable features and three zero-parameter
consensus rules. Ground truth is used only to define offline candidate quality
and evaluation labels.

To avoid test-set feature or rule selection, residual features and the primary
rule are selected from each outer model's source/train data. Held-out fold
correlations are confirmatory. R4 uses only source-side residual correlations;
R5 and the all-feature Ridge are supplementary linear upper bounds.

The experiment does not retrain F16, H3, Joint-V2, or any trajectory candidate.
