# EXP-153 FCA — Fusion Counterfactual Audit

This is a zero-training audit of the matched frozen F16 checkpoint. At F16's social residual fusion, it evaluates `h_ind + alpha * delta_social` for alpha = 0, 0.25, 0.5, 0.75, 1.0, and 1.25. Alpha 1.0 is the original forward path.

No parameter, sparse graph, candidate decoder, or checkpoint is changed. General and Challenge use their existing matched seed-1 E0 checkpoints. Per-sample output includes its best alpha, motion speed, density and—where index alignment is available—the prior CSNA social-gain label.

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\FUSION_COUNTERFACTUAL_AUDIT_SERIES\run_fca_audit.py" --device cuda:0 --seed 1
```

Output: `EXP_153_FCA/summary.json` and one per-sample CSV per split. The result decides whether a dual-stream late-fusion model is justified; it is not a training experiment.
