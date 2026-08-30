# EXP-154 Formal Encoder Ablation

This is the required common-protocol encoder evidence chain. It holds the current CV-free 20-mode / three-anchor decoder fixed and compares: trajectory-only (A), sparse graph (B), relation prior (C), F16 full temporal-social interaction (D), and F35 two-segment compression (E). All variants use exactly the same split, seed, loss, augmentation, schedule and checkpoint-selection criterion.

Run Stage A first (General and Challenge, seed 1):

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\FORMAL_ENCODER_ABLATION_SERIES\run_formal_encoder_ablation.py" --device cuda:0 --seeds 1
```

Only variants with a valid Stage-A signal are promoted to seeds 2 and 3.
