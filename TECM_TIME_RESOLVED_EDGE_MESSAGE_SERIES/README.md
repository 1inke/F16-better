# EXP-152 TECM1 — Time-Resolved Edge-Conditioned Message

TECM1 tests whether relation information matters when it changes **message content** rather than attention routing. It preserves F16's sparse graph and original attention scores, then adds a receiver-specific residual `r(i,j,t) -> 16 -> 64` to every neighbour-time value. The final layer is zero-initialized.

The matched E0 checkpoint, temporal branch, social Q/K, original relation bias/value, fusion and CV-free three-anchor decoder are frozen. Only the adapter is trained (about 1.3k parameters, under 0.5% of the model).

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\TECM_TIME_RESOLVED_EDGE_MESSAGE_SERIES\run_tecm1_batch.py" --device cuda:0 --seeds 1
```

The command runs General and Challenge seed 1. Promotion requires ≥0.5% oracle minADE/minFDE improvement on either split, no meaningful regression on the other, and a follow-up message-activation audit.
