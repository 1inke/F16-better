# Cycle1 experiment3: auxiliary causal history encoder

Hypothesis: a recurrent causal representation of observed velocity evolution may support maneuver corrections better than flattened-history processing. Additional32features from8observed increments normalized by observed last-step speed+.03. GRU2->32 vs flatMLP16->70->32; total43736/43742params,6difference(.014%). Decoder has identical seed-specific initialization; encoders cannot be identically initialized. Original observation vector/social features and nativecandidate path/family retained in BOTH arms. No additional input information, only representation/architecture; any benefit is not proof of intention understanding.

Sixfresh35epoch runs flat/gru × seeds1/2/3. Frozen original F16 backbone/scorer, K20 and physical velocitybound1.5, originalbalancedjointwinner/diversity/residual/smoothness/protection.1/fit25912cal2879/constrainedcal remain. Trainable auxiliary encoder is within generator module, not unfreezing F16. No setcontext/Bernsteinbasis/dual/family/top3loss/proposals/pruning/GTforward/test.

Fit-only normalizer buffers reconstruct rawobserved history, no future. Nativeidentity; decoder/encoder GPU gradients; permutation; nearcountmatching/decoderinit/split/hash/config checks. Both encoders exact interrupted/continuous optimizer/RNG resume. 18checkpoints locked before repeatedETHdev;18finiteK20/exactID/hash/recomputedmetrics audit inherited setcontext reporter. Generic zero/mean_std inNPZ/CSV means flat/gru, not removed setcontext. Original5%changeADE/FDE/1%protected/seedCI/coverage/kinematic plus samepathguard vsfreshflat remain.

Estimate12–20min; planning15min; one progressquery9min afterlaunch(60%). Include GRU throughput overhead, recalibrate once fromactualwork notfrequentpolling. Once complete, updatecycleledger3/3 and MUST consolidate threeexperiment outcomes before choosing nextdirection; not autorun4thminor variant.

PowerShell: & "C:\Users\linke\Documents\New project\GENERATION_HISTORY_ENCODER_SERIES\start_experiment.ps1"
