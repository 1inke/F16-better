# Formal Regime-Adaptive Selection Report

Verdict: `STOP_OR_REDESIGN_REGIME_ADAPTIVE_SELECTION`

五折 Macro；三个 seed 仅属于 detector/reranker，生成器固定为每折 seed-1。

| Comparison | Metric | Macro delta | 95% CI |
|---|---|---:|---:|
| constant_gate | selected_ade | -0.002452 | [-0.003685, -0.001248] |
| constant_gate | selected_fde | -0.000265 | [-0.002803, 0.002203] |
| shuffled_d1 | selected_ade | 0.000228 | [-0.001302, 0.001769] |
| shuffled_d1 | selected_fde | 0.004799 | [0.001705, 0.008051] |
| d0_motion_gate | selected_ade | 0.002122 | [0.000821, 0.003424] |
| d0_motion_gate | selected_fde | 0.010045 | [0.007412, 0.012678] |
| d1_social_gate | selected_ade | 0.002620 | [0.000452, 0.004737] |
| d1_social_gate | selected_fde | 0.011252 | [0.006678, 0.015751] |
| d1_vs_constant | selected_ade | 0.005072 | [0.003069, 0.007181] |
| d1_vs_constant | selected_fde | 0.011517 | [0.007199, 0.015988] |
| d1_vs_shuffled | selected_ade | 0.002392 | [0.000580, 0.004102] |
| d1_vs_shuffled | selected_fde | 0.006453 | [0.002392, 0.010344] |
| d1_vs_d0 | selected_ade | 0.000498 | [-0.001456, 0.002367] |
| d1_vs_d0 | selected_fde | 0.001206 | [-0.002748, 0.005126] |
| d1_vs_native_stable | selected_ade | 0.003739 | [0.000546, 0.007261] |
| d1_vs_native_high | selected_ade | 0.006668 | [-0.000542, 0.014511] |
| d1_vs_native_low | selected_ade | -0.001486 | [-0.002077, -0.000899] |

## Gates

- primary_gain_at_least_1pct: FAIL
- beats_constant_and_shuffled: FAIL
- fde_not_materially_worse: FAIL
- stable_not_materially_harmed: FAIL
- effect_concentrated_high_p: FAIL

Test was evaluated once after every train/dev-selected checkpoint was locked; test labels were not used for selection.
