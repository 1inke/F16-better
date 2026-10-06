# RGDM 生成侧 × 选择侧 2×2 验证报告

状态：`STOP_OR_REDESIGN_AFTER_SEED1`

本报告仅使用锁定的 ETH seed-1 scene-tail validation；没有访问 test。样本 bootstrap 只度量固定模型下的配对样本不确定性，不代表训练种子不确定性。

## 全体验证指标

### best_selection

| 条件 | Selected ADE | Selected FDE | minADE | minFDE | J | Regret | R@1 | R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c00_g0_s0 | 0.538869 | 1.187295 | 0.219566 | 0.396031 | 0.417581 | 0.319303 | 20.18% | 36.08% |
| s_only_g0_s1 | 0.536118 | 1.179417 | 0.219566 | 0.396031 | 0.417581 | 0.316553 | 20.25% | 36.81% |
| g_only_g1_s0 | 0.536677 | 1.182165 | 0.219376 | 0.395602 | 0.417177 | 0.317300 | 19.99% | 36.68% |
| joint_g1_s1 | 0.535925 | 1.179226 | 0.219565 | 0.396031 | 0.417581 | 0.316360 | 20.18% | 36.74% |

### best_generation

| 条件 | Selected ADE | Selected FDE | minADE | minFDE | J | Regret | R@1 | R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c00_g0_s0 | 0.538869 | 1.187295 | 0.219566 | 0.396031 | 0.417581 | 0.319303 | 20.18% | 36.08% |
| s_only_g0_s1 | 0.539283 | 1.187472 | 0.219566 | 0.396031 | 0.417581 | 0.319718 | 20.12% | 36.35% |
| g_only_g1_s0 | 0.536998 | 1.182153 | 0.219399 | 0.395175 | 0.416987 | 0.317599 | 19.99% | 36.54% |
| joint_g1_s1 | 0.547814 | 1.199442 | 0.219558 | 0.395326 | 0.417221 | 0.328256 | 20.84% | 37.53% |

### final

| 条件 | Selected ADE | Selected FDE | minADE | minFDE | J | Regret | R@1 | R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c00_g0_s0 | 0.538869 | 1.187295 | 0.219566 | 0.396031 | 0.417581 | 0.319303 | 20.18% | 36.08% |
| s_only_g0_s1 | 0.549658 | 1.204336 | 0.219566 | 0.396031 | 0.417581 | 0.330092 | 20.91% | 37.27% |
| g_only_g1_s0 | 0.536677 | 1.182165 | 0.219376 | 0.395602 | 0.417177 | 0.317300 | 19.99% | 36.68% |
| joint_g1_s1 | 0.547814 | 1.199442 | 0.219558 | 0.395326 | 0.417221 | 0.328256 | 20.84% | 37.53% |

## 判定门

- selection_best_gain_ge_1pct: FAIL
- selection_final_same_direction: FAIL
- selection_fde_guard: PASS
- generation_best_j_gain_ge_2pct: FAIL
- generation_final_same_direction: PASS
- generation_selected_ade_guard: PASS
- joint_selection_gain: PASS
- joint_generation_gain: PASS
- joint_final_not_worse: FAIL

## 解释边界

- `s_only_g0_s1` 必须与 C00 候选严格一致；脚本在分析前执行逐样本 identity gate。
- `best_selection`、`best_generation` 与 `final` 三个视角必须共同审阅，不能只摘取有利 checkpoint。
- motion-change 分组使用未来真值仅做验证后分析；它不进入模型推理输入。
- 只有通过预注册门后才讨论 seed 2/3；脚本不会自动扩展到更多种子、折或 test。

完整的配对区间见 `factorial_effects.csv`。
