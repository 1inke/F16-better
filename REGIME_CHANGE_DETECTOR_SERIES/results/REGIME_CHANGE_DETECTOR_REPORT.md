# Observation-Only Regime Change Detector 验证报告

状态：`H1_H2_PASS_READY_FOR_MINIMAL_INTERVENTION_REVIEW`

D-best 由内部 detector-dev AUROC 选择：`d1_motion_social`；正式 validation 未参与模型或变体选择。

## Detector performance

| Model | Dev AUROC | Val AUROC | 95% CI | PR-AUC | Prevalence | Brier | ECE | Spearman | 95% CI | Gate A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d0_motion_only | 0.7674 | 0.8685 | [0.8395, 0.8944] | 0.6493 | 0.1729 | 0.1237 | 0.1696 | 0.4723 | [0.4310, 0.5123] | PASS |
| d1_motion_social | 0.8218 | 0.8666 | [0.8375, 0.8960] | 0.6269 | 0.1729 | 0.1220 | 0.1428 | 0.4934 | [0.4544, 0.5323] | PASS |

## D-best probability groups

| Group | N | Mean p | Median p | 95% CI |
|---|---:|---:|---:|---:|
| stable | 1067 | 0.2376 | 0.1486 | [0.2232, 0.2520] |
| transition | 226 | 0.4234 | 0.4602 | [0.3858, 0.4593] |
| change | 223 | 0.6891 | 0.8046 | [0.6534, 0.7238] |

## Failure diagnosis

Gate B: `PASS`

- p(change) vs ade_regret: Spearman=0.2233, 95% CI [0.1782, 0.2677].
- p(change) vs fde_regret: Spearman=0.2331, 95% CI [0.1871, 0.2792].
- p(change) vs gt_best_candidate_rank: Spearman=0.1412, 95% CI [0.0957, 0.1865].

## Boundaries

- 输入仅包含 observed history 与冻结 F16 observation-only social representation。
- future 只用于训练标签和离线评估；没有 test 访问。
- 当前 baseline 没有 scene/map encoding，因此 D2 没有伪造执行。
- Stage C 未启动；即使 H1/H2 通过，也需要人工审阅后另立实验。
- bootstrap 只度量固定 seed 下样本不确定性，不代表 detector 训练种子不确定性。
