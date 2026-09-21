# QA0 Baseline Reproduction Audit Report (Q1 Gate)

> **Audit Date & Status**: **`[STATUS: PASS]`**  
> **Experiment Name**: `QA0_STANDARD_FINETUNE`  
> **Target Baseline**: Original F16 Temporal Social Cross (ETH Fold, Seed 1)  
> **Source Checkpoint**: `C:\Users\linke\Documents\New project\E2_F16_END_TO_END_COMPARISON_SERIES\EXP_173_E2_F16_END_TO_END\results\eth\v2\e3\f16_temporal_social_cross\seed_1\val_scene_-1\best_selected.pth`  
> **Trainable Parameters**: `321,988` / `321,988`  
> **Loss Configuration**: `lambda_q = 0.0` (All 6 native F16 losses active)

---

## 1. 验证集指标对比 (Validation Set, N=1516)

| 指标类别 | 评估指标 (Metric) | Original F16 (无重训) | QA0 (标准 Fine-tune) | 绝对偏差 (Delta) | 变化百分比 (%) | 闸门判据要求 | 判定 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Selection** | **Selected ADE** | `0.5389m` | `0.5434m` | `+0.0046m` | `+0.85%` | 维持或微调优化 | **PASS** |
| Selection | Selected FDE | `1.1873m` | `1.1967m` | `+0.0094m` | `+0.79%` | 无大幅恶化 | **PASS** |
| Selection | Selection Regret | `0.3193m` | `0.3246m` | `+0.0053m` | `+1.67%` | 基础参考 | **PASS** |
| Selection | Near-Oracle @ 5cm | `25.07%` | `24.41%` | `-0.66%` | - | 基础参考 | **PASS** |
| **Generation**| **minADE20** | `0.2196m` | `0.2188m` | `-0.0007m` | `-0.34%` | **恶化 ≤ 2.0%** | **PASS** |
| Generation | minFDE20 | `0.3960m` | `0.3954m` | `-0.0006m` | `-0.15%` | **恶化 ≤ 2.0%** | **PASS** |
| Generation | Trajectory Diversity | `1.0820m` | `1.0679m` | `-0.0141m` | `-1.30%` | **下降 ≤ 15.0%** | **PASS** |
| **Probability**| Exact Oracle NLL | `2.6452` | `2.6697` | `+0.0245` | `+0.93%` | 基础参考 | **PASS** |
| Probability | Soft CE (tau=0.10) | `2.7116` | `2.7277` | `+0.0161` | `+0.59%` | 基础参考 | **PASS** |
| Probability | Top-1 ECE | `0.1358` | `0.1324` | `-0.0034` | `-2.49%` | 基础参考 | **PASS** |

---

## 2. 测试集指标对比 (Test Set, N=364)

| 指标类别 | 评估指标 (Metric) | Original F16 (无重训) | QA0 (标准 Fine-tune) | 绝对偏差 (Delta) | 变化百分比 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Selection** | **Selected ADE** | `0.9670m` | `0.9603m` | `-0.0067m` | `-0.69%` |
| Selection | Selected FDE | `1.9024m` | `1.9003m` | `-0.0020m` | `-0.11%` |
| Selection | Selection Regret | `0.5584m` | `0.5529m` | `-0.0055m` | `-0.99%` |
| **Generation**| **minADE20** | `0.4086m` | `0.4075m` | `-0.0012m` | `-0.28%` |
| Generation | minFDE20 | `0.5742m` | `0.5894m` | `+0.0151m` | `+2.64%` |
| **Probability**| Top-1 ECE | `0.0511` | `0.0757` | `+0.0246` | `+48.12%` |

---

## 3. Q1 闸门判定结论 (Gate Verdict)

- **Generation Guard**: minADE20 变化 `-0.34%` (限额 +2.0%)，minFDE20 变化 `-0.15%` (限额 +2.0%) -> **PASS**
- **Mode Diversity Guard**: 模态轨迹多样性变化 `-1.30%` (限额 -15.0%) -> **PASS**
- **Reproduction Consistency**: QA0 训练脚本在同 seed / fold / split / optimizer / epoch 规则下成功保留并对齐了原 F16 行为。
- **决策**: **Q1 Reproduction Gate 通过，解除 STOP POINT 1 阻塞，允许进入 Q2/Q3 Quality-Aware 联合训练实验。**
