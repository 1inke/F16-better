# P0-A Protocol Reproduction & Verification Report

> **Audit Status**: **`[STATUS: PASS]`**  
> **Experiment**: `C0_EXP173_EXACT` vs `EXP-173 V2 (f16_temporal_social_cross, eth, seed_1)`  
> **Source Checkpoint**: `C:\Users\linke\Documents\New project\E2_F16_END_TO_END_COMPARISON_SERIES\EXP_173_E2_F16_END_TO_END\results\eth\h3\e3\f16_temporal_social_cross\seed_1\val_scene_-1\best_selected.pth` (SHA256: `8fee24d0bd7b009e...`)  
> **Split Indices Hash**: `1f53e20b1d2462df...` (Match: **PASS**)  
> **Decoder Fingerprint**: `4bec38da2b5907d9` (Match: **PASS**)  
> **Parameter Counts**: Trainable `321,411` / Total `321,988` (Match: **PASS**)

---

## 1. 严格复现指标比对 (C0 vs EXP-173)

| 数据集划分 | 评估指标 (Metric) | EXP-173 (历史记录) | C0 (严格重跑) | 绝对偏差 (Delta) | 允许阈值 (Tol) | 判据 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **VAL** | Selected ADE (m) | `0.5389m` | `0.5389m` | `+0.0000m` | `±0.0010m` | **PASS** |
| **VAL** | Selected FDE (m) | `1.1873m` | `1.1873m` | `+0.0000m` | `±0.0020m` | **PASS** |
| **VAL** | minADE20 (m) | `0.2196m` | `0.2196m` | `+0.0000m` | `±0.0010m` | **PASS** |
| **VAL** | minFDE20 (m) | `0.3960m` | `0.3960m` | `+0.0000m` | `±0.0020m` | **PASS** |
| **VAL** | Selection Regret (m) | `0.3193m` | `0.3193m` | `+0.0000m` | `±0.0020m` | **PASS** |
| **TEST** | Selected ADE (m) | `0.9670m` | `0.9670m` | `+0.0000m` | `±0.0010m` | **PASS** |
| **TEST** | Selected FDE (m) | `1.9024m` | `1.9024m` | `+0.0000m` | `±0.0020m` | **PASS** |
| **TEST** | minADE20 (m) | `0.4086m` | `0.4086m` | `+0.0000m` | `±0.0010m` | **PASS** |
| **TEST** | minFDE20 (m) | `0.5742m` | `0.5742m` | `+0.0000m` | `±0.0020m` | **PASS** |
| **TEST** | Selection Regret (m) | `0.5584m` | `0.5584m` | `+0.0000m` | `±0.0020m` | **PASS** |

---

## 2. 协议锁定状态 (Protocol Identity Lock)

- **Trainer**: `FRONT_ENCODER_SERIES/train_front_variant.py` (无任何额外 trainer 重写)
- **Decoder Module**: `E2_F16_END_TO_END_COMPARISON_SERIES/joint_v2_adapter.py`
- **Optimizer & LR**: `AdamW`, `LR=0.00025` 统一单学习率（无分层 LR）
- **Gradient Clipping**: `5.0`
- **Checkpoint Criterion**: `oracle_combo`
- **EMA Decay**: `0.995`
- **Augmentation**: Reflection (`p=0.5`), Speed min/max `[0.9, 1.1]`
- **Oracle Coverage**: Weight `0.20`, Temp `0.10`, FDE weight `0.25`

---

## 3. P0-A 闸门判定结论 (Verdict)

- **Selected ADE 偏差**: Val `0.00mm` (限额 1mm), Test `0.00mm` (限额 1mm)
- **minADE20 偏差**: Val `0.00mm` (限额 1mm), Test `0.00mm` (限额 1mm)
- **minFDE20 偏差**: Val `0.00mm` (限额 2mm), Test `0.00mm` (限额 2mm)
- **结论**: **P0-A Strict Reproduction PASS! 允许进入后续推理与冗余审计。**
