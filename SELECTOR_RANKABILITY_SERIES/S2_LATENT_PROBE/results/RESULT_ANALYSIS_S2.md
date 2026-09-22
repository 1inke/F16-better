# EXP-S2: F16 Latent Rankability Probe (DEPRECATED)

> **Status**: `[DEPRECATED / ISOLATED PENDING EXP-S2-v2]`  
> **Reason**: The initial exploratory S2 probe evaluated the decoder's raw `mode_mlp` output (L3) with uncorrected pairwise ranking metric directions and fixed scaling. As documented in the Validity Rerun Guide, L3's degree of sample-conditioning versus mode-embedding identity remains under verification (EXP-S2-v2A), and pre-logit/context-fused latents (L4/L5) must be extracted before drawing conclusions about deep latent information sufficiency.

---

## 1. 历史探针数据归档 (Archived Initial Probe Data)

*(Note: Pairwise metric direction was inverted in this early exploratory probe; results are archived for traceability only)*

| Feature Configuration | Selected ADE@1 (m) | Selected FDE@1 (m) | ADE Regret (m) | Oracle Hit | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **G0_native_only** | 0.5184 | 1.0757 | 0.3003 | 17.74% | Archived Baseline |
| **G0_G1_residual** | 0.5089 | 1.0546 | 0.2907 | 17.66% | Archived Geometry |
| **G0_L3_latent_residual** | 0.5148 | 1.0654 | 0.2967 | 19.40% | Archived L3 |
| **G0_G1_L3_residual** | 0.5075 | 1.0502 | 0.2894 | 19.27% | Archived Combo |
| **G0_G1_G4_L3_residual** | 0.5066 | 1.0481 | 0.2885 | 19.26% | Archived Full |

---

## 2. 状态说明与后续规划

1. **早期终局化表述已废止**：早期报告中关于“深层潜变量也证明物理不可选”的推论已经废除。
2. **迁移至 EXP-S2-v2 路线**：
   - `S2-v2A`：对 L3 进行跨样本方差比率审计（Within-mode vs Between-mode variance），确认其是样本特异性意图还是模态先验嵌入；
   - `S2-v2B`：定位并提取上下文融合后的深层隐藏层 `L4`（mode-specific hidden after context fusion）与打分头前置层 `L5`（pre-logit hidden）；
   - `S2-v2C`：采用修正后的成对指标与嵌套校准评估 L4/L5 探针。
