# P0-C: Loss & Gradient Equivalence Audit Report

> **Audit Status**: **`[STATUS: PASS]`**  
> **Device**: `cpu`  
> **Target Baseline**: `FRONT_ENCODER_SERIES.locked_mode_query_loss.ModeQueryLoss`  
> **Candidate Loss**: `QUALITY_AWARE_F16_SERIES_V2.confidence_objective_loss.ConfidenceObjectiveModeQueryLoss(objective_mode='original_exact')`  
> **Evaluation Tolerance**: `< 1.0e-08`

---

## 1. 核心结论 (Summary)

Candidate Loss 在 `original_exact` 模式下与原 Native `ModeQueryLoss` 达成严格的逐项数学等价与全图梯度对齐：
- **全局最大 Loss 绝对偏差**: `0.00e+00` (限额 `< 1.0e-08`) $\checkmark$
- **Logit 梯度全局最大绝对偏差**: `0.00e+00` (限额 `< 1.0e-08`) $\checkmark$
- **Prediction 梯度全局最大绝对偏差**: `0.00e+00` (限额 `< 1.0e-08`) $\checkmark$
- **Winner 索引一致性**: **100% 绝对一致** (全测试用例匹配) $\checkmark$

---

## 2. 详细用例对比表 (Test Cases)

| 用例 ID | 形状规格 | 样本加权 | 总 Loss 偏差 | 回归偏差 | 分类偏差 | 端点偏差 | 多样性偏差 | 短步偏差 | Winner 一致 | Logit 梯偏 | Pred 梯偏 | 判定 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `B=4, N=1, K=20, T=12` | `False` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | **100%** | `0.00e+00` | `0.00e+00` | **PASS** |
| 2 | `B=16, N=1, K=20, T=12` | `False` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | **100%** | `0.00e+00` | `0.00e+00` | **PASS** |
| 3 | `B=64, N=1, K=20, T=12` | `False` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | **100%** | `0.00e+00` | `0.00e+00` | **PASS** |
| 4 | `B=8, N=4, K=20, T=12` | `False` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | **100%** | `0.00e+00` | `0.00e+00` | **PASS** |
| 5 | `B=16, N=1, K=20, T=12` | `True` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | **100%** | `0.00e+00` | `0.00e+00` | **PASS** |

---

## 3. 闸门裁决 (Gate Verdict)

- **数学等价性**: 完全成立，无任何常数漂移或浮点溢出。
- **反向传播一致性**: `logits` 与 `predictions` 反向梯度严格重合，证明 `from_environment` 及目标解耦模块完全保留了原版计算图行为。
- **状态流转**: `p0_loss` 标记为 `PASS`，允许进入后续 P0-A/P0-B 验证与 P1 目标冗余审计。
