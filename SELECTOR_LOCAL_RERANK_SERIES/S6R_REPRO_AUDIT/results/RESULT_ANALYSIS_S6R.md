# EXP-S6R: S6 Reproducibility Audit Report

> **Dataset**: ETH/UCY Official 5 Folds (34,161 samples)  
> **Evaluation Protocol**: Deterministic Stable-Seeded Leave-One-Fold-Out  
> **Unified Hyperparameters**: `NEURAL_EPOCHS=8`, `BATCH_SIZE=256`, `LEARNING_RATE=0.001`, `WEIGHT_DECAY=0.0001`  
> **Overall Status**: **`PASS`**

---

## 1. 独立运行复现对比表 (Run 1 vs Run 2 Audit)

| Configuration | Model Architecture | Run 1 ADE (m) | Run 2 ADE (m) | $\Delta$ ADE (mm) | Tolerance | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`K3_linear_residual_g0124_gl`** | linear_residual | 0.51038 | 0.51038 | **0.000 mm** | < 0.1 mm | **`PASS`** |
| **`K5_tiny_set_attention_g0124_gl`** | tiny_set_attention | 0.51324 | 0.51324 | **0.000 mm** | < 0.5 mm | **`PASS`** |
| **`K10_tiny_set_attention_g0124_gl`** | tiny_set_attention | 0.51274 | 0.51274 | **0.000 mm** | < 0.5 mm | **`PASS`** |

---

## 2. 审计结论与工程保障

1. **确定性种子控制 (Deterministic Seeding)**：
   通过 `common/reproducibility.py` 的 `stable_seed(base_seed, cfg_id, fold, phase)` 与 `set_deterministic_seed`，彻底消除了跨脚本 RNG 状态漂移问题；
2. **超参数统一 (Unified Training Config)**：
   统一了 `NEURAL_EPOCHS=8` 与 `train_neural_local_ranker` 训练函数，所有模型的训练步数、权重衰减和批次完全一致；
3. **判定依据**：
   - 线性模型两次运行误差绝对值严格为 0.000 mm；
   - 神经网络模型两次运行误差严格控制在 0.5 mm 门限内；
   - **满足推进到后续阶段的准入标准！**
