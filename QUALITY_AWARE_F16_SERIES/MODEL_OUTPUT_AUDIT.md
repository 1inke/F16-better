# F16 Model Output & Logit Alignment Audit (Q0)

> **Execution Time**: Auto-generated during Q0 Audit  
> **Model Checkpoint**: F16 Temporal Social Cross (ETH Seed 1 Stage V2)  
> **Input Shape**: Past `[B=4, N=9, 8, 2]`, Future `[B=4, N=9, 12, 2]`

---

## 1. 形状与张量契约验证

| 张量名称 (Tensor Name) | 预期形状 (Expected) | 实测形状 (Observed) | 数据类型 (Dtype) | 检验结果 (Status) |
| :--- | :---: | :---: | :---: | :---: |
| **`pred` (Trajectories)** | `[B, N, 20, 12, 2]` | `[4, 9, 20, 12, 2]` | `torch.float32` | **PASS** |
| **`logits` (Mode Scores)**| `[B, N, 20]` | `[4, 9, 20]` | `torch.float32` | **PASS** |
| **`focal_pred`** | `[B, 20, 12, 2]` | `[4, 20, 12, 2]` | `torch.float32` | **PASS** |
| **`focal_logits`** | `[B, 20]` | `[4, 20]` | `torch.float32` | **PASS** |

所有断言严格通过：
- `pred.shape[-3] == 20` $\checkmark$
- `logits.shape[-1] == 20` $\checkmark$
- `pred.shape[0] == logits.shape[0]` $\checkmark$
- `pred.shape[1] == logits.shape[1]` $\checkmark$

---

## 2. 模式与评分对应关系核验 (Candidate / Logit Ordering)

- **前向机制确认**：在 `ControlledAnchorDecoder` 及其子类 `JointTrainableV2Decoder` 中，`last_mode_logits` 是由 `score_head(latent)` 生成的 $K=20$ 维张量，输入隐层与生成 20 条候选位移的 query mode 具有严格的逐项一一对应关系（模式索引 $k \in [0, 19]$ 严格对齐）。
- **对齐采样存证**：已保存首个批次全部样本共 80 条模式对应记录至 `results/output_alignment_debug.csv`。
- **结论**：未发现任何置换 (permutation) 或候选错位现象，**Q0 Output Audit 状态：PASS**，满足进入 Q1 Baseline Reproduction 的全部先决条件。
