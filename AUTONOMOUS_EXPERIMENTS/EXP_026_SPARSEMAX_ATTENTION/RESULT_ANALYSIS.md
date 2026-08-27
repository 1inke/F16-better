# EXP-026 Sparsemax 注意力：阶段分析

## seed=1 结果

| 模型 | 参数量 | Original ADE/FDE | Fused ADE/FDE | minADE20/minFDE20 | active modes（original/fused） |
|---|---:|---:|---:|---:|---:|
| F10+K20 | 284,642 | 0.5046 / 0.9328 | 0.4567 / 0.8325 | 0.1617 / 0.2478 | 14 / 14 |
| F17 Sparsemax | 284,642 | **0.4941 / 0.8901** | **0.4493 / 0.8233** | **0.1542** / 0.2514 | 12 / 11 |

Sparsemax 无新增参数，并同时改善 F10 的 Original ADE/FDE、Fused ADE/FDE 和 minADE20。它说明在已经稀疏的图内，标准 softmax 仍给部分弱边分配了不必要权重，精确零权重具有价值。

但 active modes 从 14 降至 11–12，minFDE20 也没有改善，说明 α=2 的 sparsemax 可能抑制过强。用户笔记中的随机非自回归 Transformer 使用 α=1.5 entmax；后者处在 softmax 与 sparsemax 之间，应作为后续独立实验。

## 阶段决策

- F17 进入多随机种子复验。
- 追加 α=1.5 entmax 版本，检验能否保留稀疏收益并减少模式收缩。
