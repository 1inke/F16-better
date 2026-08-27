# EXP-028 α=1.5 Entmax：结果分析

| 模型 | Original ADE/FDE | Fused ADE/FDE | minADE20/minFDE20 | active modes（original/fused） |
|---|---:|---:|---:|---:|
| F10 softmax | 0.5046 / 0.9328 | 0.4567 / 0.8325 | 0.1617 / 0.2478 | 14 / 14 |
| F17 sparsemax | 0.4941 / 0.8901 | 0.4493 / 0.8233 | 0.1542 / 0.2514 | 12 / 11 |
| F19 entmax-1.5 | 0.5422 / 0.9624 | 0.4479 / 0.8087 | 0.1637 / 0.2555 | 17 / 15 |

Entmax-1.5 改善了 F10 的融合指标，并避免 sparsemax 的模式活跃度收缩，但 Original ADE/FDE 和 oracle 指标退化。单独版本不追加多种子；其更温和的稀疏性将与 F16 时间—社会交叉结合测试。
