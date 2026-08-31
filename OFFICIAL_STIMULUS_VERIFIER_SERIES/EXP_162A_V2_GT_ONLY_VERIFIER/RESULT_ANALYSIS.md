# EXP-162A V2 GT-Only Verifier Result Analysis

## Decision: REJECTED

Ground-truth-only conditional likelihood verifier在五个outer fold上全部退化，不能独立承担V2候选排序。

## Macro result

| 指标 | Original V2 | Verifier-only | 变化 |
|---|---:|---:|---:|
| Top-1 ADE | 0.5160 | 0.6211 | 退化20.37% |
| Top-1 FDE | 1.0648 | 1.2828 | 退化20.47% |
| mean regret | 0.4581 | 0.6177 | 退化34.84% |
| p95 regret | 1.5343 | 2.0982 | 退化36.75% |
| Spearman | 0.7476 | 0.5471 | 明显下降 |
| score entropy | 2.6310 | 0.3877 | 下降85.26% |

ADE/FDE不退化折数均为0/5；最坏折ADE/FDE/p95-regret分别退化26.17%/27.04%/51.22%。

Verifier的hit@1从0.1546提高到0.2645，但hit@3、hit@5、hit@10均下降，同时mean regret大幅增加。这说明条件似然更容易把概率集中到少数高密度、平滑的常见轨迹；它在部分容易样本上更频繁命中，但错误时会把真正合理的候选排到更后面，形成严重尾部风险。

五折Oracle minADE/minFDE/same-path FDE在所有评分方式下严格保持0.2225/0.3649/0.4339，候选几何没有变化，失败完全来自评分。

## Conclusion

EXP-162A不通过。GT-only生成式似然不能稳定转换为个体候选质量排序，禁止继续扩大verifier网络或进行新的verifier-only超参数搜索。
