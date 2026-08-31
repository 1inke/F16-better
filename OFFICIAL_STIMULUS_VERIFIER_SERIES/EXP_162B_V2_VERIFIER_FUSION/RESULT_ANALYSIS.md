# EXP-162B V2 Verifier Fusion Result Analysis

## Decision: REJECTED

Nested scene CV在ETH、Hotel、Zara1、Zara2四折均选择alpha=0，仅Univ选择alpha=0.2。融合没有产生任何outer-fold实质改善。

## Macro result

| 指标 | Original V2 | Fusion | 变化 |
|---|---:|---:|---:|
| Top-1 ADE | 0.5160 | 0.5163 | 退化0.070% |
| Top-1 FDE | 1.0648 | 1.0657 | 退化0.085% |
| mean regret | 0.4581 | 0.4587 | 退化0.128% |
| p95 regret | 1.5343 | 1.5360 | 轻微退化 |
| Spearman | 0.7476 | 0.7502 | 小幅提高 |

表面上的4/5折不退化来自四折alpha=0，即完全使用Original V2 score，并不是融合取得了四折收益。唯一采用非零融合的Univ：

- Top-1 ADE：0.5826→0.5844，退化0.31%；
- Top-1 FDE：1.2429→1.2474，退化0.36%；
- mean regret：0.5365→0.5394，退化0.55%；
- p95 regret：0.005级别退化；
- Spearman虽提高，但没有转化为Top-1收益。

宏平均没有达到ADE/FDE改善3%和mean-regret改善10%的预设门槛，因此明确拒绝。

## Conclusion

EXP-162B不通过。原始generator prior与GT-only likelihood在当前实现下没有形成可用互补性。最终继续保留V2原始score head，并正式停止Top-1 scorer/verifier主线。
