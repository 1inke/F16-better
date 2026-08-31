# EXP-162 最终五折分析

## 最终结论

```text
EXP-162A Verifier-only = REJECTED
EXP-162B Original+Verifier Fusion = REJECTED
```

本实验是完整的nested scene-held-out验证：每个outer fold在另外四个场景上进行inner leave-one-scene-out，只用inner结果选择min_scale、alpha与final epoch；outer test在设置锁定和final verifier训练完成后才首次读取。60个inner训练、5个final verifier和5个outer test均已完成。

## 1. Inner选择结果

| Outer fold | min_scale | alpha | final epochs |
|---|---:|---:|---:|
| ETH | 0.03 | 0.0 | 18 |
| Hotel | 0.03 | 0.0 | 26 |
| Univ | 0.03 | 0.2 | 24 |
| Zara1 | 0.03 | 0.0 | 21 |
| Zara2 | 0.03 | 0.0 | 18 |

所有fold均选择min_scale=0.03，说明增大最小方差不能解决排序问题。四个fold选择alpha=0，意味着inner场景验证已经主动拒绝verifier信号。

## 2. Verifier-only

Verifier-only宏平均Top-1 ADE/FDE从0.5160/1.0648恶化到0.6211/1.2828，分别退化20.37%/20.47%；mean regret退化34.84%，p95 regret退化36.75%，五折全部退化。

最关键的行为是：hit@1从0.1546提高到0.2645，但hit@3、hit@5、hit@10和Spearman均下降，mean/p95 regret显著增加。评分分布entropy从2.6310下降到0.3877，表明verifier对少数高密度轨迹过度自信。生成式“总体合理性”不等于具体样本的正确未来分支。

## 3. Fusion

Fusion宏平均Top-1 ADE/FDE为0.5163/1.0657，相对Original分别退化0.070%/0.085%，mean regret退化0.128%。四折alpha=0使结果与Original完全相同；唯一alpha=0.2的Univ仍然小幅退化。Spearman的小幅改善没有转化为Top-1收益。

因此fusion不是“接近成功”，而是nested CV依靠alpha=0安全回退避免了更严重的失败。

## 4. 对研究主线的含义

EXP-161已经证明candidate-supervised listwise scorer不能跨场景泛化；EXP-162进一步证明，不使用candidate winner标签的GT-only conditional likelihood也不能可靠完成Top-1选择。两种实质不同的监督范式均失败后，继续开发更大的MLP、Transformer reranker、pairwise scorer或新verifier缺乏证据支持。

当前正式决策：

1. 停止Top-1 scorer/verifier主线；
2. 主模型恢复并保留V2原始score head；
3. 将Top-1作为辅助指标和局限性，不再作为近期模型开发目标；
4. 模型定位为lightweight multimodal set predictor；
5. 如果继续提升minADE/minFDE，进入EXP-163 Explicit Endpoint Prototype K20，从候选生成机制本身解决跨fold覆盖不稳定。

EXP-163不应加入新score head、M2、scene branch或attention修改。第一阶段只比较matched K20、当前V2和5×4显式原型K20的五折seed1 Oracle质量。
