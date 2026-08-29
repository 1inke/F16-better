# TNPA正式结果分析

## 1. 结论

TNPA未通过CAT1预注册门槛：

```text
cat1_training_allowed = false
```

观测历史、社会状态、场景和冻结encoder表示中确实存在弱到中等的tail-risk信号，但该信号在motion-challenge分布上不足以可靠决定何时提升score rank 11–20候选。当前不应训练简单的Condition-Aware Tail Prior。

## 2. 最佳Union分类结果

| Split | 最佳配置 | Train正例率 | Val正例率 | AUROC (95% CI) | AUPRC | AUPRC lift | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| General | P6 + Logistic | 21.02% | 20.98% | 0.6931 (0.6560–0.7218) | 0.3782 | 1.803 | 0.2086 |
| Challenge | P6 + Logistic | 20.23% | 35.65% | 0.6263 (0.6053–0.6451) | 0.4681 | 1.313 | 0.1983 |

General接近但没有稳定超过0.70；其置信区间包含明显低于0.70的区域。Challenge AUROC只有0.6263，是CAT1被否决的主要原因。

## 3. 分布偏移是主要困难

General的train/validation tail比例基本一致，约21%。Challenge则从训练的20.23%升至验证的35.65%。

这与TCA结论一致：motion-challenge validation富集了训练折中相对稀少的困难未来。探针虽然能捕捉一部分风险，但无法消除这一condition shift。

Challenge最佳探针在训练正例率匹配阈值下：

- positive recall：67.44%；
- negative recall：50.49%；
- precision：43.01%。

较高的正例召回是以把接近一半负例也判为tail-needed为代价，不能作为精确的候选分数门控。

## 4. 固定风险预算的实际价值

| Split | 风险预算 | 捕获tail正例 | 选中样本精度 |
|---|---:|---:|---:|
| General | top 10% | 20.75% | 43.42% |
| General | top 20% | 36.79% | 38.61% |
| General | top 35% | 56.29% | 33.71% |
| Challenge | top 10% | 14.99% | 53.47% |
| Challenge | top 20% | 28.49% | 50.83% |
| Challenge | top 35% | 46.16% | 47.03% |

Challenge中使用20%的tail-risk预算只能覆盖28.49%的真实tail-needed样本。相较随机20%有所提升，但不足以承担动态候选分配或强score prior。

## 5. 特征贡献

### General Union

| 特征 | Logistic AUROC | MLP AUROC |
|---|---:|---:|
| P0 scene only | 0.6413 | 0.6672 |
| P1 individual | 0.6388 | 0.6669 |
| P2 social | 0.6226 | 0.6340 |
| P3 individual + social | 0.6560 | 0.6877 |
| P4 encoder representation | 0.6704 | 0.6811 |
| P5 individual + social + scene | 0.6679 | 0.6669 |
| P6 all | **0.6931** | 0.6915 |

General的scene-only已经达到0.64–0.67，说明部分可预测性来自场景构成，而不完全来自逐样本运动机制。P6提供额外信息，但提升有限。

### Challenge Union

| 特征 | Logistic AUROC | MLP AUROC |
|---|---:|---:|
| P0 scene only | 0.5420 | 0.5541 |
| P1 individual | 0.5520 | 0.5208 |
| P2 social | 0.5919 | 0.5358 |
| P3 individual + social | 0.5869 | 0.5992 |
| P4 encoder representation | 0.5837 | 0.5783 |
| P5 individual + social + scene | 0.5945 | 0.5858 |
| P6 all | **0.6263** | 0.6136 |

Challenge中social-only优于individual-only，且Logistic下P1→P3带来：

- AUROC +0.0349；
- AUPRC +0.0427。

P3→P5的scene增量只有AUROC +0.0076。因此Challenge困难样本与社会状态确有关系，但这种关系不足以被当前特征稳定分离。

P4冻结encoder表示没有明显超过手工条件特征；P6组合后才达到0.6263，说明各组包含互补但都较弱的信息。

## 6. 非线性probe没有解决问题

Challenge中P6 Logistic为0.6263，MLP为0.6136；其他特征组也没有稳定的MLP优势。因此失败不能简单归因于Logistic表达能力不足。继续增大probe很可能把训练分布拟合得更好，但不能证明跨困难分布泛化。

## 7. ADE与FDE不是本轮失败来源

Challenge最佳结果：

- ADE label：AUROC 0.6157；
- FDE label：AUROC 0.6253；
- Union label：AUROC 0.6263。

ADE和FDE都处于相同的弱可预测区间。当前问题不是选择了错误的tail标签，也不是ADE/FDE冲突导致分类失败，而是可观测条件对困难未来的解释能力有限。

## 8. 连续收益预测只能弱排序，不能估计幅度

| Split | Target | Spearman | Pearson | Ridge MAE | 恒预测0 MAE | MAE变化 |
|---|---|---:|---:|---:|---:|---:|
| General | ADE gain | 0.2075 | 0.2027 | 0.0958 | 0.0605 | +58.31% |
| General | FDE gain | 0.2086 | 0.1997 | 0.2355 | 0.1500 | +56.99% |
| Challenge | ADE gain | 0.1983 | 0.2205 | 0.1681 | 0.1323 | +27.06% |
| Challenge | FDE gain | 0.1907 | 0.2017 | 0.4109 | 0.3200 | +28.39% |

正相关表明模型能做一点风险排序，但gain是大量0和少数大值构成的零膨胀分布，Ridge的绝对误差反而显著差于恒预测0。因此不能根据probe输出可靠决定tail prior强度。

## 9. 研究决策

1. 不训练当前建议中的CAT1简单score prior。
2. 保留K20及现有模式+三锚点解码器。
3. 不继续静态tail family、tail slot替换和确定性tail gate。
4. 将tail-needed作为困难样本/分布偏移的诊断标签，而不是直接控制解码器的监督信号。
5. 后续优先研究输入侧的domain-robust motion/social representation；Challenge结果显示social信息有独立价值，但应增强表示与泛化，而不是直接给tail logits加偏置。

TNPA证明了“存在部分可预测风险”，但没有证明“风险已经可被可靠利用”。这一区分决定了当前应停止CAT1，而不是绕过门槛继续训练。
