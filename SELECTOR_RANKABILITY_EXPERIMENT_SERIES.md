# F16 Candidate Selection：第二阶段 Rankability 系列实验路线

## 0. 背景与当前判断

上一轮 Offline Selector Zoo 已经完成了较大范围的 selector 搜索，但结果显示：

- F16 Native Top-1 ADE 约为 `0.5184m`
- Oracle minADE20 约为 `0.2181m`
- 最好的 accuracy selector `linear_residual_g012` 仅将 ADE 降到约 `0.5098m`
- safety 方向稍好，`native_plus_mpn_risk_grid` 将碰撞率从约 `6.68%` 降到约 `5.71%`
- 说明现有 G0~G4 deployable information 中存在可利用信号，但无法解释绝大多数 oracle gap

因此，下一阶段不再继续扩充 Selector Zoo，而是将问题重新定义为：

> **Oracle 候选虽然存在，但它是否能从部署时可观测信息中被识别？**

新的核心研究对象不再是 selector architecture 本身，而是：

```text
Rankability
Information sufficiency
Representation adequacy
Generalization
Conditional intervention
```

---

# 1. 总体实验路线

建议按以下顺序推进：

```text
EXP-S1 Rankability Audit
        ↓
        ├── train 也学不会
        │       ↓
        │   Representation / Information bottleneck
        │       ↓
        │   EXP-S2 Latent Probe
        │
        └── train 能学会、test 不行
                ↓
            Generalization bottleneck
                ↓
            fold / scene adaptation

EXP-S4 Accuracy Regret Gate
        ↓
只在 Native 高 regret 样本上 rerank

EXP-S3 Large Capacity Probe
        ↓
判断“信息存在但轻量模型能力不足”
还是“deployable information 本身不足”
```

推荐执行顺序：

```text
S1 → S2 → S4 → S3
```

原因：

1. S1 成本最低，但能直接判断问题类型
2. S2 能测试 F16 内部 latent 是否保留了 mode-level intention
3. S4 能绕开“每个样本都重新选第一”的难题
4. S3 用大模型作为诊断上界，不用于最终部署

---

# 2. EXP-S1：Rankability Audit

## 2.1 研究问题

回答：

> 当前 G0~G4 特征下，oracle ranking 是“可学习但泛化差”，还是“训练集都学不会”？

这是下一阶段最关键的实验。

## 2.2 方法

固定少量代表性模型，不再广撒网：

```text
linear_residual_g012
mlp_pairwise_g0124
mlp_listwise_g0124
tiny_transformer_listwise_g0124
```

所有模型都同时记录：

```text
Train metrics
Validation / held-out fold metrics
```

禁止只记录 test。

## 2.3 重点指标

### Top-1 accuracy metrics

```text
Selected ADE@1
Selected FDE@1
ADE Regret
FDE Regret
Near-Oracle@0.05m
Near-Oracle@0.10m
Oracle Hit@1
```

定义：

$$
Regret_{ADE}=ADE_{selected}-\min_k ADE_k
$$

### Ranking metrics

至少：

```text
Pairwise ranking accuracy
Spearman correlation
Kendall Tau
NDCG@5
NDCG@20
```

## 2.4 关键判定

### Case A：Train 很好，Test 很差

例如：

```text
Train ADE → 0.25~0.30
Held-out ADE → 0.50
```

说明 ranking signal 存在，但跨 scene / fold 泛化失败。

下一步转向：

```text
scene conditioning
domain normalization
fold-independent feature scaling
density / motion regime conditioning
```

### Case B：Train 也很差

例如：

```text
Train ADE 仍 > 0.45
```

说明 G0~G4 对 oracle mode 的辨识能力不足。

直接进入 EXP-S2 Latent Probe。

### Case C：Train 接近 Oracle，Test 有中等提升

说明信息足够、模型可学，但数据量 / regularization / generalization 不足。

---

# 3. EXP-S1B：Oracle Margin / Ambiguity Audit

## 3.1 目的

判断“找出单一最佳轨迹”这个监督目标本身是否过于尖锐。

对每个样本排序：

```text
ADE_(1)
ADE_(2)
ADE_(3)
...
ADE_(20)
```

计算：

$$
\Delta_{1,2}=ADE_{(2)}-ADE_{(1)}
$$

$$
\Delta_{1,5}=ADE_{(5)}-ADE_{(1)}
$$

以及：

$$
\Delta_{native}=ADE_{native}-ADE_{oracle}
$$

## 3.2 输出分布

必须统计：

```text
P(Δ1,2 < 0.01m)
P(Δ1,2 < 0.02m)
P(Δ1,2 < 0.05m)
P(Δ1,2 < 0.10m)

P(native regret < 0.05m)
P(native regret < 0.10m)
P(native regret > 0.20m)
P(native regret > 0.30m)
P(native regret > 0.50m)
```

## 3.3 Near-Oracle Set

定义：

$$
O_\epsilon=\{k \mid ADE_k \le \min ADE + \epsilon \}
$$

推荐：

```text
ε = 0.02m
ε = 0.05m
ε = 0.10m
```

统计：

```text
平均 |Oε|
Native ∈ Oε 的比例
Top-5 Native Ranking 中是否存在 Oε
Selector 选择是否落入 Oε
```

## 3.4 决策意义

如果大量样本 `Δ1,2≈0`，则不应继续用 single oracle index classification，应转向 soft target、near-oracle set classification 或 listwise distribution learning。

---

# 4. EXP-S2：F16 Latent Rankability Probe

## 4.1 研究问题

回答：

> F16 在生成轨迹坐标之前，其内部 latent 是否已经包含 mode-level intention / quality information？

当前 G0~G4 主要基于：

```text
native logit
candidate geometry
history
social context
predicted neighbor risk
```

但这些是生成后的外显信息。

F16 内部可能还保留：

```text
mode query
decoder hidden state
social latent
focal context token
mode-conditioned representation
```

## 4.2 优先抽取层级

```text
L1: focal encoder output
L2: pooled social representation
L3: decoder mode query
L4: mode-specific pre-output hidden
L5: pre-logit hidden / confidence-head input
```

最重要的是 L3 / L4 / L5。

## 4.3 数据形式

每个 sample：

```python
{
    "mode_latent": [20, D],
    "native_logits": [20],
    "candidate_traj": [20, 12, 2],
    ...
}
```

禁止使用 future GT 生成 latent。

---

# 5. EXP-S2B：Latent Probe

先不要用复杂模型，只跑：

```text
Linear probe
2-layer MLP probe
Residual MLP probe
Pairwise latent ranker
```

### Probe 1

```text
latent_k → predicted ADE_k
```

### Probe 2

```text
latent_i, latent_j → which one has lower ADE
```

### Probe 3

```text
native_logit_k + Δscore(latent_k)
```

## 5.1 Feature ablation

至少做：

```text
G0
G0 + G1
G0 + G1 + G4
G0 + L3
G0 + L4
G0 + L5
G0 + G1 + G4 + L4
```

## 5.2 成功判定

如果 latent probe 将 ADE 从约 `0.5098` 明显降到 `<0.48`，说明 latent 有额外价值。

如果能到 `<0.42`，说明 selection bottleneck 很可能主要是 confidence head 没有正确利用 decoder representation。

如果仍在 `0.50` 左右，则 mode latent 本身也没有稳定表达未来真实选择信息。

---

# 6. EXP-S2C：Latent Linear Separability

构造 pairwise classification：

```text
positive pair:
ADE_i + margin < ADE_j
```

margin：

```text
0.05m
0.10m
0.20m
```

输入：

```text
latent_i - latent_j
```

统计 pairwise classification accuracy。

如果 margin 越大越容易分类，说明 latent 至少能判断“明显好/明显坏”的 mode。

---

# 7. EXP-S4：Accuracy Regret Gate

## 7.1 研究问题

当前 selector 试图每个 sample 都重选 Top-1，但大量样本 Native 可能已经足够好。

改为：

```text
先判断 Native 是否值得干预
只有高 regret 样本才 rerank
```

## 7.2 Regret 定义

$$
r=ADE_{native}-\min ADE_{20}
$$

统计：

```text
r < 0.05
r < 0.10
0.10 <= r < 0.20
0.20 <= r < 0.30
r >= 0.30
r >= 0.50
```

---

# 8. EXP-S4B：Bad-Native Gate

构造二分类：

```text
bad_native = regret >= threshold
```

thresholds：

```text
0.10m
0.20m
0.30m
```

模型：

```text
Logistic Regression
Small MLP
GBDT
```

评估：

```text
AUROC
AUPRC
Recall@fixed precision
Precision@fixed recall
```

重点不是分类 accuracy，而是能否高 recall 找出高 regret 样本。

---

# 9. EXP-S4C：Conditional Reranking

流程：

```text
Native
  ↓
Regret Gate
  ├── low risk → keep Native
  └── high risk → rerank Top-K
```

Top-K：

```text
K = 3
K = 5
K = 10
```

reranker：

```text
linear_residual
mlp_pairwise
latent residual
```

指标：

```text
Selected ADE
Intervention Rate
Useful Intervention Rate
False Intervention Rate
Recovered Regret
```

---

# 10. EXP-S4D：Regret Recovery Curve

对不同 intervention budget：

```text
5%
10%
20%
30%
50%
100%
```

记录：

```text
ADE improvement
Recovered regret
False intervention
```

最终绘制：

```text
Intervention Rate vs ADE Improvement
```

这能直接判断是否只有少量 hard cases 值得重新筛选。

---

# 11. EXP-S3：Large Capacity Probe

## 11.1 目的

这是 diagnostic upper-bound probe，不是最终部署模型。

问题：

> 如果给 deployable information 一个明显更强的模型，是否能显著逼近 oracle？

## 11.2 输入

尽量完整：

```text
focal history
all neighbor histories
20 candidate trajectories
20 native logits
20 mode latent
neighbor predicted futures
candidate set context
```

仍然禁止：

```text
future GT as input
GT collision as input
GT ADE/FDE as input
```

## 11.3 架构

```text
Candidate Encoder
    ↓
20 candidate tokens

History Encoder
    ↓
context token

Neighbor Encoder
    ↓
social tokens

Cross Attention
    ↓
Set Transformer / Transformer Encoder
    ↓
20 ranking scores
```

建议规模：

```text
hidden = 128 / 256
layers = 3~6
heads = 4~8
```

## 11.4 判定

### Large Probe ≈ 0.50

说明即使模型容量增加，deployable information 也不足。

### Large Probe < 0.42

说明信息存在，但 lightweight selector capacity / inductive bias 不足。

### Large Probe < 0.35

说明 selection bottleneck 很值得继续作为后续主线，可以进入 distillation / compression。

---

# 12. 新实验系列编号建议

```text
EXP-S1A  Train-Test Rankability Audit
EXP-S1B  Oracle Margin & Near-Oracle Audit
EXP-S1C  Ranking Metric Decomposition

EXP-S2A  Latent Extraction
EXP-S2B  Linear / MLP Latent Probe
EXP-S2C  Pairwise Latent Separability

EXP-S4A  Native Regret Distribution
EXP-S4B  Bad-Native Gate
EXP-S4C  Conditional Accuracy Reranking
EXP-S4D  Intervention Budget Curve

EXP-S3A  Large Capacity Probe
EXP-S3B  Large Probe Ablation
EXP-S3C  Distillation Feasibility
```

---

# 13. 实验停止条件

## 停止继续普通 selector 搜索

如果：

```text
S1 Train rankability 也弱
+
S2 latent probe 也弱
+
S3 large probe 仍 > 0.48~0.50
```

则正式停止 offline selector-only optimization。

结论转向：

```text
oracle support != deployable identifiability
```

## 继续 selector 主线

如果：

```text
S2 / S3 明显 < 0.45
```

则说明 selection information 存在，后续转向 distill / compress / lightweight ranker。

---

# 14. 第一批立即运行

```text
EXP-S1A
EXP-S1B
EXP-S4A
```

它们不需要改 F16 主体。

然后运行：

```text
EXP-S2A
EXP-S2B
```

需要在 F16 forward 中加入 latent hook / cache。

最后根据结果决定是否值得做：

```text
EXP-S3A Large Probe
```

不要一开始直接实现大型 Transformer probe。
