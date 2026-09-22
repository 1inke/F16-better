# Local Fine Reranking：新系列实验设计

> Repository: `1inke/trajectory-candidate-selection`  
> Target: Frozen F16 on ETH/UCY 5-fold Leave-One-Out  
> New research focus: **Global ranking is already strong; the bottleneck is fine discrimination among the highest-ranked candidates.**

---

# 0. 当前已确认的事实

Validity Audit 修正后：

- 全局 Pairwise Accuracy 约 `84%~85%`
- 全局 Spearman 约 `0.75`
- Native Top-5 Pairwise Accuracy 约 `64%`
- Native Top-3 Pairwise Accuracy 约 `60%`
- Exact Oracle Recall:
  - Top-3: `46.53%`
  - Top-5: `60.82%`
  - Top-10: `85.24%`
- Near-Oracle@5cm Recall:
  - Top-5: `64.33%`
  - Top-10: `83.28%`
- Native Selected ADE: 约 `0.5184m`
- 现有 global reranker ceiling: 约 `0.509~0.512m`

因此新的核心问题是：

> **如何在 Native 已经完成的 coarse ranking 基础上，用低成本方法对 Top-K 候选做局部精细判别？**

---

# 1. 总体路线

```text
S1A-v3 Nested Calibration
        ↓
S5-v2 Unified Micro/Macro Audit
        ↓
S6F Local Oracle Upper Bound
        ↓
S6A Local Top-K Reranker
        ↓
S6B Hard-Case Specialized Training
        ↓
S6C Near-Oracle / Top-Heavy Objective
        ↓
S6D Gate + Local Reranker
        ↓
S6E Adaptive-K
        ↓
若仍停留 ~0.50
        ↓
S2-v2 Context-Conditioned Latent
```

---

# 2. EXP-S1A-v3：Nested Calibration Confirmation

## 目的

确认当前 residual selector 中 `alpha` 的选择不是由 inner-validation 不严格导致。

## Outer Protocol

```text
4 folds train
1 fold test
```

## Inner Protocol

在 4 个 train folds 内做 fold-level inner validation：

```text
3 folds train
1 fold validation
```

不要再使用：

```text
concat 后最后 20%
```

## 方法

只跑：

```text
linear_residual_g012
mlp_residual_g0124
transformer_residual_g0124
```

## Alpha Grid

```text
0.05
0.10
0.20
0.30
0.50
0.75
1.00
1.50
```

## 判定

如果 S1A-v3 与 v2 的 Test ADE 差异 `< 0.002m`，则当前 residual calibration 结论基本稳定。

---

# 3. EXP-S5-v2：统一统计口径

同时输出：

```text
micro
macro
per-fold
```

需要统一：

- Exact Oracle Recall@K
- Near-Oracle Recall@K
- Pairwise@5cm / @10cm
- Spearman
- NDCG

避免 Exact Oracle 用 micro、Near-Oracle 用 macro 后直接横向比较。

---

# 4. EXP-S6F：Local Oracle Upper Bound

这是 S6 正式训练前必须先做的实验。

对 Native：

```text
Top-3
Top-5
Top-10
```

分别计算：

$$
ADE^{oracle}_K
=
\min_{k\in TopK} ADE_k
$$

以及：

$$
FDE^{oracle}_K
=
\min_{k\in TopK} FDE_k
$$

输出：

```text
LocalOracleADE
LocalOracleFDE
GapToGlobalOracle
ExactOracleRecall
NearOracleRecall@2cm
NearOracleRecall@5cm
NearOracleRecall@10cm
```

意义：

> 先确认 Top-K 子池本身还剩多少可利用上限，再决定是否值得训练 reranker。

---

# 5. EXP-S6A：Local Top-K Reranker

比较：

```text
K = 3
K = 5
K = 10
```

不要提前把 Top-5 写死。

Pipeline：

```text
Native logits
    ↓
Top-K candidates
    ↓
Local Reranker
    ↓
Select one candidate
```

第一轮只跑：

```text
A1 Linear Residual
A2 MLP Residual
A3 Tiny Set Attention
```

---

# 6. Local Feature Set

基础：

```text
G0 native confidence
G1 candidate geometry
G2 candidate-set relative information
G4 predicted-neighbor risk
```

新增局部特征组：

```text
GL = Local Top-K Features
```

包括：

```text
local rank
logit gap to local top1
logit gap to local mean
probability gap to local top1
distance to local medoid
distance to local barycenter
mean local candidate distance
minimum local candidate distance
local density
```

第一轮比较：

```text
G0+G1+G2+GL
G0+G1+G2+G4+GL
```

---

# 7. S6A 成功标准

当前 global reranker 约 `0.509~0.512m`。

因此：

```text
ADE <= 0.505
→ Local paradigm 有明确价值

ADE <= 0.495
→ 值得作为重点方向继续

ADE <= 0.480
→ 可以考虑成为正式主模块
```

不能只看 ADE，同时报告：

```text
FDE
ADE Regret
Oracle Hit
NearOracle@5cm
Collision
False Intervention
Latency
Parameter Count
```

---

# 8. EXP-S6B：Hard-Case Specialized Training

定义：

$$
R_{native}
=
ADE_{native}
-
ADE_{oracle}
$$

样本分层：

```text
Easy:
R < 0.10m

Medium:
0.10 <= R < 0.20m

Hard:
0.20 <= R < 0.30m

Very Hard:
R >= 0.30m
```

比较三种训练方案。

## B0 Baseline

```text
all samples weight = 1
```

## B1 Weighted

```text
Easy      ×1
Medium    ×2
Hard      ×4
Very Hard ×6
```

## B2 Curriculum

```text
Stage 1: all samples
Stage 2: regret >= 0.10m
Stage 3: regret >= 0.20m
```

优先：

```text
B0 → B1 → B2
```

---

# 9. EXP-S6C：Near-Oracle / Top-Heavy Objective

避免强迫模型识别唯一 argmin。

定义：

$$
O_\epsilon
=
\{k: ADE_k\le \min_j ADE_j+\epsilon\}
$$

扫描：

```text
ε = 0.02m
ε = 0.05m
ε = 0.10m
```

重点是：

```text
ε = 0.05m
```

---

# 10. 三种 Loss

## C1 Multi-positive target

所有 Near-Oracle 都作为正样本。

## C2 Soft relative-regret target

$$
q_k=
\frac{
\exp(-(ADE_k-ADE_{min})/\tau)
}{
\sum_j\exp(-(ADE_j-ADE_{min})/\tau)
}
$$

扫描：

```text
τ = 0.02
0.05
0.10
0.20
```

## C3 Top-heavy Pairwise

只在 Native Top-K 中比较。

权重：

$$
w_{ij}=|ADE_i-ADE_j|
$$

推荐首个组合：

$$
L=0.5L_{soft}+0.5L_{pair}
$$

---

# 11. EXP-S6D：Gate + Local Reranker

完整结构：

```text
Native
  ↓
Regret Gate
  ├── Low-risk → keep Native
  │
  └── High-risk
         ↓
      Native Top-K
         ↓
     Local Reranker
```

必须比较：

```text
D0 Native
D1 Global Reranker
D2 Local Top-K Reranker
D3 Gate + Global Reranker
D4 Gate + Local Top-K Reranker
```

否则无法分辨收益来自 Gate 还是 Local Reranking。

---

# 12. Intervention Budget

不要固定 gate probability=0.5。

按照：

```text
5%
10%
20%
30%
50%
100%
```

评估：

```text
Selected ADE
Selected FDE
Intervention Rate
Useful Intervention Rate
False Intervention Rate
Recovered Regret
Collision
```

---

# 13. EXP-S6E：Adaptive-K

固定 K 不一定最优。

根据 Gate score：

$$
K=
\begin{cases}
3,&p<t_1\\
5,&t_1\le p<t_2\\
10,&p\ge t_2
\end{cases}
$$

比较：

```text
Fixed K=3
Fixed K=5
Fixed K=10
Adaptive-K
```

额外统计：

```text
Average K
Average reranker FLOPs
Average latency
```

这一步最符合轻量化主线。

---

# 14. 第一阶段实验矩阵

Seed 1：

```text
K:
3 / 5 / 10

Models:
Linear Residual
MLP Residual
Tiny Set Attention

Feature:
G0+G1+G2+GL
G0+G1+G2+G4+GL
```

总共：

$$
3\times3\times2=18
$$

个配置。

这一规模足够广，但不会重回上一轮 Selector Zoo 的无效搜索。

---

# 15. 第一阶段筛选规则

保留最多 4 个：

```text
2 × lowest ADE
1 × best NearOracle@5cm
1 × lowest latency
```

如果重复则顺延。

第二阶段只对 finalists 测：

```text
hard weighting
curriculum
near-oracle loss
```

第三阶段只对最终 1~2 个配置：

```text
Gate
Adaptive-K
Seeds 1,2,3
```

---

# 16. 当前不建议做的事

暂时不要：

```text
重新训练 F16
继续扩展 20-way Selector Zoo
马上做大型 Transformer probe
把旧 L3 latent 结果继续当最终证据
```

如果 S6 最佳 ADE 仍 `>0.500m`，再回到：

```text
S2-v2 Context-Conditioned L4/L5 Latent
```

---

# 17. 最终研究问题

该系列最终需要回答：

1. Global ranking 强而 Top-1 差的原因是否主要在 Top-K fine discrimination？
2. Top-3/5/10 各自还保留多少 local oracle upper bound？
3. Local reranking 是否真正突破 global selector ceiling？
4. Hard-case specialized training 是否优于全样本同权训练？
5. Near-oracle supervision 是否比 exact argmin 更稳定？
6. Gate 能否降低误干预？
7. Adaptive-K 是否得到更好的 accuracy-efficiency trade-off？
8. 在不修改 F16 的前提下，selection gap 最多能闭合多少？

---

# 18. 当前工作假设

> F16 Native confidence head 已具备较强的 coarse ranking 能力，但高分候选之间的细粒度排序能力明显不足。针对高 regret 风险样本，在 Native Top-K 子池中进行残差式、Near-Oracle-aware 的轻量局部重排，可能比全局 20-way selector 更有效，并能以条件执行和 Adaptive-K 控制额外计算。
