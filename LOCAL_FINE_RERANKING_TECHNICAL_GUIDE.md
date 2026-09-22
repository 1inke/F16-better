# Local Fine Reranking：具体技术实现指导

> Repository: `1inke/trajectory-candidate-selection`  
> Goal: 在 Frozen F16 和现有 offline cache 基础上实现 S6 系列，不重跑主生成器。

---

# 0. 推荐目录

```text
SELECTOR_LOCAL_RERANK_SERIES/
├── common/
│   ├── local_pool.py
│   ├── local_features.py
│   ├── local_metrics.py
│   ├── hard_weighting.py
│   └── loss_functions.py
│
├── S6F_LOCAL_ORACLE/
│   ├── run_s6f_local_oracle_bound.py
│   └── results/
│
├── S6A_LOCAL_K_SWEEP/
│   ├── run_s6a_local_k_sweep.py
│   └── results/
│
├── S6B_HARDCASE_TRAINING/
│   ├── run_s6b_hardcase_training.py
│   └── results/
│
├── S6C_NEAR_ORACLE_LOSS/
│   ├── run_s6c_near_oracle_loss.py
│   └── results/
│
├── S6D_GATE_LOCAL/
│   ├── run_s6d_gate_local.py
│   └── results/
│
├── S6E_ADAPTIVE_K/
│   ├── run_s6e_adaptive_k.py
│   └── results/
│
└── run_local_rerank_series.py
```

---

# 1. 数据来源

优先复用：

```text
SELECTOR_BENCHMARK_SERIES/features
```

必须能读取：

```text
g0
g1
g2
g3
g4
ade
fde
native_top1_idx
```

不要重新跑 F16。

---

# 2. Local Pool Builder

文件：

```text
common/local_pool.py
```

接口：

```python
def build_native_topk_pool(data, k):
    native_order = np.argsort(
        -data["g0"][:, :, 0],
        axis=-1
    )

    topk_idx = native_order[:, :k]

    return {
        "indices": topk_idx,
        "logits": np.take_along_axis(
            data["g0"][:, :, 0],
            topk_idx,
            axis=-1
        ),
        "ade": np.take_along_axis(
            data["ade"],
            topk_idx,
            axis=-1
        ),
        "fde": np.take_along_axis(
            data["fde"],
            topk_idx,
            axis=-1
        ),
    }
```

注意：

```text
ade / fde 只能作为 label/evaluation
```

---

# 3. Local Feature Extraction

文件：

```text
common/local_features.py
```

先 gather：

```text
G0
G1
G2
G4
```

到 Top-K。

新增：

```text
GL Local Features
```

---

# 4. GL 特征

每个 candidate：

```text
local_rank
logit_gap_to_top1
logit_gap_to_mean
prob_gap_to_top1
distance_to_local_medoid
distance_to_local_barycenter
mean_distance_to_other_local_candidates
min_distance_to_other_local_candidate
local_density
```

最终 shape：

```text
[N,K,D]
```

---

# 5. S6F 必须先实现

文件：

```text
S6F_LOCAL_ORACLE/run_s6f_local_oracle_bound.py
```

对：

```text
K=3,5,10
```

计算：

```python
local_oracle_ade = topk_ade.min(axis=-1)
local_oracle_fde = topk_fde.min(axis=-1)
```

输出：

```text
K
LocalOracleADE
LocalOracleFDE
GapToGlobalOracle
ExactOracleRecall
NearOracle02
NearOracle05
NearOracle10
```

同时输出：

```text
micro
macro
per-fold
```

---

# 6. 统一 Local Ranker API

```python
class LocalRanker:
    def fit(self, x, labels, **kwargs):
        ...

    def score(self, x):
        # [N,K], higher = better
        ...
```

实现三种：

```text
LinearResidualLocalRanker
MLPResidualLocalRanker
TinySetAttentionRanker
```

---

# 7. Linear Residual

训练：

```python
ridge.fit(
    train_x.reshape(-1, D),
    -train_ade.reshape(-1)
)
```

最终：

```python
score = (
    native_logit
    + alpha * scale * delta
)
```

其中：

```text
scale = train-only
alpha = nested validation
```

---

# 8. MLP Residual

结构：

```text
D
↓
Linear 64
↓
LayerNorm
↓
ReLU
↓
Linear 64
↓
ReLU
↓
Linear 1
```

最终仍然：

```text
native_logit + residual correction
```

不要直接丢掉 Native prior。

---

# 9. Tiny Set Attention

输入：

```text
[B,K,D]
```

建议：

```text
projection D→64
1 × TransformerEncoderLayer
4 heads
FFN=128
dropout=0
head 64→1
```

由于：

```text
K <= 10
```

计算成本很低。

---

# 10. Nested Fold Protocol

Outer：

```text
4 folds train
1 fold test
```

Inner：

```text
3 folds train
1 fold validation
```

禁止：

```text
直接按 concat 后 80/20 切
```

Inner 选择：

```text
alpha
learning rate
tau
epsilon
gate thresholds
```

---

# 11. S6A 第一轮配置

```text
K:
3 / 5 / 10

Model:
linear_residual
mlp_residual
tiny_set_attention

Feature:
G0+G1+G2+GL
G0+G1+G2+G4+GL
```

总共 18 configs。

---

# 12. Sample Weight

文件：

```text
common/hard_weighting.py
```

```python
def regret_weight(regret):
    w = np.ones_like(regret)

    w[(regret >= 0.10) & (regret < 0.20)] = 2.0
    w[(regret >= 0.20) & (regret < 0.30)] = 4.0
    w[regret >= 0.30] = 6.0

    return w
```

`regret` 只能用于训练权重，不得作为 test input。

---

# 13. Curriculum

推荐：

```text
Epoch 1-5:
all samples

Epoch 6-10:
regret >= 0.10

Epoch 11-15:
regret >= 0.20
```

保留 validation early stopping。

---

# 14. Near-Oracle Mask

```python
def near_oracle_mask(ade, eps):
    min_ade = ade.min(
        dim=-1,
        keepdim=True
    ).values

    return (
        ade <= min_ade + eps
    )
```

---

# 15. Soft Relative-Regret Target

```python
def soft_regret_target(ade, tau):
    rel = (
        ade
        - ade.min(
            dim=-1,
            keepdim=True
        ).values
    )

    return torch.softmax(
        -rel / tau,
        dim=-1
    )
```

---

# 16. Local Pairwise Loss

由于 K 最大为 10：

```text
最多 45 pairs/sample
```

可以枚举所有 pair，不需要随机采样。

只对：

```text
|ADE_i - ADE_j| >= margin
```

的 pair 训练。

建议：

```text
margin = 0.02m / 0.05m
```

---

# 17. Pair Weight

```python
gap = torch.abs(
    ade_i - ade_j
)

weight = torch.clamp(
    gap / 0.10,
    min=0.5,
    max=3.0
)
```

---

# 18. Combined Loss

第一轮推荐：

```python
loss = (
    0.5 * listwise_loss
    + 0.5 * pairwise_loss
)
```

Residual model：

```python
loss += (
    1e-3
    * delta.pow(2).mean()
)
```

---

# 19. S6C Grid

先固定：

```text
K = S6A 最优 K
Model = S6A 最优 model
```

只扫：

```text
epsilon:
0.02 / 0.05 / 0.10

tau:
0.05 / 0.10 / 0.20
```

避免再次形成无边界 grid search。

---

# 20. Gate 直接复用 S4

S6D 不重新发明 Gate。

从 S4 提供：

```text
P(high_regret)
```

按照 probability 排序执行 intervention。

---

# 21. Gate + Local 伪代码

```python
final_idx = native_idx.copy()

hard_idx = select_by_budget(
    gate_prob,
    budget
)

local_scores = reranker.score(
    local_x[hard_idx]
)

local_choice = np.argmax(
    local_scores,
    axis=-1
)

final_idx[hard_idx] = topk_idx[
    hard_idx,
    local_choice
]
```

---

# 22. Intervention Budget

扫描：

```text
5%
10%
20%
30%
50%
100%
```

不要只用 probability 0.5。

---

# 23. Adaptive-K

```python
def choose_k(p, t1, t2):
    if p < t1:
        return 3
    elif p < t2:
        return 5
    return 10
```

Inner grid：

```text
t1 ∈ {0.2,0.3,0.4}
t2 ∈ {0.6,0.7,0.8}
t1 < t2
```

---

# 24. Adaptive-K 目标

最终报告必须同时给：

```text
ADE
Average K
Latency
```

不要只用合成 objective。

如需 inner validation objective：

$$
J = ADE + \lambda_c \frac{\overline K}{10}
$$

可以从：

```text
lambda_c = 0.005
```

起步。

---

# 25. 统一 Evaluator

至少输出：

```text
Selected ADE
Selected FDE
ADE Regret
FDE Regret
Oracle Hit
NearOracle@0.02
NearOracle@0.05
NearOracle@0.10
Collision Rate
Useful Intervention
False Intervention
Average K
Latency
Parameter Count
```

---

# 26. Local Ranking Metrics

额外：

```text
Local Pairwise@0.05
Local Pairwise@0.10
Local Spearman
Local NDCG@K
```

---

# 27. Prediction 文件标准

每样本：

```text
sample_id
fold
native_idx
selected_idx
local_k
gate_prob
intervened
native_ade
selected_ade
oracle_ade
native_regret
selected_regret
```

其中 privileged 字段仅用于离线分析。

---

# 28. 结果目录标准

每个实验：

```text
config.json
metrics.json
per_fold.csv
predictions.csv
run.log
RESULT_ANALYSIS.md
```

神经模型：

```text
best_model.pt
training_curve.csv
```

---

# 29. Orchestrator

新增：

```text
run_local_rerank_series.py
```

stage：

```text
stage0:
S1A-v3
S5-v2
S6F

stage1:
S6A

stage2:
S6B
S6C

stage3:
S6D
S6E
```

---

# 30. Resume 规则

```text
success → skip
failed → log and continue
--force → rerun
```

不要无限 retry。

---

# 31. 第一轮自动筛选

从 S6A 18 configs 保留：

```text
Top-2 lowest ADE
Top-1 highest NearOracle@5cm
Top-1 lowest latency
```

最多 4 个。

---

# 32. 第二轮

只对 finalists：

```text
Hard weighting
Curriculum
Near-oracle loss
```

---

# 33. 第三轮

只对最终 1~2 个：

```text
Gate integration
Adaptive-K
Seeds 1,2,3
```

---

# 34. S2-v2 触发条件

如果最终 S6：

```text
ADE > 0.500
```

再进入：

```text
S2-v2
```

重点寻找：

```text
context-fused mode hidden
pre-logit hidden
```

如果：

```text
ADE <= 0.495
```

则先继续 Local Reranking 主线。

---

# 35. Agent 第一批任务

建议只给 Agent：

```text
TASK 1
创建 SELECTOR_LOCAL_RERANK_SERIES

TASK 2
实现 local_pool.py

TASK 3
实现 local_features.py

TASK 4
实现 S6F Local Oracle Bound

TASK 5
实现 S6A K=3/5/10

TASK 6
实现 Linear / MLP / Tiny Set Attention

TASK 7
统一 micro/macro/per-fold metrics

TASK 8
输出 S6F + S6A 后暂停
```

不要一口气实现 S6B~S6E。

---

# 36. 第一批最低交付

```text
S6F_LOCAL_ORACLE/results/
    local_oracle_bound.csv
    per_fold.csv
    RESULT_ANALYSIS_S6F.md

S6A_LOCAL_K_SWEEP/results/
    leaderboard.csv
    per_fold.csv
    RESULT_ANALYSIS_S6A.md
```

---

# 37. 最重要的决策点

跑完 S6F + S6A 后比较：

```text
Local Oracle ADE
vs
Actual Local Reranker ADE
```

如果二者差距很大：

```text
local discrimination 仍有明显可学习空间
→ 继续 S6B/S6C
```

如果 Local Oracle 本身已经很差：

```text
候选截断是瓶颈
→ 提高 K / Adaptive-K
```

如果 Local Oracle 很好但实际模型始终约 0.51：

```text
当前 deployable local features 不足
→ 转 S2-v2 latent
```

---

# 38. 工程边界

整个 S6 必须保持：

```text
Frozen F16
Offline selector
No future-GT input
No joint retraining
Low extra compute
```

如果以后需要重新训练 F16，应新开 joint-training 系列，而不是继续把它归入 S6。
