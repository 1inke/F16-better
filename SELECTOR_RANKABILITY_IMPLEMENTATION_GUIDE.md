# F16 Rankability 第二阶段：实现指导

## 0. 目标

本文件用于指导本地 Coding Agent 实现新的 Rankability 系列实验。

原则：

```text
不重训 F16
最大化复用已有 Candidate Cache
新增统计优先
新增 latent cache 次之
大型模型最后实现
```

---

# 1. 推荐目录

```text
SELECTOR_RANKABILITY_SERIES/
├── README.md
├── common/
│   ├── cache_io.py
│   ├── metrics.py
│   ├── folds.py
│   ├── feature_schema.py
│   └── utils.py
│
├── S1_RANKABILITY_AUDIT/
│   ├── run_s1a_train_test_rankability.py
│   ├── run_s1b_oracle_margin.py
│   ├── run_s1c_ranking_metrics.py
│   └── results/
│
├── S2_LATENT_PROBE/
│   ├── extract_latent_cache.py
│   ├── latent_hooks.py
│   ├── run_s2b_latent_probe.py
│   ├── run_s2c_pairwise_separability.py
│   └── results/
│
├── S4_REGRET_GATE/
│   ├── run_s4a_regret_distribution.py
│   ├── run_s4b_bad_native_gate.py
│   ├── run_s4c_conditional_rerank.py
│   ├── run_s4d_budget_curve.py
│   └── results/
│
├── S3_LARGE_PROBE/
│   ├── large_probe_model.py
│   ├── run_s3a_large_probe.py
│   ├── run_s3b_ablation.py
│   └── results/
│
└── run_rankability_series.py
```

---

# 2. 优先复用上一轮缓存

Agent 首先检查：

```text
SELECTOR_BENCHMARK_SERIES/cache
SELECTOR_BENCHMARK_SERIES/features
```

是否包含：

```text
sample_id
fold
seed
candidate trajectories
native logits
candidate ADE
candidate FDE
history
neighbor history
MPN risk feature
```

如果有，S1 / S4 都不要重新跑 F16。

---

# 3. Cache Adapter

新增：

```text
common/cache_io.py
```

统一输出概念结构：

```python
class SelectorSample:
    sample_id
    fold
    seed

    past
    num_valid

    candidates
    native_logits

    candidate_ade
    candidate_fde

    feature_g0
    feature_g1
    feature_g2
    feature_g3
    feature_g4

    mode_latent = None
```

要求：

```text
旧 cache schema 不一致时在 adapter 内兼容
不要修改历史实验文件
```

---

# 4. EXP-S1A 实现

## 4.1 输入

第一轮：

```text
Seed 1
5 folds
```

代表模型：

```text
linear_residual_g012
mlp_pairwise_g0124
mlp_listwise_g0124
tiny_transformer_listwise_g0124
```

如果上一轮已有训练代码，直接复用已有 selector implementation，不要复制一套新实现。

## 4.2 Train / Test 两套 evaluator

增加：

```python
evaluate_split(model, train_loader)
evaluate_split(model, heldout_loader)
```

输出：

```json
{
  "train": {
    "selected_ade": 0.0,
    "selected_fde": 0.0,
    "ade_regret": 0.0,
    "oracle_hit": 0.0,
    "near_oracle_005": 0.0,
    "near_oracle_010": 0.0,
    "pairwise_acc": 0.0,
    "spearman": 0.0,
    "kendall": 0.0,
    "ndcg5": 0.0
  },
  "test": {}
}
```

---

# 5. Ranking Metrics 实现

## 5.1 Pairwise accuracy

对 candidate i/j：

```python
gt = ade_i < ade_j
pred = score_i > score_j
```

忽略：

```text
|ADE_i - ADE_j| < margin
```

默认 margin：

```text
0.01m
```

另外输出：

```text
margin 0.05
margin 0.10
margin 0.20
```

## 5.2 Spearman

每个 sample 计算 predicted rank vs ADE rank，然后 macro average。

## 5.3 NDCG

relevance：

```python
relevance = exp(-ADE / tau)
```

建议：

```text
tau = 0.1
```

输出：

```text
NDCG@5
NDCG@20
```

---

# 6. EXP-S1B Oracle Margin

脚本：

```text
run_s1b_oracle_margin.py
```

完全不训练模型。

每个 sample：

```python
ade_sorted = sort(candidate_ade)

delta12 = ade_sorted[1] - ade_sorted[0]
delta15 = ade_sorted[4] - ade_sorted[0]

native_idx = argmax(native_logits)
native_regret = candidate_ade[native_idx] - ade_sorted[0]
```

输出：

```text
oracle_margin_samples.csv
summary.json
RESULT_ANALYSIS.md
```

## 6.1 Near-Oracle

```python
for eps in [0.02, 0.05, 0.10]:
    near_mask = candidate_ade <= min_ade + eps
```

统计：

```text
near_oracle_count
native_is_near_oracle
native_top5_contains_near_oracle
```

---

# 7. EXP-S4A Native Regret

可直接复用 S1B 结果。

阈值：

```python
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]
```

输出：

```text
regret_distribution.csv
regret_by_fold.csv
```

---

# 8. EXP-S4B Bad-Native Gate

## 8.1 Dataset

每个 sample 一条：

```text
input:
scene-level deployable feature

label:
native_regret >= threshold
```

不要把 20 candidate 展平成 20 条样本。

## 8.2 Gate Feature

建议：

```text
native logit entropy
top1-top2 logit margin
candidate set dispersion
candidate medoid distance
focal speed / acceleration
neighbor count
nearest distance
MPN predicted risk
native trajectory smoothness
native candidate consensus distance
```

## 8.3 Gate Models

第一轮：

```text
LogisticRegression
2-layer MLP
Gradient Boosted Tree
```

## 8.4 Class imbalance

必须：

```text
report class prevalence
class weighting
AUPRC
```

不要只看 accuracy。

---

# 9. EXP-S4C Conditional Rerank

实现：

```python
if gate_prob < threshold:
    selected = native_idx
else:
    selected = reranker(...)
```

reranker 先支持：

```text
linear_residual_g012
mlp_pairwise_g0124
```

后续增加 latent ranker。

## 9.1 Gate Threshold

不要固定 0.5。

扫描：

```text
0.1
0.2
...
0.9
```

或者更推荐按照 intervention budget：

```text
top 5%
top 10%
top 20%
...
```

---

# 10. EXP-S4D Budget Curve

实现：

```python
gate_score = P(high_regret)
rank samples descending
```

budget：

```python
[0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
```

输出：

```text
budget
selected_ade
ade_improvement
regret_recovery
false_intervention
useful_intervention
```

---

# 11. EXP-S2 Latent Hook

这一部分需要改 F16 inference path。

原则：

```text
不要改变 forward numerical result
只增加 hook / cache
```

---

# 12. 首先定位 latent

Agent 需要检查：

```text
FrontEncoderASTRA
decoder adapter
Q20 decoder
mode query
confidence head / mode logit head
```

优先寻找：

```text
[batch, agent, mode, dim]
```

形状 tensor。

最理想：

```text
[B, N, 20, D]
```

---

# 13. Latent Hook 设计

推荐统一：

```python
self.debug_cache = {}
```

并增加：

```python
def cache_debug_tensor(name, tensor):
    if self.enable_debug_cache:
        self.debug_cache[name] = tensor.detach()
```

默认：

```python
enable_debug_cache = False
```

只有 latent extraction script 打开。

不要为了缓存 latent 改变任何模型计算路径。

---

# 14. 建议缓存的 latent

```text
focal_encoder_token
social_context_token
mode_query
mode_hidden_pre_output
mode_hidden_pre_logit
```

如果没有 mode-specific hidden，至少缓存 mode query + focal context。

---

# 15. Latent Cache

单独存：

```text
latent_cache/
seed_1/
eth/
shard_000.pt
...
```

不要覆盖旧 candidate cache。

manifest 保存：

```json
{
  "git_sha": "...",
  "checkpoint": "...",
  "latent_names": [],
  "shapes": {}
}
```

---

# 16. EXP-S2B Probe

## Linear Probe

```python
score_k = Linear(latent_k)
```

训练目标：

```text
predict -ADE
```

或 pairwise ranking。

## Residual Probe

重点：

```python
score_k = native_logit_k + alpha * mlp(latent_k)
```

`alpha` 可以固定为 1，或作为一个 learnable scalar。

第一轮不要做大规模 grid。

---

# 17. EXP-S2C Pairwise Separability

构造：

```python
if ade_i + margin < ade_j:
    label = 1
```

输入优先：

```text
latent_i - latent_j
```

比 concat 参数更少。

---

# 18. Pair Sampling

20 modes 会产生 190 pairs/sample。

避免数据爆炸：

```text
每 sample 随机 32 pairs
+
native vs oracle
+
top1 native vs top5 candidates
```

保证包含关键 pair。

---

# 19. EXP-S3 Large Probe

只有 S2 后再实现。

目录：

```text
S3_LARGE_PROBE/large_probe_model.py
```

---

# 20. 模型结构建议

## Candidate encoder

输入：

```text
candidate trajectory [12,2]
mode latent
native score
G1/G4 features
```

输出：

```text
candidate token [D]
```

## History encoder

输入 focal history，使用小型 TCN / Transformer。

## Social encoder

输入 neighbor histories，输出 social tokens。

## Set ranker

```text
20 candidate tokens
+
context token
+
social tokens
↓
3~6 layer Transformer
↓
20 scalar scores
```

---

# 21. S3 Loss

第一轮：

```text
50% listwise
50% pairwise
```

$$
L=0.5L_{listwise}+0.5L_{pairwise}
$$

暂时不要加 safety multi-task，先回答 accuracy rankability upper bound。

---

# 22. S3 Early Stop

validation metric：

```text
ADE Regret
```

而不是 loss。

建议：

```text
patience = 10 epochs
max_epochs = 100
```

---

# 23. 结果目录标准

每个实验必须输出：

```text
config.json
metrics.json
per_fold.csv
predictions.csv/parquet
run.log
RESULT_ANALYSIS.md
```

训练实验额外：

```text
best_model.pt
training_curve.csv
```

---

# 24. 结果表标准

统一至少包含：

```text
Method
Train_Selected_ADE
Test_Selected_ADE
Train_Regret
Test_Regret
Oracle_Hit
NearOracle_005
NearOracle_010
PairwiseAcc
Spearman
NDCG5
NDCG20
```

S4 额外：

```text
InterventionRate
UsefulInterventionRate
FalseInterventionRate
RecoveredRegret
```

S2 额外：

```text
LatentName
LatentDim
ProbeType
```

---

# 25. Leave-One-Fold-Out

保持上一轮协议：

```text
train 4 folds
test 1 fold
```

但 S1 必须同时保存 train evaluation 与 held-out evaluation。

注意 train evaluation 只用于 rankability diagnosis，不用于最终泛化结论。

---

# 26. Seed 策略

第一阶段只跑：

```text
Seed 1
```

适用于：

```text
S1
S1B
S4A
S2 exploratory
```

只有看到明显 signal 后，再跑：

```text
Seeds 1,2,3
```

不要一开始所有实验都 3 seeds。

---

# 27. 主 Orchestrator

新增：

```text
run_rankability_series.py
```

推荐 stage：

```text
stage1:
    S1A
    S1B
    S4A

stage2:
    latent extraction
    S2B
    S2C

stage3:
    S4B
    S4C
    S4D

stage4:
    S3A
```

---

# 28. 自动诊断逻辑

可增加简单 rule：

```python
if train_selected_ade > 0.45:
    recommend = "representation_bottleneck"

elif test_selected_ade - train_selected_ade > 0.10:
    recommend = "generalization_bottleneck"

elif latent_probe_ade < 0.45:
    recommend = "latent_selector_promising"

else:
    recommend = "run_large_probe"
```

这只用于自动总结，不要让脚本完全自动决定科学结论。

---

# 29. 夜间模式

支持：

```bash
python run_rankability_series.py --stage stage1 --resume
```

再：

```bash
python run_rankability_series.py --stage stage2 --resume
```

每个 job：

```text
独立 subprocess
独立 log
失败后继续
成功自动 skip
```

---

# 30. Windows 注意事项

夜间：

```text
机器接电
关闭自动睡眠
可以关闭屏幕
不要让系统 suspend
```

---

# 31. Agent 第一批具体任务

直接给 Coding Agent：

```text
TASK 1
检查上一轮 cache schema
实现 common/cache_io.py

TASK 2
实现 EXP-S1B oracle margin
实现 EXP-S4A regret distribution

TASK 3
修改现有 selector evaluator
增加 train split metrics
完成 EXP-S1A

TASK 4
输出 S1 总结报告

TASK 5
定位 F16 mode-specific latent
只写 latent extraction hook
不要改模型计算逻辑

TASK 6
实现 linear / residual latent probe
完成 S2B

TASK 7
根据 S1/S2 结果再决定是否实现 Large Probe
```

---

# 32. 第一阶段最低可交付物

```text
S1_RANKABILITY_AUDIT/results/
    train_test_rankability.csv
    oracle_margin_summary.json
    near_oracle_summary.json
    RESULT_ANALYSIS.md

S4_REGRET_GATE/results/
    regret_distribution.csv
    regret_by_fold.csv
    RESULT_ANALYSIS.md
```

这一步完成后先分析，不要直接让 Agent 一口气实现 S3 Large Probe。

---

# 33. 最重要的工程约束

必须保持：

```text
future GT
candidate ADE
candidate FDE
GT collision
```

只能用于：

```text
label
evaluation
diagnostic
```

绝不能进入 deployable feature。

建议代码直接做：

```python
assert no_privileged_feature_leakage(...)
```

避免后续实验出现隐蔽数据泄漏。

---

# 34. 推荐实现优先级

```text
P0:
S1A
S1B
S4A

P1:
S2 latent extraction
S2B latent probe

P2:
S4 gate / conditional rerank

P3:
S3 large probe
```

除非 S1 已经给出非常明确的结论，否则不要改变这个顺序。
