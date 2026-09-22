# Trajectory Candidate Selection：Validity Audit、实验重跑与结论修订指导

> Repository: `1inke/trajectory-candidate-selection`  
> Current reference commit: `f4ffe325417d51dec6a2ddf19344547af38b69f9`  
> Purpose: 修复当前 Rankability 实验中会影响科学结论的评估问题，重新确认真实瓶颈，并同步修改 README / Final Report 中过度终局化的表述。

---

# 0. 当前阶段总判断

当前项目已经得到三个相对可靠的事实：

1. F16 的候选覆盖能力远强于 Native Top-1：
   - Oracle minADE20 约 `0.2181m`
   - Native Selected ADE 约 `0.5184m`
2. 已测试的大量 post-hoc selector 只能闭合很小一部分 ADE oracle gap。
3. Bad-Native Regret Gate 存在明显可预测性：
   - Gate ROC-AUC 约 `0.78`
   - 条件干预比全量重排更有性价比。

但以下“终局性”结论目前证据不足，必须暂时撤回或降级：

```text
“已经证明 information-theoretic limit”
“已经证明 aleatoric uncertainty 是主要原因”
“训练集内部所有 selector 都学不会”
“deep latent 也不包含有效选择信息”
“候选在物理上不可被筛选”
```

原因不是这些结论一定错误，而是当前代码中存在至少两个会直接影响 Rankability 诊断的评估问题，并且 S2 latent probe 还不足以支撑“深层表示信息也不存在”的结论。

因此下一阶段最高优先级不是增加新 selector，也不是直接上 Large Probe，而是：

```text
S0 Validity Audit
        ↓
S1A-v2 Corrected Rankability
        ↓
S5 Native Rank Position / Near-Oracle Recall
        ↓
S6 Hard-Case Local Reranking
        ↓
S2-v2 Context-Conditioned Latent Probe
        ↓
必要时再做 S3 Large Probe
```

---

# 1. P0：立即修复的两个核心问题

## 1.1 S1A 神经网络 Train Evaluation 样本错位

当前 `run_s1a_train_test_rankability.py` 中，神经模型训练后：

```python
eval_tr_idx = np.random.choice(
    N_tr,
    min(N_tr, 5000),
    replace=False
)

tr_scores_sub = net(
    tr_norm[eval_tr_idx].to(device)
).cpu().numpy()
```

这里 prediction 对应的是随机抽取的 `eval_tr_idx` 样本，但随后评估时使用：

```python
tr_ade[:len(tr_scores)]
```

以及其他数组的前 `len(tr_scores)` 条。

因此当前实际计算是：

```text
Random Sample Predictions
        vs
First-N Ground Truth
```

样本并不对应。

## 1.2 修复方式

推荐方案 A：直接完整评估 Train。

34161 个 sample × 20 candidate 的规模完全可以接受。

改成 batch inference：

```python
def infer_scores_batched(net, x, device, batch_size=512):
    outputs = []

    net.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            bx = x[start:start + batch_size].to(device)
            outputs.append(net(bx).cpu().numpy())

    return np.concatenate(outputs, axis=0)
```

然后：

```python
train_scores = infer_scores_batched(
    net,
    tr_norm,
    device
)
```

最终：

```python
evaluate_split_predictions(
    train_scores,
    full_train_data
)
```

这是推荐方案。

如果必须使用 subset，则必须同步返回 `eval_tr_idx`，并严格：

```python
sub_train_data = {
    "ade": tr_ade[eval_tr_idx],
    "fde": tr_fde[eval_tr_idx],
    "min_ade20": tr_min_ade20[eval_tr_idx],
    "min_fde20": tr_min_fde20[eval_tr_idx],
    "native_top1_idx": tr_native_idx[eval_tr_idx],
}
```

禁止使用：

```python
[:len(tr_scores)]
```

---

# 2. P0：Pairwise Accuracy 方向错误

当前 `common/metrics.py`：

```python
ade_diff = ade[:, :, None] - ade[:, None, :]
score_diff = scores[:, :, None] - scores[:, None, :]

valid_mask = ade_diff > margin
correct = (score_diff > 0) & valid_mask
```

但：

```text
ade_diff[i,j] = ADE_i - ADE_j
```

当：

```text
ade_diff > margin
```

意味着：

```text
ADE_i > ADE_j
```

即 candidate i 比 candidate j 更差。

随后却把：

```text
score_i > score_j
```

作为正确预测。

因此当前 Pairwise Accuracy 实际在大量情况下测的是逆排序正确率 / ranking error。

---

# 3. Pairwise Metric 正确实现

推荐直接写成：

```python
ade_gap = (
    candidate_ade[:, None, :]
    - candidate_ade[:, :, None]
)
# ade_gap[i,j] = ADE_j - ADE_i
# > 0 表示 i 比 j 更好

score_gap = (
    predicted_scores[:, :, None]
    - predicted_scores[:, None, :]
)

valid = ade_gap > margin
correct = score_gap > 0

accuracy = (
    correct[valid].mean()
    if valid.any()
    else np.nan
)
```

---

# 4. 必须新增 Ranking Metric 单元测试

新增：

```text
tests/test_ranking_metrics.py
```

## Test A：Perfect Ranking

```python
ade = np.array([
    [0.1, 0.2, 0.3, 0.4]
])

scores = np.array([
    [4.0, 3.0, 2.0, 1.0]
])
```

必须：

```text
Pairwise Accuracy = 1.0
Spearman = 1.0
Kendall = 1.0
```

## Test B：Reverse Ranking

```python
scores = np.array([
    [1.0, 2.0, 3.0, 4.0]
])
```

必须：

```text
Pairwise Accuracy = 0.0
Spearman = -1.0
Kendall = -1.0
```

## Test C：Margin

```python
ADE = [0.10, 0.11, 0.30]
margin = 0.05
```

第一、第二候选的 pair 不应进入统计；第一、第三候选应进入统计。

---

# 5. P0：Residual Score Scaling 改为 Train-only Calibration

当前部分代码使用：

```python
std_te_log = np.std(test_logits)
std_te_delta = np.std(test_delta)
```

再生成 test score。

这不使用 GT，因此不属于标签泄漏，但属于 transductive test-distribution calibration，而且 Train/Test 分数不再完全可比。

统一改为：

```python
train_logit_std = np.std(train_logits)
train_delta_std = np.std(train_delta) + 1e-6

scale = train_logit_std / train_delta_std

train_scores = (
    train_logits
    + alpha * scale * train_delta
)

test_scores = (
    test_logits
    + alpha * scale * test_delta
)
```

注意：

```text
scale 只能来自 train folds
```

---

# 6. Alpha 不再固定 0.3

当前 Residual Ranker 使用：

```text
alpha = 0.3
```

建议在 4 个训练 folds 内部做简单 inner validation：

```text
alpha ∈ {
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    1.00
}
```

协议：

```text
4 training folds
    ↓
内部留 1 fold validation
    ↓
选择 alpha
    ↓
在全部 4 folds 重训
    ↓
评估 held-out fold
```

不要用 held-out test fold 选择 alpha。

---

# 7. P0：Final Report 中的硬编码结论必须修复

当前 `run_rankability_series.py` 中直接硬编码：

```text
“超过 83% 的样本 Δ1,2 < 5cm”
```

但 S1B 实际结果：

```text
Δ1,2 < 2cm : 22.66%
Δ1,2 < 5cm : 37.70%
```

因此必须让最终报告从 JSON / CSV 读取结果，而不是手写数字。

---

# 8. 建议增加统一 Report Data Layer

新增：

```text
common/report_data.py
```

例如：

```python
def load_s1b_summary():
    ...

def load_s1a_summary():
    ...

def load_s2_summary():
    ...

def load_s4_summary():
    ...
```

最终报告：

```python
s1b = load_s1b_summary()
p_delta_005 = s1b["p_delta12_lt_005"]
```

禁止 `83%`、`0.7872`、`37.70%` 等数字直接写死在报告代码中。

---

# 9. EXP-S0：Validity Audit

建议新增：

```text
SELECTOR_RANKABILITY_SERIES/
└── S0_VALIDITY_AUDIT/
    ├── run_s0a_metric_tests.py
    ├── run_s0b_alignment_audit.py
    ├── run_s0c_report_consistency.py
    └── results/
```

---

# 10. EXP-S0A：Metric Correctness Audit

运行：

```text
synthetic perfect ranking
synthetic reverse ranking
margin filtering
NDCG perfect ranking
Spearman / Kendall
```

输出：

```text
metric_tests.json
```

必须全部：

```text
PASS
```

否则禁止运行后续实验。

---

# 11. EXP-S0B：Alignment Audit

对 S1A：

```text
sample_id
prediction
GT
```

必须显式对齐。

随机抽取 100 个样本：

```text
assert prediction sample_id == label sample_id
```

如果 feature cache 没有 `sample_id`，补充 index / fold-local index。

---

# 12. EXP-S0C：Report Consistency Audit

自动比较：

```text
source JSON / CSV
vs
generated Markdown
```

至少验证：

```text
Δ1,2 < 5cm
Gate AUC
Gate AUPRC
Native ADE
Best Selector ADE
Latent Probe ADE
```

如果报告数字和原始结果不同：

```text
CI / script fail
```

---

# 13. EXP-S1A-v2：重新跑 Train-vs-Test Rankability

修复上述问题后，重新运行 S1A。

这次代表方法改成：

```text
linear_residual_g012
mlp_residual_g0124
transformer_residual_g0124
mlp_pairwise_g0124
mlp_listwise_g0124
```

其中重点是 Residual Rankers，因为 Phase 1 已经表明：

```text
保留 Native prior
+
学习 correction
```

明显比从零重学排序稳定。

---

# 14. S1A-v2 必须输出

每 fold：

```text
Train Selected ADE
Test Selected ADE

Train Regret
Test Regret

Train PairwiseAcc@0.05
Train PairwiseAcc@0.10
Train PairwiseAcc@0.20

Test PairwiseAcc@0.05
Test PairwiseAcc@0.10
Test PairwiseAcc@0.20

Spearman
Kendall
NDCG@5
NDCG@20

Oracle Hit@1
NearOracle@0.02
NearOracle@0.05
NearOracle@0.10
```

---

# 15. 修正后的 S1A 判定规则

## Case A：Train 明显优于 Test

例如：

```text
Train ADE < 0.40
Test ADE > 0.50
```

结论：

```text
GENERALIZATION / DOMAIN SHIFT
```

下一步：

```text
scene normalization
fold adaptation
density-conditioned ranker
```

## Case B：Train 与 Test 都约 0.48~0.51

结论可以写：

> Current feature representation shows limited top-of-list discriminability.

不要直接写：

```text
information-theoretic limit proved
```

## Case C：Train 可以到 <0.35

结论：

> candidate ranking signal is learnable, but current cross-fold generalization is insufficient.

---

# 16. EXP-S5：Native Rank Position Audit

这是修复后最重要的新实验之一。

目标：

> Oracle / Near-Oracle candidate 在 Native ranking 中通常排第几？

---

# 17. S5 需要统计

Exact Oracle：

```text
Recall@1
Recall@2
Recall@3
Recall@5
Recall@10
Recall@20
```

Near-Oracle：

```text
ε = 0.02m
ε = 0.05m
ε = 0.10m
```

对于每个 ε：

```text
NearOracle Recall@1
Recall@2
Recall@3
Recall@5
Recall@10
```

---

# 18. 推荐输出

```text
native_oracle_rank_distribution.csv
near_oracle_recall_at_k.csv
oracle_rank_by_fold.csv
RESULT_ANALYSIS_S5.md
```

---

# 19. S5 的关键科学意义

如果：

```text
NearOracle@0.05 Recall@5 很高
```

例如：

```text
> 80%
```

则下一步不再做：

```text
20-way global ranking
```

而改为：

```text
Native coarse ranking
        ↓
Top-5
        ↓
Local Fine Reranker
```

这和当前结果中的高 Spearman、低 Oracle Hit 非常一致。

---

# 20. 新假设：Global Rankability 可能并不差

修正 Pairwise 指标后，如果出现：

```text
Pairwise Accuracy ≈ 0.80~0.90
Spearman ≈ 0.75
```

而：

```text
Oracle Hit ≈ 0.18~0.20
```

则新的研究结论应该是：

> 当前 candidate set 在全局上具有较强的可排序性，但最优附近的候选之间缺乏足够的 top-of-list discrimination。

这是一个比“候选不可筛选”更准确、更有研究价值的结论。

---

# 21. EXP-S5B：Top-K Fine Ranking Difficulty

只在 Native Top-K 内评估：

```text
K = 3
K = 5
K = 10
```

指标：

```text
Pairwise Accuracy within Top-K
Spearman within Top-K
NDCG within Top-K
NearOracle Recall
```

重点比较：

```text
Global 20-way ranking
vs
Top-5 ranking
```

---

# 22. EXP-S6：Hard-Case Specialized Local Reranker

当前 S4 Gate 是值得保留的。

已有结果说明：

```text
预测 Native 是否高 regret
```

比：

```text
预测 20 条里谁是绝对最佳
```

容易得多。

因此下一步推荐：

```text
Native
  ↓
Bad-Native Gate
  ├── Easy → Native
  │
  └── Hard
        ↓
     Native Top-5
        ↓
Specialized Local Reranker
```

---

# 23. Hard-Case 定义

尝试三档：

```text
Regret > 0.10m
Regret > 0.20m
Regret > 0.30m
```

训练方式：

## Option A

只训练 hard samples。

## Option B

全部样本训练：

```text
hard sample weight = 5
easy sample weight = 1
```

## Option C

curriculum：

```text
先全部训练
后 hard-only fine-tune
```

推荐优先：

```text
B → C
```

---

# 24. Local Reranker 只处理 Native Top-K

第一轮：

```text
K=5
```

模型：

```text
Linear Residual
MLP Residual
Tiny Set Attention
```

不要直接再上复杂 Transformer。

---

# 25. 新训练目标：Near-Oracle Set

不再强制：

```text
exact argmin ADE
```

定义：

$$
O_\epsilon
=
\left\{
k:
ADE_k
\le
\min_j ADE_j+\epsilon
\right\}
$$

推荐：

```text
ε = 0.02m
ε = 0.05m
ε = 0.10m
```

---

# 26. Near-Oracle Set Loss

可以使用 multi-positive soft target：

```python
target = np.zeros(K)
target[near_oracle_mask] = 1.0
target /= target.sum()
```

然后使用 KL / Cross Entropy。

这样不再把：

```text
0.201m
vs
0.204m
```

强行当成完全不同类别。

---

# 27. Top-heavy Pairwise Loss

只对 Native Top-K：

```text
Top-3
Top-5
```

采 pair。

权重：

$$
w_{ij}
=
|ADE_i-ADE_j|
$$

loss：

$$
L_{pair}
=
w_{ij}
\log
\left(
1+\exp(-(s_i-s_j))
\right)
$$

其中只保留：

```text
ADE_i + margin < ADE_j
```

---

# 28. Listwise Temperature 重新扫描

当前部分 listwise 实验：

```text
tau = 0.5
```

可能过软。

建议：

```text
τ = 0.02
0.05
0.10
0.20
```

推荐 target：

$$
q_k
=
softmax
\left(
-\frac{
ADE_k-\min ADE
}{\tau}
\right)
$$

而不是直接：

$$
softmax(-ADE/\tau)
$$

---

# 29. EXP-S2-v2：Latent Probe 重新定义

当前 S2 使用：

```text
decoder.mode_mlp output
L3
[B,20,64]
```

但还不能证明这个 tensor 是充分 sample-conditioned 的。

所以 S2-v2 第一件事不是继续加 probe，而是：

```text
检查 L3 到底包含多少 sample-specific information
```

---

# 30. S2-v2A：L3 Sample Variance Audit

对每个 mode k：

```text
Var over samples
Mean cosine similarity over samples
Within-mode variance
Between-mode variance
```

计算：

$$
R
=
\frac{
Var_{between-mode}
}{
Var_{within-mode}
}
$$

如果：

```text
same mode across different samples
高度相似
```

则说明：

```text
L3 更接近 mode embedding
而不是 sample-conditioned intention representation
```

此时当前 S2 结论必须撤回。

---

# 31. S2-v2B：寻找真正 Context-Conditioned Latent

优先定位：

```text
L4:
mode-specific hidden after context fusion

L5:
confidence-head input / pre-logit hidden
```

理想 shape：

```text
[B, N, 20, D]
```

或者 focal：

```text
[B,20,D]
```

---

# 32. S2-v2C：新的 Probe

测试：

```text
G0 + L4
G0 + L5
G0 + G1 + L4
G0 + G1 + L5
G0 + G1 + G4 + L5
```

Probe：

```text
Linear Residual
MLP Residual
Top-5 Local MLP
```

---

# 33. S3 Large Probe 暂缓

只有在：

```text
S0 全部修复
S1A-v2 结果可信
S5 完成
S6 仍不足
S2-v2 仍不足
```

之后，才进入：

```text
S3 Large Capacity Probe
```

当前不要优先实现。

---

# 34. README / Final Report 的具体结论修改建议

下面给出建议直接替换的科学表述。

---

# 35. 当前不建议保留的表述

## 原表述类型 A

```text
REPRESENTATION_BOTTLENECK 确立
```

改为：

> Current deployable feature sets provide only limited top-of-list discrimination under cross-fold evaluation. The exact source of the limitation remains under validation after correcting train-evaluation and ranking-metric issues.

中文：

> 当前可部署特征在跨折评估中仅表现出有限的 Top-of-list 判别能力。由于 Train evaluation 与 Pairwise metric 正在进行有效性修正，目前尚不能将瓶颈终局归因于 representation insufficiency。

---

# 36. 原表述类型 B

```text
彻底排除了纯泛化差
```

改为：

> 当前结果尚不能排除泛化因素。现有 S1A 神经模型的 Train 指标存在样本对齐问题，需要修复后重新比较 Train/Test gap。

---

# 37. 原表述类型 C

```text
无可辩驳地证明 aleatoric uncertainty
```

改为：

> Oracle margin audit 显示部分样本存在显著的 near-oracle ambiguity，例如 37.70% 的样本中 Top-1 与 Top-2 ADE 差距低于 5cm。这说明单一 argmin 标签可能过于尖锐，但尚不足以证明大部分 oracle gap 都来自不可约 aleatoric uncertainty。

---

# 38. 原表述类型 D

```text
Deep latent 也受到同样的物理极限
```

改为：

> 当前基于 decoder `mode_mlp` 输出的 L3 probe 仅带来小幅增益。由于尚未确认 L3 的 sample-conditioned 程度，也尚未评估 context-fused mode hidden / pre-logit hidden，因此不能据此排除更深层表示中存在可用于 candidate selection 的信息。

---

# 39. 原表述类型 E

```text
候选在物理上不可筛选
```

改为：

> 当前结果显示，候选的精确 Top-1 选择明显比整体排序困难。若修正后的 Pairwise 与 Spearman 结果继续维持较高水平，则更合理的解释是“global ranking 较强，但 top-of-list fine discrimination 较弱”。

---

# 40. 可以保留的结论

以下可以继续保留：

1. F16 candidate support 与 Native Top-1 之间存在显著 Oracle Gap。
2. 目前 tested post-hoc selectors 只能闭合较小部分 ADE gap。
3. Residual ranking 比完全从零学习 ranking 更稳定。
4. Model-predicted neighbor risk 对 safety reranking 有稳定价值。
5. Bad-Native detection 明显比 exact-best candidate identification 更容易。

---

# 41. README 建议的新 Executive Summary

建议改为：

> The frozen F16 generator exhibits a large gap between multi-hypothesis coverage and native Top-1 selection. Extensive offline selectors recover only a small fraction of this gap. However, current evidence does not yet support the stronger claim that the oracle gap is fundamentally unidentifiable.
>
> A validity audit identified issues in the train-set evaluation alignment and pairwise-ranking metric implementation. The next phase therefore focuses on corrected rankability evaluation, native rank-position analysis, near-oracle recall, and conditional local reranking.
>
> Preliminary evidence suggests a potentially more specific bottleneck: candidate sets may be globally rankable while remaining difficult to discriminate near the top of the ranking.

---

# 42. RANKABILITY_FINAL_REPORT 建议的新标题

原：

```text
Rankability & Information Sufficiency Final Report
```

建议：

```text
Rankability Diagnostic Report — Validity-Audited Revision
```

不要再使用：

```text
Final
Theoretical Limit
Information-Theoretic Limit
Ultimate Verdict
```

直到 S0/S1-v2/S5 完成。

---

# 43. 建议新的实验编号

```text
EXP-S0A  Ranking Metric Correctness
EXP-S0B  Train Evaluation Alignment
EXP-S0C  Report Consistency

EXP-S1A-v2 Corrected Train-Test Rankability

EXP-S5A Native Oracle Rank Distribution
EXP-S5B Near-Oracle Recall@K
EXP-S5C Top-K Fine Ranking Audit

EXP-S6A Hard-Case Weighted Reranker
EXP-S6B Gate + Top-5 Local Reranker
EXP-S6C Near-Oracle / Top-heavy Loss

EXP-S2-v2A L3 Sample-Variance Audit
EXP-S2-v2B Context-Conditioned L4/L5 Extraction
EXP-S2-v2C L4/L5 Local Probe

EXP-S3 Large Probe
```

---

# 44. 推荐执行顺序

严格建议：

```text
Phase A — Validity
S0A
S0B
S0C

Phase B — Re-evaluation
S1A-v2
S5A
S5B
S5C

Phase C — Focused Improvement
S6A
S6B
S6C

Phase D — Latent Revisit
S2-v2A
S2-v2B
S2-v2C

Phase E — Upper Bound
S3
```

---

# 45. 重跑时 Seed 策略

Validity 修复阶段只用：

```text
Seed 1
```

先验证方向。

如果 S1A-v2 / S5 / S6 出现明显信号，再运行：

```text
Seeds = 1,2,3
```

不要一开始把所有修复实验全跑 3 seeds。

---

# 46. Agent 第一轮任务清单

建议直接交给 Agent：

```text
TASK 1
修复 common/metrics.py pairwise direction

TASK 2
新增 ranking metric synthetic tests

TASK 3
修复 S1A train prediction / GT alignment

TASK 4
Residual scale 改成 train-only

TASK 5
alpha 增加 inner-validation

TASK 6
删除 report 中硬编码的 83% 等数字

TASK 7
新增 S0 report consistency checks

TASK 8
重跑 S1A-v2

TASK 9
新增 S5 native oracle rank / near-oracle Recall@K

TASK 10
只在 S1-v2 + S5 分析完成后实现 S6
```

---

# 47. 第一批重跑结束后的最低交付物

必须产生：

```text
S0_VALIDITY_AUDIT/results/
    metric_tests.json
    alignment_audit.json
    report_consistency.json
    RESULT_ANALYSIS_S0.md

S1_RANKABILITY_AUDIT/results_v2/
    train_test_rankability_v2.csv
    RESULT_ANALYSIS_S1A_V2.md

S5_TOPK_AUDIT/results/
    native_oracle_rank_distribution.csv
    near_oracle_recall_at_k.csv
    topk_ranking_metrics.csv
    RESULT_ANALYSIS_S5.md
```

---

# 48. 第一批结果出来后如何决策

## Scenario A

如果：

```text
Corrected Pairwise ≈ 80%+
Spearman ≈ 0.75+
NearOracle@0.05 Recall@5 高
```

则主线改为：

```text
Global ranking is largely solved
Top-K discrimination is the bottleneck
```

后续重点：

```text
Gate + Top-5 Local Reranker
```

## Scenario B

如果：

```text
Train ADE << Test ADE
```

主线：

```text
Generalization
```

后续重点：

```text
scene-conditioned selector
fold normalization
domain adaptation
```

## Scenario C

如果：

```text
Train/Test 都差
且
Corrected Pairwise 也接近随机
```

才有更充分理由推进：

```text
representation insufficiency
latent audit
large probe
```

---

# 49. 当前最推荐的研究假设

在修复之前，最合理的工作假设应调整为：

> F16 已生成具有较强覆盖能力的多模态候选。现有 selector 对候选整体质量可能已有较强排序能力，但在 near-oracle 模态之间的 Top-of-list discrimination 上仍然不足。与此同时，Native 高 regret 样本具有一定可预测性，因此“先判断是否需要干预，再对局部 Top-K 做细粒度重排”可能比全局 20-way selection 更适合该问题。

这一假设必须由：

```text
S1A-v2
+
S5
```

验证，而不是提前写成最终结论。

---

# 50. 最终原则

接下来不要继续追求：

```text
更多 selector 数量
```

而要提高：

```text
实验有效性
指标可信度
Top-K 问题定位
结论强度与证据匹配程度
```

当前项目最需要的不是“再赢 2mm”，而是先确认：

```text
我们到底在测什么，
哪些指标是真的，
真正困难的是 global ranking，
还是 top-of-list discrimination。
```

只有这一步确认后，再继续设计轻量 selector 才有意义。
