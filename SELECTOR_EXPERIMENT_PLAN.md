# F16 Offline Selector Zoo：候选轨迹筛选器夜间实验方案

## 0. 实验目标

当前问题不是简单的候选生成能力不足，而是 **F16 已经生成了高质量候选，但 Native Score 无法稳定把最优/更安全的候选排到 Top-1**。

因此将生成器与筛选器彻底解耦：

```text
Frozen F16
    ↓
一次性 Candidate Cache
    ↓
Feature Bank
    ↓
Selector Zoo（20~30+ 个方案）
    ↓
统一 Leave-One-Fold-Out Evaluator
    ↓
Pareto / Finalists
    ↓
Seed 2/3 Confirmatory Run
```

核心目标：

1. 不重复训练 F16。
2. F16 前向只负责生成缓存。
3. selector 的几十种方法全部读取同一份缓存。
4. 第一阶段广搜，第二阶段自动淘汰，第三阶段多 seed 确认。
5. 严格隔离 deployable feature 与 privileged GT label，避免测试信息泄漏。

---

# 1. 数据协议：优先使用 OOF Candidate Cache

ETH/UCY 五个 fold：

```text
eth
hotel
univ
zara01
zara02
```

对每个 fold 使用该 fold 已训练好的 F16 checkpoint，在该 fold 的 held-out test scene 上生成候选缓存。

例如：

```text
F16-eth checkpoint    → eth held-out cache
F16-hotel checkpoint  → hotel held-out cache
...
```

这样每条 cache 样本对于其 F16 generator 都是 out-of-fold 样本。

## Selector 的 Leave-One-Fold-Out 训练

当评估 selector 在 `eth` 上：

```text
selector train:
    hotel + univ + zara01 + zara02 OOF caches

selector test:
    eth OOF cache
```

评估 `hotel` 时：

```text
selector train:
    eth + univ + zara01 + zara02

selector test:
    hotel
```

依次循环五次。

这比直接在 F16 自己的训练数据输出上训练 selector 更干净。

---

# 2. Candidate Cache 设计

建议新目录：

```text
SELECTOR_BENCHMARK_SERIES/
├── build_candidate_cache.py
├── build_feature_bank.py
├── run_selector_benchmark.py
├── aggregate_selector_results.py
├── select_finalists.py
├── selectors/
├── configs/
├── cache/
└── results/
```

## 2.1 每条样本建议缓存

```python
{
    # identity
    "sample_id": ...,
    "fold": ...,
    "seed": ...,

    # deployment-time observation
    "past": Tensor[N, T_obs, 2],
    "num_valid": int,

    # F16 focal outputs
    "focal_candidates": Tensor[20, 12, 2],
    "focal_logits": Tensor[20],

    # 推荐保存：邻居完整预测分布
    "neighbor_candidates": Tensor[N-1, 20, 12, 2],
    "neighbor_logits": Tensor[N-1, 20],

    # GT: 只允许用于 label / evaluation
    "focal_future_gt": Tensor[12, 2],
    "neighbor_future_gt": Tensor[N-1, 12, 2],

    # privileged labels，可离线预计算
    "candidate_ade": Tensor[20],
    "candidate_fde": Tensor[20],
    "candidate_gt_min_clearance": Tensor[20],
    "candidate_gt_collision": Tensor[20],
}
```

## 2.2 强制分区

代码层面明确：

```python
DEPLOYABLE_KEYS = {...}
PRIVILEGED_LABEL_KEYS = {...}
```

训练 selector 时：

- `PRIVILEGED_LABEL_KEYS` 可作为监督目标；
- 不允许进入 feature tensor；
- test-time selector 只能看到 `DEPLOYABLE_KEYS`。

## 2.3 缓存形式

不要把整个 fold 写成一个超大文件。

推荐：

```text
cache/f16/seed_1/eth/
    shard_00000.pt
    shard_00001.pt
    ...
    manifest.json
```

每 1024 或 2048 samples 一个 shard。

优点：

- 中断可恢复；
- 单个 shard 损坏不影响整折；
- 夜间运行更安全；
- 后续 DataLoader 可以流式读取。

轨迹坐标推荐保留 float32。
如果空间压力明显，再只把非关键 feature / latent 改 float16。

---

# 3. Feature Bank

Feature 不要每个 selector 自己重复计算。

统一预计算为 Feature Bank。

## G0 — Native confidence

每个 candidate：

```text
native_logit
native_softmax_prob
native_rank
logit_margin_to_top1
global_logit_entropy
```

## G1 — Candidate intrinsic geometry

```text
endpoint displacement
path length
mean / max speed
mean / max acceleration
jerk
curvature / turning
direction change
deviation from constant velocity extrapolation
temporal smoothness
```

## G2 — Candidate-set relative features

```text
mean distance to other 19 candidates
minimum distance to other candidates
distance to unweighted medoid
distance to native-probability-weighted barycenter
candidate local density
MBR cost
candidate diversity neighborhood
```

## G3 — History / social context

只使用观测历史：

```text
focal speed / acceleration / turn rate
neighbor count
nearest observed distance
relative velocity
relative heading
local density
CV-TTC
CV-DCA
history interaction strength
```

## G4 — Model-Predicted-Neighbor future risk

使用 F16 同一次 forward 对邻居产生的预测，不使用 GT：

```text
min predicted clearance
expected predicted clearance
probability-weighted collision risk
time of minimum predicted separation
integrated risk over 12 steps
predicted near-collision duration
worst-neighbor risk
```

如果缓存了邻居 20-mode distribution，不要只使用 neighbor Top-1。

可计算：

```text
E[risk]
P(collision)
CVaR-like predicted risk
```

## G5 — Encoder latent（第二阶段再加）

第一晚不建议把 latent 作为基础依赖。

只有当 G0~G4 的结果仍明显不足时，再增加：

```text
focal encoder token
pooled social token
graph context token
decoder query embedding
```

否则 selector 会和 F16 内部结构耦合过深。

---

# 4. 两条任务线

## Track-A：Accuracy Selector

目标：

```text
尽量逼近 argmin ADE candidate
```

核心指标：

```text
Selected ADE@1
Selected FDE@1
ADE Regret = Selected ADE - minADE20
FDE Regret
Oracle-hit@1
Near-Oracle@ε
NDCG / ranking correlation
```

训练 label：

```text
candidate_ade
```

或者 soft list target：

$$
q_k =
\frac{\exp(-ADE_k/\tau)}
{\sum_j \exp(-ADE_j/\tau)}
$$

## Track-B：Safety-Aware Selector

不要直接替换 Track-A。

独立研究：

```text
accuracy + social safety Pareto
```

指标：

```text
Collision Rate
Near-Collision Rate
Q5 Clearance
Bottom-5% CVaR
False Intervention Rate
Selected ADE/FDE
Oracle Gap Closure
```

推荐两个设计：

### B1. Two-stage gate

```text
Native candidate
    ↓
Risk detector
    ↓
safe → 保持 Native
unsafe → rerank Top-K
```

### B2. Multi-task candidate scorer

同时预测：

```text
predicted ADE
predicted collision probability
```

测试时：

```text
score_k =
predicted_ADE_k + λ * predicted_collision_risk_k
```

λ 只在开发阶段选择。

---

# 5. 第一晚 Selector Zoo

建议第一晚至少包含以下配置。

## 无训练 / 极低成本

```text
native
cv_shield_k5
cv_shield_k20
mpn_shield_k5
mpn_shield_k20
mbr_uniform
mbr_native_weighted
native_plus_mbr_grid
native_plus_cv_risk_grid
native_plus_mpn_risk_grid
native_plus_smoothness_grid
consensus_plus_mpn_risk
```

其中 grid 方法不要只跑一个 λ。

例如：

```text
λ ∈ {0.1, 0.25, 0.5, 1.0, 2.0}
```

这些本身就能产生大量低成本候选。

## 线性 / 树模型

```text
ridge_utility_g01
linear_pointwise_g012
linear_pairwise_g012
linear_residual_g012
gbdt_pointwise_g012
lambdarank_g012
```

如果本地没有 LightGBM / XGBoost：

- 不要让整个夜间任务失败；
- 自动标记 SKIPPED_DEPENDENCY；
- 继续其他实验。

## Tiny Neural Rankers

```text
mlp_pointwise_g01
mlp_pointwise_g012
mlp_pairwise_g012
mlp_listwise_g012
mlp_residual_g012
mlp_multitask_g01234
deepsets_listwise_g0124
tcn_pairwise_g0124
gru_pairwise_g0124
tiny_transformer_listwise_g0124
two_stage_gate_mpn
```

### 特别优先

第一晚最值得关注：

```text
linear_pairwise
mlp_pairwise
mlp_listwise
mlp_residual
two_stage_gate_mpn
lambdarank（如果依赖可用）
```

---

# 6. 不建议直接做 Mode-ID 20 分类

不要：

```python
oracle_index = argmin(ADE)
loss = CrossEntropy(selector(history), oracle_index)
```

Mode 0~19 没有固定跨样本语义。

应该学习：

```text
f(context, candidate_k) → candidate score
```

然后：

```text
argmax(score_1 ... score_20)
```

---

# 7. 推荐 loss

## Pointwise regression

预测 candidate ADE：

$$
L_{point} =
|\hat{ADE}_k - ADE_k|
$$

建议对每个 sample 内 ADE 做相对归一化，减少 fold scale 差异。

## Pairwise RankNet

当：

$$
ADE_i < ADE_j
$$

要求：

$$
s_i > s_j
$$

损失：

$$
L_{rank}
=
-\log \sigma(s_i-s_j)
$$

每个 20-candidate sample 最多：

$$
\binom{20}{2}=190
$$

对 pair 太多时可以采样：

```text
hard pair：ADE 接近
easy pair：差距明显
native-vs-oracle pair
top5 internal pair
```

## Listwise soft target

$$
q_k = softmax(-ADE_k/\tau)
$$

$$
L_{list}=KL(q || p)
$$

## Residual ranking

推荐重点：

$$
s_k =
s_k^{native}
+
\alpha \Delta s_k
$$

Tiny selector 只学习修正 Native Ranking，而不是完全重新学习。

---

# 8. 第一晚的自动筛选逻辑

不要用一个人为总分把所有方法排死。

先计算 Pareto front。

## 基础淘汰约束（配置项，不要写死）

示例默认值：

```text
max ADE degradation vs Native: 1.0%
max FDE degradation vs Native: 1.0%
max collision increase: 0.10 percentage point
```

## Finalist 组成

自动保留最多 6 个：

```text
2 个：ADE regret 最低
2 个：在 ADE 约束下 collision / safety 最好
1 个：Pareto balanced
1 个：最低 latency / 最小参数量
```

如果重复则补下一个 Pareto candidate。

这样不会因为单一指标丢掉不同方向的好方案。

---

# 9. Overnight 三阶段

## Stage 1 — Seed 1 广搜

```text
build seed1 caches for 5 folds
        ↓
build feature bank
        ↓
run all 20~30+ configs
        ↓
5-fold selector LOO evaluation
        ↓
screen leaderboard
        ↓
select 6 finalists
```

## Stage 2 — Seed 2 / 3 Confirmatory

只为 finalists：

```text
build seed2 caches
build seed3 caches
        ↓
run 6 finalists
        ↓
aggregate 3 seeds × 5 folds
```

这里 Seed 1 是 method discovery；
Seed 2/3 用于确认稳定性。

正式论文结果不要只展示 Seed 1 的“广搜赢家”。

## Stage 3 — 自动生成最终报告

输出：

```text
FINAL_LEADERBOARD.csv
FINAL_PARETO.csv
FINAL_REPORT.md
failure_summary.json
```

最终报告至少包含：

```text
mean ± std across seeds
per-fold metrics
paired delta vs Native
ADE/FDE regret
collision / false intervention
latency
parameter count
```

---

# 10. 统计建议

最终 finalists 建议做 paired evaluation。

因为 Native 和 selector 对的是同一批样本，可计算：

```text
ΔADE(sample)
ΔFDE(sample)
Δcollision(sample)
```

再用 paired bootstrap 给 95% CI。

这比只报告五折平均更有解释力。

---

# 11. 推荐 CLI 契约

让本地 Agent 按这个接口实现，夜间 orchestrator 就不需要频繁改。

## Candidate cache

```bash
python SELECTOR_BENCHMARK_SERIES/build_candidate_cache.py \
  --model f16 \
  --seed 1 \
  --fold eth \
  --split test \
  --shard-size 2048 \
  --save-neighbor-distribution \
  --output-root SELECTOR_BENCHMARK_SERIES/cache
```

要求：

- 已完成 shard 自动跳过；
- `manifest.json` 标记完整性；
- 原子写文件：先 `.tmp` 后 rename；
- 任意 fold 中断后可 resume。

## Feature bank

```bash
python SELECTOR_BENCHMARK_SERIES/build_feature_bank.py \
  --seed 1 \
  --cache-root SELECTOR_BENCHMARK_SERIES/cache \
  --feature-groups g0,g1,g2,g3,g4
```

## 单个 selector benchmark

```bash
python SELECTOR_BENCHMARK_SERIES/run_selector_benchmark.py \
  --seed 1 \
  --method mlp_pairwise_g0124 \
  --protocol leave_one_fold_out \
  --cache-root SELECTOR_BENCHMARK_SERIES/cache \
  --feature-root SELECTOR_BENCHMARK_SERIES/features \
  --output-root SELECTOR_BENCHMARK_SERIES/results/screen
```

要求每个 method 单独输出：

```text
metrics.json
per_fold.csv
predictions.parquet/csv
train_log.json
```

## Screen 聚合

```bash
python SELECTOR_BENCHMARK_SERIES/aggregate_selector_results.py \
  --input-root SELECTOR_BENCHMARK_SERIES/results/screen \
  --output SELECTOR_BENCHMARK_SERIES/results/SCREEN_LEADERBOARD.csv
```

## 自动选 finalists

```bash
python SELECTOR_BENCHMARK_SERIES/select_finalists.py \
  --leaderboard SELECTOR_BENCHMARK_SERIES/results/SCREEN_LEADERBOARD.csv \
  --top-k 6 \
  --max-ade-degradation-pct 1.0 \
  --max-fde-degradation-pct 1.0 \
  --max-collision-increase-pp 0.10 \
  --output SELECTOR_BENCHMARK_SERIES/results/finalists.json
```

输出格式：

```json
{
  "methods": [
    "mlp_pairwise_g0124",
    "mlp_residual_g0124"
  ]
}
```

---

# 12. 夜间运行可靠性要求

本地 Agent 实现时必须满足：

```text
1. 每个实验独立进程
2. 每个 job 独立 log
3. 一个方法失败不能终止整个队列
4. success job 自动跳过
5. failed job 记录后继续
6. 支持 --resume
7. cache shard 可恢复
8. 所有随机种子显式记录
9. 保存 git commit SHA
10. 保存完整 config snapshot
11. GPU OOM 后记录错误，不无限自动重试
12. 最终生成 failure summary
```

不要设计“无限 while retry”。

真正的 unattended runner 应该：

```text
失败 → 记录 → 跳过 → 跑下一个
```

否则一个坏方法可能卡一整晚。

---

# 13. 第一轮之后如何决策

如果：

```text
G0/G1 已经明显降低 regret
```

说明主要是 native mode score calibration / candidate geometry 排序问题。

如果：

```text
G3 加入后明显提升
```

说明 selector 缺历史 social context。

如果：

```text
G4 加入后才出现明显跃升
```

说明 future interaction prediction 是关键。

如果：

```text
G0~G4 都无法显著闭合 oracle gap
```

再进入第二轮：

```text
G5 encoder latent
candidate-set cross attention
joint selector-generator fine-tuning
uncertainty-aware decoder
```

这样实验不会直接跳到最复杂方案。

---

# 14. 本轮最关键的科研输出

这轮实验不只是寻找一个“最好 selector”。

还应该回答：

```text
1. 最优候选是否能被 deployable information 识别？
2. Candidate intrinsic geometry 是否足够？
3. Social history 是否必要？
4. Model-predicted neighbor future 是否必要？
5. Native logit 是不是仍然包含有效先验？
6. 学习式 ranker 是否明显优于手工 rule？
7. selector bottleneck 能闭合多少 oracle gap？
```

如果这些问题能形成清晰的 feature/loss ablation，你最终得到的是一个可解释的 selector 研究结论，而不是单纯一次参数搜索。
