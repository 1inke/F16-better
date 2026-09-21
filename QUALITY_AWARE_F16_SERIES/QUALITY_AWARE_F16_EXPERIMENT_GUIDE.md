# F16 Quality-Aware Mode Probability Learning
## 全量解冻重训练 + 多模态轨迹概率输出实验与技术执行方案

> 目标：不再增加独立 Post-hoc Selector，而是在 **F16 原模型内部**直接训练更可靠的 mode confidence / probability，使模型在一次前向中同时输出 20 条候选轨迹及其对应概率，并用概率最大轨迹作为最终 Top-1。
>
> 当前代码已确认：
>
> - F16 输出 20 条候选轨迹，形状约为 `(B, agents, 20, 12, 2)`
> - 原生 mode logits 来自 `model.last_mode_logits`，形状约为 `(B, agents, 20)`
> - Focal agent 使用 `pred[:, 0]` 和 `logits[:, 0]`
> - 当前 Native Top-1 本质上就是对原生 logits 取 `argmax`
>
> 本方案要求：**F16 全量解冻，全部参数参与训练**。同时通过 differential learning rate、原始损失保护、Oracle generation guard 和严格 reproduction gate，避免为了改善概率头而破坏候选生成能力。

---

# 1. 本轮核心问题

这次不是研究：

> “softmax 后能不能得到概率？”

这在工程上已经可以做到。

真正的问题是：

> **如果在 F16 原始端到端训练阶段，显式监督 20 个 mode logits，使高质量候选获得更高概率，能否降低 Top-1 ADE/FDE，并缩小 Native Top-1 与 Oracle minADE20 之间的差距？**

因此本轮把 candidate-selection 问题重新吸收到生成器本体中：

```text
Past trajectories / graph
        ↓
       F16
        ↓
20 trajectory candidates
+
20 native logits
        ↓
Quality-aware probability supervision
        ↓
20 mode probabilities
        ↓
argmax(probability)
        ↓
Final Top-1 trajectory
```

---

# 2. 核心训练目标

第 $k$ 个候选轨迹记为：

$$
\tau_k
$$

F16 原始 logit：

$$
s_k
$$

对应 mode probability：

$$
p_k
=
\frac{\exp(s_k)}
{\sum_{j=1}^{20}\exp(s_j)}
$$

训练阶段根据 GT 计算每条候选轨迹质量。

候选 ADE：

$$
ADE_k
=
\frac{1}{T}
\sum_{t=1}^{T}
\left\|
\tau_{k,t}-y_t
\right\|_2
$$

第一版 quality error 直接使用：

$$
e_k = ADE_k
$$

构造 soft quality target：

$$
q_k
=
\frac{
\exp(-e_k/\tau)
}{
\sum_j\exp(-e_j/\tau)
}
$$

其中 $\tau$ 为 quality temperature。

Quality loss：

$$
\mathcal{L}_{quality}
=
-\sum_{k=1}^{20}
q_k\log p_k
$$

总损失：

$$
\mathcal{L}_{total}
=
\mathcal{L}_{F16,original}
+
\lambda_q\mathcal{L}_{quality}
$$

核心思想：

> 不只告诉模型“哪个 mode 是第一名”，而是让 score head 学习整个 candidate set 的相对质量结构。

---

# 3. P0 规则：Quality Target 必须 Detach

因为 target $q_k$ 是由当前预测轨迹与 GT 的误差构造的，所以 target 本身不能参与反向传播。

必须：

```python
with torch.no_grad():
    err = torch.linalg.vector_norm(
        focal_pred - focal_gt[:, None],
        dim=-1,
    )
    candidate_ade = err.mean(dim=-1)

    target_q = torch.softmax(
        -candidate_ade / quality_tau,
        dim=-1,
    )
```

然后：

```python
log_p = torch.log_softmax(
    focal_logits,
    dim=-1,
)

loss_quality = -(
    target_q * log_p
).sum(dim=-1).mean()
```

禁止让 `candidate_ade` 对 `target_q` 的梯度继续反传。

---

# 4. 禁止重写或猜测 F16 原始 Loss

正式改代码前，Agent 必须先审计本地：

```text
FRONT_ENCODER_SERIES/train_front_variant.py
E2_F16_END_TO_END_COMPARISON_SERIES/joint_v2_adapter.py
FRONT_ENCODER_SERIES/front_encoder_models.py
```

生成：

```text
QUALITY_AWARE_F16_SERIES/TRAINING_LOSS_MAP.md
```

至少记录：

```text
Original loss component
Source file
Function
Input tensor
Loss weight
是否影响 trajectory generation
是否影响 score/logit head
```

原则：

```text
原始 F16 loss 完全保留
```

这次只新增：

```text
lambda_q * quality_loss
```

不得自行删除、替换或重写原训练目标。

---

# 5. 推荐目录

```text
QUALITY_AWARE_F16_SERIES/
├── QUALITY_AWARE_F16_EXPERIMENT_GUIDE.md
├── TRAINING_LOSS_MAP.md
├── MODEL_OUTPUT_AUDIT.md
│
├── common/
│   ├── quality_targets.py
│   ├── probability_metrics.py
│   ├── reproducibility.py
│   └── baseline_identity.py
│
├── train_quality_aware_f16.py
├── evaluate_quality_aware_f16.py
├── calibrate_temperature.py
├── run_quality_aware_series.py
│
└── results/
    ├── baseline_reproduction/
    ├── pilot/
    ├── multiseed/
    └── final/
```

---

# 6. Q0：Model Output Audit

先确认训练模式下：

```text
pred trajectories
last_mode_logits
```

都存在且顺序一致。

第一 batch 必须：

```python
assert pred.shape[-3] == 20
assert logits.shape[-1] == 20
assert pred.shape[0] == logits.shape[0]
assert pred.shape[1] == logits.shape[1]
```

预期：

```text
pred:   [B, N, 20, 12, 2]
logits: [B, N, 20]
```

---

# 7. Candidate / Logit Ordering Audit

必须确认：

> 第 $k$ 个 logit 对应第 $k$ 条候选轨迹。

保存：

```text
output_alignment_debug.csv
```

字段：

```text
sample_id
mode_id
trajectory_endpoint_x
trajectory_endpoint_y
logit
ADE
FDE
```

若发现 permutation 或错位：

```text
STOP
```

---

# 8. Q1：Baseline Reproduction Gate

先跑：

```text
lambda_q = 0
```

即：

$$
\mathcal{L}_{total}
=
\mathcal{L}_{F16,original}
$$

该实验命名：

```text
QA0_STANDARD_FINETUNE
```

要求新的训练脚本在同 seed / fold / split / optimizer / epoch 规则下能复现原 F16 fine-tuning 行为。

至少比较：

```text
Selected ADE
Selected FDE
minADE20
minFDE20
Collision Social
Native logits statistics
```

如果 `lambda_q=0` 就已经与原 F16 明显不一致：

```text
STOP
```

不得继续 quality-aware 实验。

---

# 9. Baseline Identity Lock

每个 run 都输出：

```text
baseline_identity.json
```

记录：

```text
architecture
variant
decoder
source checkpoint
seed
fold
candidate count
trainable parameter count
original loss hash
quality loss type
lambda_q
quality_tau
optimizer
learning rate groups
epochs
batch size
validation split
checkpoint rule
```

以后任何横向比较必须先比对这个文件。

---

# 10. F16 全量解冻

本轮主实验严格：

```python
for p in model.parameters():
    p.requires_grad = True
```

不允许把：

```text
encoder
trajectory decoder
score head
```

中的任何主模块冻结。

但不建议全模型用同一个大 LR。

---

# 11. Differential Learning Rate

先读取原 F16 trainer 的实际 base LR。

然后推荐：

```text
Front encoder / backbone:
0.1 × base_lr

Decoder 非 score 部分:
0.5 × base_lr

score_head:
1.0 × base_lr
```

示意：

```python
optimizer = AdamW([
    {
        "params": encoder_params,
        "lr": base_lr * 0.1,
    },
    {
        "params": decoder_non_score_params,
        "lr": base_lr * 0.5,
    },
    {
        "params": score_head_params,
        "lr": base_lr,
    },
], weight_decay=original_weight_decay)
```

不要凭空设置绝对 LR；以原 F16 的真实训练配置为基础。

---

# 12. 为什么全模型解冻但 Backbone LR 更低

我们要改善的是：

```text
candidate quality → confidence mapping
```

而不是重新训练一个完全不同的 generator。

如果 probability loss 对 backbone 拉动过强，可能出现：

```text
Top-1 ADE 下降
但 minADE20 变差
```

这不是我们要的结果。

---

# 13. 第一版只做 ADE Quality Target

文件：

```text
common/quality_targets.py
```

实现：

```python
def build_quality_target_ade(
    candidates,
    gt,
    tau,
):
    # candidates: [B, 20, T, 2]
    # gt:         [B, T, 2]

    with torch.no_grad():
        err = torch.linalg.vector_norm(
            candidates - gt[:, None],
            dim=-1,
        )

        ade = err.mean(dim=-1)

        q = torch.softmax(
            -ade / tau,
            dim=-1,
        )

    return q
```

---

# 14. 第二阶段可选：ADE + FDE

只有 ADE target 确认有信号后再测试。

定义：

$$
e_k
=
ADE_k
+
\beta FDE_k
$$

推荐只测试：

```text
beta = 0.25
```

第一轮不要大 grid。

---

# 15. 第一轮不要引入 Collision / Social Target

禁止一开始把：

```text
collision
social risk
neighbor GT
```

混入 quality target。

原因是先单独回答：

> quality-aware confidence learning 能否改善 Top-1 selection？

否则结果无法解释。

---

# 16. 总 Loss

必须是：

```python
loss_total = (
    loss_original
    + lambda_q * loss_quality
)
```

不要删除：

```text
original trajectory loss
original confidence loss
original regularization
```

如果原模型已有 score loss，本轮 quality loss 是 auxiliary loss，不是默认替换项。

---

# 17. Loss Logging

每 epoch 至少记录：

```text
loss_total
loss_original
loss_quality
```

若原 loss 可拆分，再记录：

```text
loss_traj
loss_score_original
loss_regularization
```

同时记录：

```text
mean max probability
mean probability entropy
mean oracle probability
mean near-oracle probability
```

---

# 18. Probability Entropy

定义：

$$
H(p)
=
-\sum_k p_k\log p_k
$$

用于监测 confidence collapse。

危险情况：

```text
max probability 接近 1
但 Selected ADE 没改善
```

说明模型只是变得更自信，并没有更会选。

---

# 19. Pilot 参数搜索：先 Lambda 后 Tau

不要一次做大规模二维 grid。

## P1：固定 tau

固定：

```text
tau = 0.10 m
```

测试：

```text
lambda_q = 0
lambda_q = 0.05
lambda_q = 0.10
lambda_q = 0.20
```

---

# 20. P2：固定最佳 Lambda

从 P1 选出最佳 `lambda_q` 后：

```text
tau = 0.05
tau = 0.10
tau = 0.20
```

不要继续扩展。

---

# 21. Tau 的含义

小 tau：

```text
target 接近 one-hot oracle
```

大 tau：

```text
多个 near-oracle candidate 共享概率质量
```

这正对应当前我们发现的 Top-K fine discrimination 问题。

---

# 22. 数据划分

必须复用 F16 原始训练协议。

禁止：

```text
concat 后 random 80/20
```

优先完全复用：

```text
EXP_173 / F16 原始 train / validation split
```

Test fold 永远不能参与：

```text
lambda_q selection
tau selection
checkpoint selection
temperature scaling
early stopping
```

---

# 23. Checkpoint 选择

Primary：

```text
Validation Selected ADE
```

并设置 generation guard。

推荐规则：

```text
1. Validation Selected ADE 最低
2. 若差异 < 1mm，选择 minADE20 更低者
3. 若仍接近，选择 probability NLL 更低者
```

---

# 24. 必须把 Generation 和 Selection 分开

每个 epoch validation 都算：

```text
Selected ADE
Selected FDE

minADE20
minFDE20
```

定义逐样本 selection regret：

$$
Regret_i
=
ADE_{selected,i}
-
minADE20_i
$$

最终报告：

```text
Mean Selection Regret
```

我们真正希望下降的是：

```text
Selection Regret
```

---

# 25. Generation Regression Guard

相对 QA0：

如果：

```text
minADE20 恶化 > 2%
```

或者：

```text
minFDE20 恶化 > 2%
```

标记：

```text
GENERATION_REGRESSION
```

即使 Selected ADE 有改善，也不能直接判成功。

---

# 26. Mode Diversity Guard

因为整个 F16 解冻，quality loss 可能间接造成 mode collapse。

至少报告：

```text
mean pairwise endpoint distance
mean pairwise trajectory distance
```

如果相对 QA0：

```text
diversity 下降 > 15%
```

标记：

```text
MODE_DIVERSITY_REGRESSION
```

---

# 27. Gradient Protection

沿用原 F16 gradient clipping。

如果原模型没有 clipping，可以增加：

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
```

但 QA0 和 QA1 必须完全一致。

不能：

```text
QA0 不 clip
QA1 才 clip
```

---

# 28. 推理接口

最终：

```python
output = model(...)

pred = ...
logits = model.last_mode_logits

focal_pred = pred[:, 0]
focal_logits = logits[:, 0]

prob = torch.softmax(
    focal_logits,
    dim=-1,
)

top1_idx = prob.argmax(dim=-1)

selected_traj = focal_pred[
    torch.arange(len(focal_pred)),
    top1_idx,
]
```

最终输出：

```text
20 trajectories
20 logits
20 probabilities
top1 index
top1 trajectory
```

---

# 29. Temperature Calibration

训练后可以额外使用 validation-only temperature scaling：

$$
p_k^{cal}
=
softmax(s_k/T_{cal})
$$

只改善概率 calibration。

注意：

$$
\arg\max s_k
=
\arg\max s_k/T_{cal}
$$

所以它不会改变 Top-1。

---

# 30. 三类指标必须分开

## A. Generation Quality

```text
minADE20
minFDE20
trajectory diversity
endpoint diversity
```

回答：

> 候选本身有没有变好或变坏？

## B. Selection Quality

```text
Selected ADE
Selected FDE
Selection Regret
Exact Oracle Hit
NearOracle@5cm
Pairwise@5cm
Pairwise@10cm
```

回答：

> score/probability head 是否更会选？

## C. Probability Quality

```text
Exact Oracle NLL
Soft-target Cross Entropy
Brier Score
Top-1 ECE
Mean Max Probability
Mean Entropy
```

回答：

> probability 是否更可信？

---

# 31. Exact Oracle NLL

定义：

$$
k^*
=
\arg\min_k ADE_k
$$

然后：

$$
NLL
=
-\log p_{k^*}
$$

---

# 32. Soft Quality CE

测试时固定：

```text
tau_eval = 0.10m
```

所有模型用同一个 evaluation target：

$$
q_k^{eval}
=
softmax(-ADE_k/\tau_{eval})
$$

然后：

$$
CE_{soft}
=
-\sum_k q_k^{eval}\log p_k
$$

---

# 33. Top-1 ECE

定义 correctness：

```text
Selected candidate 的 ADE 距离 Oracle minADE20 <= 5cm
```

即：

$$
ADE_{top1}-minADE20 \le 0.05
$$

用：

```text
max probability
```

作为 confidence。

这样可以回答：

> 当模型给 Top-1 80% confidence 时，它有多大概率真的选到 near-oracle candidate？

---

# 34. Safety

继续统一输出：

```text
Collision_Social
Collision_All
```

主报告采用：

```text
Collision_Social
```

不能和旧口径混用。

---

# 35. 主实验组

至少四组：

```text
Original:
现有 F16 checkpoint
不重新训练

QA0:
标准 F16 fine-tune
lambda_q = 0

QA1:
全量解冻 F16
+ ADE quality distribution loss

QA2:
Best QA1
+ validation-only temperature calibration
```

---

# 36. 最终强对照

最终 multi-seed 阶段比较：

```text
Original F16 Native Top-1
QA0 Standard Fine-tune
QA1 Quality-Aware F16
Tiny L5 Post-hoc Selector
Audited Strongest Transformer Selector
```

注意：

```text
Transformer 必须先完成 baseline identity audit
```

不能继续混用此前身份不一致的 Transformer 数字。

---

# 37. Seed 1 Pilot 成功标准

相对 QA0：

```text
Selected ADE 改善 >= 0.005m
Selection Regret 明显下降
minADE20 不恶化 > 2%
FDE 不明显恶化
Collision Social 不明显恶化
NLL / ECE 改善
```

判：

```text
STRONG_SIGNAL
```

---

# 38. 弱但值得继续

如果：

```text
Selected ADE 改善 0.002 ~ 0.005m
```

同时：

```text
NLL / ECE 显著改善
generation quality 保持
```

可以进入 3-seed。

---

# 39. 无明显价值

如果：

```text
Selected ADE 改善 < 0.002m
且
calibration 改善很小
```

则：

```text
QUALITY_PROB_WEAK
```

不要继续增加复杂 loss。

---

# 40. 失败情况

如果：

```text
Top-1 ADE 改善
但 minADE20 明显变差
```

说明出现：

```text
selection-generation tradeoff
```

优先：

```text
降低 lambda_q
```

而不是增加复杂模型。

---

# 41. Multi-Seed

Pilot 只在 Seed 1 调：

```text
lambda_q
tau
```

确定后固定超参数。

最终：

```text
Seed 1
Seed 2
Seed 3
×
5 folds
```

Seed 2/3 禁止重新调参。

---

# 42. Multi-Seed 输出

至少生成：

```text
quality_aware_multiseed.csv
quality_aware_per_fold.csv
generation_quality.csv
selection_quality.csv
probability_quality.csv
```

报告：

```text
Macro mean ± std
Micro mean ± std
Per-fold
Per-seed
```

主结论：

```text
Macro first
```

---

# 43. Statistical Test

最终 QA1 vs QA0 使用：

```text
seed-aware + fold-aware hierarchical bootstrap
```

不要把：

```text
seed × sample
```

全部当成独立 observation。

可以额外报告 micro paired bootstrap，但只能作为 secondary result。

---

# 44. 训练代码骨架

```python
model.train()

for batch in train_loader:
    past, future, num_valid, *_ = batch

    past, future = normalize_to_focal_heading(
        past,
        future,
    )

    center = past.mean(
        dim=-2,
        keepdim=True,
    )

    output = model(
        past - center,
        None,
        None,
        mode="training",
        num_valid=num_valid,
        graph_loc=past,
    )

    raw_pred = extract_pred(output)

    pred = (
        raw_pred
        + center.unsqueeze(-3)
    )

    logits = model.last_mode_logits

    focal_pred = pred[:, 0]
    focal_logits = logits[:, 0]
    focal_gt = future[:, 0]

    # 必须复用真实 F16 原始 loss
    loss_original = compute_original_f16_loss(...)

    with torch.no_grad():
        ade = torch.linalg.vector_norm(
            focal_pred - focal_gt[:, None],
            dim=-1,
        ).mean(dim=-1)

        target_q = torch.softmax(
            -ade / quality_tau,
            dim=-1,
        )

    log_p = torch.log_softmax(
        focal_logits,
        dim=-1,
    )

    loss_quality = -(
        target_q * log_p
    ).sum(dim=-1).mean()

    loss_total = (
        loss_original
        + lambda_q * loss_quality
    )

    optimizer.zero_grad()
    loss_total.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()
```

注意：

```text
compute_original_f16_loss(...)
```

必须来自真实原 trainer。

此文档只规定新增 quality loss 的位置，不允许 Agent 杜撰原始 loss。

---

# 45. 第一版只监督 Focal Agent

第一轮：

```text
Quality loss only for focal agent
```

即：

```python
focal_logits = logits[:, 0]
focal_pred = pred[:, 0]
focal_gt = future[:, 0]
```

先不要给 neighbor agents 加 quality loss。

这样更容易解释最终结果。

---

# 46. Quality Target 动态更新

每一步都基于当前：

```text
20 current candidate trajectories
```

重新和 GT 比较，生成 target distribution。

因此 target 会随 generator 变化。

但 target 必须始终：

```text
detach / no_grad
```

---

# 47. 推荐实验顺序

严格：

```text
Q0  Loss / output / alignment audit
↓
Q1  lambda=0 reproduction
↓
Q2  lambda sweep @ tau=0.10
↓
Q3  tau sweep @ best lambda
↓
Q4  best QA model + temperature calibration
↓
人工审阅
↓
Q5  Seed 1/2/3 full validation
↓
Q6  QA vs Tiny L5 vs audited Transformer
↓
FINAL DECISION
```

---

# 48. STOP POINT 1

完成：

```text
Q1 lambda=0 reproduction
```

若失败：

```text
STOP
```

不允许进入 QA1。

---

# 49. STOP POINT 2

完成 Seed 1：

```text
Q2 + Q3
```

必须输出：

```text
PILOT_ANALYSIS.md
```

然后暂停。

禁止 Agent 自动继续：

```text
3-seed full rerun
更多 loss
更多 architecture
```

---

# 50. 第一阶段最低交付物

```text
QUALITY_AWARE_F16_SERIES/
├── TRAINING_LOSS_MAP.md
├── MODEL_OUTPUT_AUDIT.md
│
└── results/
    ├── baseline_reproduction/
    │   ├── baseline_identity.json
    │   ├── reproduction_metrics.csv
    │   └── REPRODUCTION_AUDIT.md
    │
    └── pilot/
        ├── lambda_sweep.csv
        ├── tau_sweep.csv
        ├── generation_metrics.csv
        ├── selection_metrics.csv
        ├── probability_metrics.csv
        └── PILOT_ANALYSIS.md
```

---

# 51. Final Decision A：成功替代 Selector

如果 QA1：

```text
Selected ADE 明显优于 Native / QA0
Selection Regret 明显下降
minADE20 保持
probability calibration 改善
```

则：

```text
QUALITY_AWARE_CONFIDENCE_SUCCESS
```

后续可以直接：

```text
F16 → probabilities → Top1
```

而不使用独立 post-hoc selector。

---

# 52. Final Decision B：和 Tiny L5 相当

如果：

```text
QA-F16 ≈ Tiny L5
```

但 QA-F16 无额外 selector：

优先考虑 QA-F16。

因为推理结构更干净：

```text
F16
→ probability
→ Top1
```

而不是：

```text
F16
→ L5
→ Tiny selector
→ Top1
```

---

# 53. Final Decision C：Selection 不如 Tiny，但概率更可靠

如果：

```text
Top-1 ADE 不如 Tiny L5
```

但：

```text
NLL / ECE 显著改善
```

可以保留 QA-F16 的 probability 输出，用于：

```text
uncertainty estimation
downstream planning
risk-aware decision
```

而 Top-1 继续使用 Tiny L5。

---

# 54. Final Decision D：联合训练失败

如果：

```text
Top-1 无改善
minADE20 下降
diversity 下降
```

则：

```text
Quality-aware joint training 不适合当前 F16
```

停止继续堆：

```text
更复杂 probability head
更多 loss term
```

---

# 55. Agent 第一批任务

直接交给 Agent：

```text
TASK 1
审计本地 F16 原训练代码
生成 TRAINING_LOSS_MAP.md

TASK 2
生成 MODEL_OUTPUT_AUDIT.md
确认 train-mode trajectory/logit 对齐

TASK 3
新增 common/quality_targets.py

TASK 4
新增 common/probability_metrics.py

TASK 5
实现 train_quality_aware_f16.py

TASK 6
先跑 lambda_q=0 baseline reproduction

TASK 7
若 reproduction PASS：
运行 Seed1 lambda sweep

TASK 8
固定 best lambda：
运行 tau sweep

TASK 9
生成 PILOT_ANALYSIS.md 后立即停止
```

---

# 56. 严格禁令

本轮禁止：

```text
未经 QA0 reproduction 就开始 quality-aware 训练
```

禁止：

```text
test set 调 lambda / tau / checkpoint / temperature
```

禁止：

```text
只汇报 Selected ADE，不汇报 minADE20
```

禁止：

```text
Top-1 变好但 Oracle 变差仍宣布成功
```

禁止：

```text
只看到 ECE 下降就说筛选精度改善
```

禁止：

```text
把 softmax confidence 直接称为真实未来概率
```

禁止：

```text
Pilot 好看后自动扩展更多 loss / architecture
```

禁止：

```text
硬编码任何历史 baseline 数字
```

---

# 57. 本轮最核心的科学判据

真正有价值的结果不是：

> “F16 现在会输出 probability。”

因为 F16 原本已经有 logits。

真正要证明的是：

$$
\boxed{
\text{Quality-aware joint confidence learning}
\rightarrow
\text{lower selection regret}
}
$$

同时：

$$
\boxed{
\text{candidate generation quality is preserved}
}
$$

如果成立，就意味着可以把此前复杂的 candidate-selection 问题重新收回原模型内部：

```text
F16
↓
20 trajectories + 20 quality-aware probabilities
↓
argmax probability
↓
Top-1 trajectory
```

这比继续增加独立 reranker 更符合轻量化研究主线。
