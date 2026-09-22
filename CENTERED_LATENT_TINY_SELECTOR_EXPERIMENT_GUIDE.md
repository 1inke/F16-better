# Centered Latent Audit + Tiny L5 Selector：实验设计与具体执行文档

> Repository: `1inke/trajectory-candidate-selection`  
> Current status: Post-hoc selector / Local reranker / raw latent probe 已进入明显平台期。  
> 本轮目标：完成 candidate-selection 分支最后一轮高价值验证，不再扩展 selector zoo。

---

# 0. 本轮核心问题

当前已经得到两个重要事实：

1. F16 内部 L3/L4/L5 确实是 sample-conditioned；
2. raw latent 对 Top-K fine discrimination 的提升很弱。

当前最值得最后验证的可能性是：

> latent 中大量 sample-common context 成分过强，掩盖了真正区分不同 candidate mode 的相对信息。

因此本轮分成两条主线：

```text
EXP-C1 ~ C4:
Centered Latent Audit
        +
EXP-T1 ~ T4:
Tiny L5 Selector 3-Seed Robustness & Efficiency Validation
```

最终目标不是继续调参，而是给出 candidate-selection 分支的“继续 / 封板”决策。

---

# 1. 总体执行顺序

```text
C1  Centered Latent Construction
        ↓
C2  Centered Latent Statistical Audit
        ↓
C3  Global Centered Latent Probe
        ↓
C4  Local Top-K Centered Latent Probe
        ↓
Centered-Latent Decision Gate
        ↓
T1  Tiny L5 Selector 3-Seed Accuracy Validation
        ↓
T2  Efficiency Benchmark
        ↓
T3  Statistical Robustness / Fold Stability
        ↓
T4  Final Tiny-Selector Decision
```

---

# 2. 新目录结构

建议新增：

```text
SELECTOR_RANKABILITY_SERIES/
└── S2V3_CENTERED_LATENT/
    ├── build_centered_latents.py
    ├── run_c2_centered_variance_audit.py
    ├── run_c3_global_centered_probe.py
    ├── run_c4_local_centered_probe.py
    ├── run_t1_tiny_selector_multiseed.py
    ├── run_t2_efficiency_benchmark.py
    ├── run_t3_fold_stability.py
    ├── run_centered_latent_series.py
    └── results/
```

不要修改已有 S2-v2 原始结果。

---

# 3. EXP-C1：Centered Latent Construction

对于某个 sample $i$ 的 $K=20$ 个 mode latent：

$$
z_{i,k}\in\mathbb{R}^{D}
$$

先计算 sample 内 mode mean：

$$
ar z_i=rac{1}{K}\sum_{k=1}^{K}z_{i,k}
$$

再构造 centered latent：

$$
z'_{i,k}=z_{i,k}-ar z_i
$$

定义：

```text
L3c = centered L3
L4c = centered L4
L5c = centered L5
```

当前 L3 已表现出：

```text
same-mode cross-sample cosine ≈ 0.524
different-mode same-sample cosine ≈ 0.905
```

这说明同一 sample 内不同 mode latent 高度相似。可假设：

$$
z_{i,k}=c_i+m_{i,k}
$$

其中：

```text
c_i   = sample-common context
m_i,k = mode-specific residual
```

selector 真正需要的是 $m_{i,k}$，因此 centering 的目的就是去掉 sample-shared context。

---

# 4. C1 实现

文件：

```text
build_centered_latents.py
```

读取：

```text
S2V2_CONTEXT_LATENT/latent_cache_v2/
```

输出：

```text
S2V3_CENTERED_LATENT/centered_cache/
seed_1/
    eth/
    hotel/
    univ/
    zara01/
    zara02/
```

每 fold 保存：

```text
l3_centered.npy
l4_centered.npy
l5_centered.npy
manifest.json
```

实现：

```python
def center_modes(z):
    # z: [N, K, D]
    mean = z.mean(axis=1, keepdims=True)
    return z - mean
```

必须验证：

```text
abs(centered.mean(axis=1)).max() < 1e-6
```

并输出：

```text
centered_alignment.json
```

验证：

```text
sample count
fold order
K=20
candidate order
sample_id
```

全部一致。

---

# 5. EXP-C2：Centered Latent Statistical Audit

对以下全部计算：

```text
L3 / L3c
L4 / L4c
L5 / L5c
```

指标：

```text
same-mode cross-sample cosine
different-mode same-sample cosine
within-mode variance
between-mode variance
mean pairwise mode distance
mode centroid distance
```

新增 Relative Mode Separation：

$$
S_{rel}=1-\operatorname{CosSim}_{different-mode,same-sample}
$$

原始 L3：

```text
Cos ≈ 0.905
S_rel ≈ 0.095
```

如果 centered 后 $S_{rel}$ 显著增大，说明 mode-specific component 被成功放大。

再定义：

$$
E_{sample}=\|ar z_i\|_2
$$

$$
E_{mode}=rac1K\sum_k\|z_{i,k}-ar z_i\|_2
$$

报告：

$$
R_E=rac{E_{mode}}{E_{sample}}
$$

输出：

```text
centered_variance_audit.json
centered_variance_per_fold.csv
RESULT_ANALYSIS_C2.md
```

---

# 6. EXP-C3：Global Centered Latent Probe

保留基线：

```text
Native G0                         0.5184m
Global Transformer Residual      0.5077m
G0+G1+raw L5 Linear              0.5075m
```

只测试：

```text
G0 + L3c
G0 + L4c
G0 + L5c
G0 + G1 + L3c
G0 + G1 + L4c
G0 + G1 + L5c
G0 + G1 + raw L5
```

模型优先：

```text
Linear Residual
```

仅对最优 centered latent 可补一个：

```text
MLP Residual
```

不要扩展新模型。

协议：

```text
Outer: 4 folds train / 1 fold test
Inner: 3 folds train / 1 fold val
```

只调：

```text
alpha ∈ {0.05,0.10,0.20,0.30,0.50,0.75,1.00,1.50}
```

输出：

```text
ADE Macro
ADE Micro
FDE Macro
ADE Regret
Oracle Hit
NearOracle@5cm
Pairwise@5cm
Pairwise@10cm
Collision Social
Mean Alpha
Latency
Params
```

判据：

```text
ADE >= 0.505  → weak / no gain
ADE < 0.500   → meaningful new signal
ADE < 0.480   → strong signal
```

---

# 7. EXP-C4：Local Top-K Centered Latent Probe

测试：

```text
K = 3 / 5 / 10
```

重点：

```text
K = 5
```

每个 K 只跑：

```text
G0 + GL + L5c
G0 + G1 + GL + L5c
```

必要时补：

```text
G0 + G1 + GL + L3c
```

模型：

```text
Linear Residual
```

只有 centered L5 出现明显提升时，再补：

```text
Tiny Set Attention
```

指标：

```text
Selected ADE
Local Pairwise@5cm
Local Pairwise@10cm
Local Spearman
Oracle Hit
NearOracle@5cm
False Intervention
Useful Intervention
Collision Social
```

当前 Native Top-5：

```text
Pairwise@5cm ≈ 64.28%
```

raw latent：

```text
≈ 64.37%
```

判断：

```text
< 66%   → 无明显新信号
>= 70%  → centered latent 有明确 fine-discrimination signal
>= 75%  → 强信号
```

---

# 8. Centered Latent Decision Gate

如果：

```text
Global centered ADE >= 0.505
且
Top5 Pairwise@5cm <= 66%
```

则：

```text
Centered Latent hypothesis rejected
```

此时正式结束：

```text
candidate fine-ranking signal mining
```

不要再做：

```text
更深 MLP
更大 Transformer
更多 latent 拼接
```

只有当：

```text
Global ADE < 0.500
或
Top5 Pairwise@5cm >= 70%
```

才允许进入：

```text
S2V4 Tiny Centered Latent Reranker
```

---

# 9. EXP-T1：Tiny L5 Selector 三 Seed 稳健性验证

这一条与 Centered Latent 是否成功无关。

当前候选：

```text
G0 + G1 + raw L5
Linear Residual
```

Seed 1：

```text
ADE     = 0.5075m
Params  = 86
Latency ≈ 0.127ms
```

目标：

> 验证它是否能稳定替代 Global Transformer Residual。

必须比较：

```text
A. Native
B. Global Transformer Residual (G0+G1+G2+G4)
C. Tiny L5 Linear Residual (G0+G1+L5)
D. Global Linear Residual（可选）
```

Seeds：

```text
1 / 2 / 3
```

如果 candidate cache / latent cache 按 seed 不同，则必须使用对应 seed 的 cache。

禁止：

```text
Seed 2/3 复用 Seed 1 latent
```

---

# 10. T1 每 Seed 输出

每种方法：

```text
ADE Macro
ADE Micro
FDE Macro
ADE Regret
Oracle Hit
NearOracle@5cm
Pairwise@10cm
Collision Social
Intervention Rate
False Intervention
Latency
Params
```

最终同时输出：

```text
mean ± std
per-fold × per-seed
```

主结论以 Macro 为准。

---

# 11. Tiny Selector Accuracy 判据

定义：

$$
\Delta ADE=ADE_{Tiny}-ADE_{Transformer}
$$

判据：

```text
|ΔADE| <= 0.001m
→ 近似等精度

|ΔADE| <= 0.002m
→ 仍可作为轻量替代候选

Tiny 比 Transformer 差 > 0.003m
→ 轻量替代价值明显下降
```

---

# 12. EXP-T2：Efficiency Benchmark

比较：

```text
Global Transformer Residual
Tiny L5 Linear Residual
```

必须测：

```text
Parameters
MACs / FLOPs
Peak memory
CPU latency
GPU latency
Batch-1 latency
Batch-64 latency
```

Timing Protocol：

```text
warmup = 100
repeat = 1000
```

GPU 前后：

```python
torch.cuda.synchronize()
```

报告：

```text
mean
median
p95
```

---

# 13. Selector-only 与 End-to-End 延迟分开

必须分别报告：

```text
Selector-only latency
```

以及：

```text
F16 + selector total latency
```

因为 L5 本身来自 F16 内部，不应重复计算 latent 提取成本。

---

# 14. 参数压缩率

定义：

$$
Compression=1-rac{P_{tiny}}{P_{transformer}}
$$

当前理论上约：

```text
86 vs 52,865 params
```

但最终必须由脚本动态输出，不手写。

---

# 15. EXP-T3：Fold Stability

三 seed 后逐 fold 比较：

```text
ETH
Hotel
Univ
Zara1
Zara2
```

定义：

$$
\Delta ADE_f=ADE^{Tiny}_f-ADE^{Transformer}_f
$$

重点看 Tiny 是否只在 Univ 上好，而在小 fold 上明显退化。

理想条件：

```text
至少 4/5 folds:
|ΔADE_fold| <= 0.005m
```

如果：

```text
某一 fold 差 > 0.02m
```

必须单独分析。

---

# 16. Macro / Micro 规则

由于 Univ 样本量远大于其他 fold：

```text
最终主指标 = Macro
Micro = secondary
```

禁止仅用 Micro 证明 Tiny 更优。

---

# 17. EXP-T4：Accuracy-Efficiency Pareto

最终输出：

```text
Method
ADE
FDE
Collision Social
Params
FLOPs
Latency
Peak Memory
```

至少：

```text
Native
Global Linear
Global Transformer
Tiny L5 Linear
```

---

# 18. Tiny Selector 最终通过标准

如果满足：

```text
3-seed Macro ADE difference <= 1~2mm
FDE 不明显恶化
Collision Social 不明显恶化
参数减少 > 99%
Selector latency 显著下降
fold stability acceptable
```

则可以得出：

> Tiny L5 linear residual can serve as an efficient replacement for the considerably larger Transformer reranker with negligible accuracy loss.

注意：即使单 seed `0.5075 < 0.5077`，也不要直接写：

```text
Tiny outperforms Transformer
```

优先写：

```text
matches
approaches
retains comparable accuracy
```

---

# 19. Paired Bootstrap

保存 per-sample prediction 后，对：

```text
ADE_tiny - ADE_transformer
```

做 paired bootstrap：

```text
10,000 resamples
```

输出：

```text
mean ΔADE
95% CI
```

如果 CI 跨 0：

```text
不能声称显著优于
```

但如果区间足够窄，可写：

```text
accuracy difference is statistically negligible / indistinguishable within the tested protocol
```

---

# 20. Prediction Cache

必须保存：

```text
sample_id
fold
seed
native_idx
transformer_selected_idx
tiny_selected_idx
native_ade
transformer_ade
tiny_ade
oracle_ade
```

用于：

```text
paired bootstrap
fold analysis
case study
```

---

# 21. 新 Orchestrator

新增：

```text
run_centered_latent_series.py
```

支持：

```text
--stage c1
--stage c2
--stage c3
--stage c4
--stage t1
--stage t2
--stage t3
--stage all
```

Resume：

```text
success → skip
failed → stop current stage
--force → rerun
```

不要自动进入：

```text
Large Probe
S2V4
Joint Training
```

---

# 22. Agent 第一批任务

直接交给 Agent：

```text
TASK 1
创建 S2V3_CENTERED_LATENT 目录

TASK 2
实现 build_centered_latents.py

TASK 3
实现 C2 statistical audit

TASK 4
实现 C3 global centered probe

TASK 5
实现 C4 local centered probe

TASK 6
输出 centered-latent verdict

TASK 7
构建 Seed 2/3 对应 latent cache

TASK 8
实现 T1 Tiny-vs-Transformer 3-seed benchmark

TASK 9
实现 T2 efficiency benchmark

TASK 10
实现 T3 fold stability + paired bootstrap

TASK 11
输出最终 synthesis report 后停止
```

---

# 23. 最低交付物

```text
S2V3_CENTERED_LATENT/results/
├── centered_alignment.json
├── centered_variance_audit.json
├── centered_global_probe.csv
├── centered_local_probe.csv
├── tiny_selector_multiseed.csv
├── tiny_selector_per_fold.csv
├── efficiency_benchmark.csv
├── paired_bootstrap.json
├── FINAL_TINY_SELECTOR_REPORT.md
└── state.json
```

---

# 24. Final Decision Tree

## Branch A：Centered Latent 成功

如果：

```text
Global ADE < 0.500
或
Top5 Pairwise@5cm >= 70%
```

则：

```text
继续 S2V4 centered latent selector
```

---

## Branch B：Centered Latent 失败，但 Tiny 稳定

如果：

```text
Centered:
ADE >= 0.505
Top5 Pairwise <= 66%

同时 Tiny:
与 Transformer ADE 差 <= 1~2mm
```

则：

```text
candidate fine-ranking science branch 封板
Tiny L5 selector 作为轻量化工程/模型贡献保留
```

这是当前最可能、也最有价值的结果。

---

## Branch C：Centered 与 Tiny 都失败

如果：

```text
Centered 无突破
Tiny 3-seed 明显差于 Transformer > 3mm
```

则：

```text
整个 post-hoc candidate selector 分支停止
```

回到主模型结构轻量化方向。

---

# 25. 推荐最终研究表述

如果 Branch B 成立，可以形成以下结论：

> Multi-hypothesis candidate coverage is substantially stronger than top-1 recoverability, and neither post-hoc local reranking nor raw/centered decoder latents provide a large fine-ranking breakthrough. However, the decoder's pre-logit latent can be reused by an extremely small linear residual calibrator to reproduce the accuracy of a much larger Transformer reranker at a fraction of the computational cost.

它同时包含：

```text
科学负结果：
fine-ranking information difficult to recover

工程正结果：
complex selector unnecessary
```

并与轻量化主线直接一致。
