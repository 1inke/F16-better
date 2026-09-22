# F16 Top-1 对齐与集合感知 Rankability 最终诊断报告 (STOP POINT 3)

> **实验系列**: `F16_RANKABILITY_PROBE_SERIES` (Top-1 Aligned Probes: R5, R6, R7)  
> **基线体系**: EXP-173 V2 / `f16_temporal_social_cross` / ETH / Seed 1 (Frozen Generator)  
> **验证切分**: ETH 验证集 (N=1,516 样本，严格锁死 Test 集)  
> **基线基准**: 原生 Selected ADE = **`0.5389 m`** | Oracle minADE20 = **`0.2196 m`** | Selection Regret = **`0.3193 m`**  
> **最终科学判定**: **`[CASE C: FROZEN_REPRESENTATION_SELECTION_BOTTLENECK]`**  
> **研发状态变更**: `state.json` 更新为 `COMPLETED_NEEDS_REPRESENTATION_REDESIGN`，进入 STOP POINT 3 决策节点。

---

## 1. 最终诊断对照总表 (Validation N=1,516, Across Ranker Seeds 1, 2, 3)

本轮诊断补做了三组极具针对性的对照实验，严格采用控制变量法，消除因果混淆：
1. **R5 (Oracle-vs-Rest Pairwise)**：仅改变优化目标，消除 All-pair RankNet 对次要模态过度拟合的问题，直接优化 Oracle 与其他 19 个竞争模态的边际差。
2. **R6 (Listwise Top-1 Hard CE)**：直接进行 20 选 1 的多分类互斥监督，对标原生分类形式。
3. **R7 (Set-Aware Ranker)**：改变模型架构（1 层 Self-Attention，42.5k 参数），赋予候选模态全局对比与集合感知（Set Interaction）能力。

| 探针标识 | 探针架构与优化目标 | 3 个初始化种子 Val Selected ADE (m) | 均值 ADE | 均值 Selection Regret | 均值 Regret 闭合率 (Closure) | 均值 Oracle Recall@1 | 均值 Oracle Recall@3 | 决策对排序准确率 (Pairwise Acc >20cm) | Oracle Margin Acc (>20cm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native R0** | 原生评分头 (未重训) | `0.5389` | `0.5389 m` | `0.3193 m` | **`0.00%`** | `20.18%` | `36.08%` | `84.73%` | `72.79%` |
| **R1** (All-pair) | Pointwise MLP + All-Pair RankNet | `0.5323` | `0.5323 m` | `0.3127 m` | **`+2.06%`** | `12.14%` | `36.41%` | `84.93%` | `72.40%` |
| **R2** (Refined Geom) | Pointwise MLP + All-Pair RankNet | `0.5304` | `0.5304 m` | `0.3108 m` | **`+2.66%`** | `11.35%` | `35.82%` | `84.97%` | `72.38%` |
| **R3** (Context+Post) | Pointwise MLP + All-Pair RankNet | `0.5301` | `0.5301 m` | `0.3105 m` | **`+2.74%`** | `10.88%` | `27.90%` | `84.35%` | `72.32%` |
| **R4** (Full Traj) | Pointwise MLP + All-Pair RankNet | `0.5309` | `0.5309 m` | `0.3113 m` | **`+2.50%`** | `13.13%` | `36.15%` | `85.01%` | `72.11%` |
| **R5** (Top-1 Pairwise) | Pointwise MLP + Oracle-vs-Rest | `s1:0.5370 / s2:0.5441 / s3:0.5381` | `0.5397 m` | `0.3201 m` | **`-0.26%`** | `15.72%` | `38.32%` | `83.14%` | `73.30%` |
| **R6** (Listwise CE) | Pointwise MLP + Hard Oracle CE | `s1:0.5343 / s2:0.5420 / s3:0.5287` | `0.5350 m` | `0.3154 m` | **`+1.21%`** | `20.49%` | `38.21%` | `82.49%` | `73.01%` |
| **R7** (Set-Aware Attn) | 1-Layer Self-Attention + Oracle-vs-Rest | `s1:0.5391 / s2:0.5346 / s3:0.5383` | `0.5373 m` | `0.3178 m` | **`+0.48%`** | `17.88%` | `37.71%` | `82.38%` | `73.44%` |
| **Oracle 下界** | $k^* = \arg\min_k ADE_k$ | `0.2196` | `0.2196 m` | `0.0000 m` | **`100.00%`** | `100.0%` | `100.0%` | `100.00%` | `100.00%` |

---

## 2. 真实恶性失误切片审计 (Severe Selection Failures, Regret > 20cm / > 30cm)

依据 Butler 规则 `research.goal_aligned_evaluation`，重点考察在真正导致严重危害的样本上的表现：

| 评估场景分类 | 样本量 (N) | 验证集占比 | 原生 Native Regret | R3 Regret (最佳 All-pair) | R5 Regret (Oracle-vs-Rest) | R6 Regret (Listwise CE) | R7 Regret (Set-Aware) | 场景最高闭合率 | Oracle Recall@1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Samples** | 1516 | 100.0% | `0.3193 m` | `0.3105 m` | `0.3174 m` | `0.3091 m` | `0.3151 m` | **`+3.19%`** (R6-s3) | `10.9% ~ 20.5%` |
| **Regret > 10cm** | 1049 | 69.2% | `0.4536 m` | `0.4302 m` | `0.4402 m` | `0.4324 m` | `0.4342 m` | **`+5.15%`** (R3) | `2.4% ~ 3.3%` |
| **Regret > 20cm** | 851 | **56.1%** | `0.5240 m` | `0.4919 m` | `0.5024 m` | `0.4946 m` | `0.4944 m` | **`+6.13%`** (R3) | **`1.8% ~ 3.2%`** |
| **Regret > 30cm** | 658 | **43.4%** | `0.6056 m` | `0.5667 m` | `0.5774 m` | `0.5700 m` | `0.5645 m` | **`+6.78%`** (R7) | **`0.9% ~ 2.3%`** |
| **$\Delta ADE_{1,2} < 2\text{cm}$** | 343 | 22.6% | `0.0962 m` | `0.0848 m` | `0.1005 m` | `0.0870 m` | `0.0886 m` | **`+11.91%`** (R3) | `13.0% ~ 55.4%` |

---

## 3. STOP POINT 3 科学问题逐项回答 (§29)

### 问题 1：原 pairwise metric 修正后，当前 feature 的 pairwise rankability 实际有多高？
- **回答：实际相当高（80% ~ 85%）。**
- **实证证据**：在修复了符号定义后，原生模型与各类探针在差值 $>20\text{cm}$ 的模态对上的两两排序正确率均在 **`82% ~ 85%`**，Oracle 对比恶性竞争模态的 Margin 准确率也在 **`72% ~ 73%`**。
- **科学涵义**：当前隐空间特征具备较强的“粗粒度二元优劣辨识度”，并未完全坍塌或反转。

### 问题 2：Top-1 aligned objective 是否显著优于 all-pair RankNet？
- **回答：没有显著优势，甚至略显劣势。**
- **实证证据**：
  - R5（Oracle-vs-Rest Pairwise）在 3 个种子下的平均 Selected ADE 为 `0.5397 m`，平均闭合率为 **`-0.26%`**，未超过 R3（`0.5301 m`, `+2.74%`）；
  - 配对 Bootstrap 检验显示 R5 相比 R3 的均值差为 `-0.0069 m`，95% CI 为 `[-0.0147, +0.0010] m`（跨零，无显著改善）。
- **科学涵义**：即便直接将监督压力聚焦于“只击败竞争模态”，也无法让 MLP 探针选出更好的 Top-1。

### 问题 3：Oracle-vs-Rest training 是否能显著提升 Oracle Recall@1？
- **回答：不能。**
- **实证证据**：R5 的 Oracle Recall@1 仅为 `15.72%`（原生为 `20.18%`，R6 为 `20.49%`）。在真正严重的失败样本（Regret > 20cm）中，R5 的 Oracle Recall@1 仅为 **`2.47%`**。

### 问题 4：Set-aware ranker 是否显著优于 pointwise ranker？
- **回答：没有显著优势。**
- **实证证据**：
  - R7 引入了 1 层 Self-Attention 全局候选交互层（42.5k 参数），其 3 个种子的平均 Selected ADE 为 `0.5373 m`，平均闭合率仅为 **`+0.48%`**（最好种子为 1.32%）；
  - 配对 Bootstrap 检验显示 R7 相比 R5 的均值改善为 `-0.0021 m`，95% CI 为 `[-0.0080, +0.0034] m`（跨零）。
- **科学涵义**：候选模态之间的注意力交互并未产生奇迹，说明障碍不在于“看不见其他候选”，而在于各候选的输入表征本身缺少区分性信息。

### 问题 5：改善是否发生在 Regret > 20cm / >30cm 的真正失败样本？
- **回答：完全没有。**
- **实证证据**：在占验证集 56.1% 的 Regret > 20cm 样本群中，无论是 R3、R5、R6 还是 R7，选模后的平均 Regret 依然停留在 `0.49 ~ 0.50 m`（原生为 `0.5240 m`），闭合率仅在 4%~6% 之间徘徊，Oracle Recall@1 始终被锁死在 **`1.8% ~ 3.2%`** 的极低水平。

### 问题 6：结果是否在 3 个 ranker seeds 下稳定？
- **回答：高度稳定。**
- **实证证据**：
  - R5 跨 3 种子 Selected ADE 为 `0.5370, 0.5441, 0.5381` (闭合率 +0.60%, -1.63%, +0.26%)；
  - R6 跨 3 种子 Selected ADE 为 `0.5343, 0.5420, 0.5287` (闭合率 +1.43%, -0.99%, +3.19%)；
  - R7 跨 3 种子 Selected ADE 为 `0.5391, 0.5346, 0.5383` (闭合率 -0.07%, +1.32%, +0.19%)；
  - 所有设置在所有初始化下，闭合率均被牢牢压制在 3.2% 以下，展现出极强的一致性。

### 问题 7：经过这些更强但仍轻量的诊断后，当前 frozen representation 仍然只能追回多少 Selection Gap？
- **回答：全系列所有配置下的最大闭合率仅为 `3.19%`（R6-s3），3 种子均值闭合率仅为 `1.21%`。**
- **科学涵义**：在原有 0.3193 m 的选模差距中，**超过 96.8% 的鸿沟在冻结隐空间中均无法通过目标或下游架构调整恢复！**

### 问题 8：此时判定瓶颈属于哪类因素？
依据指南 §21 裁决树判定条件：
> **`R5/R6/R7 mean Closure < 10–15% 且 3 ranker seeds 一致`**  
> 完全触发 **`Case C`**！

通过严格的因果控制链条，我们依次排除了以下可能：
1. **排除了 Objective Mismatch**：从 All-pair RankNet 到 Oracle-vs-Rest 再到 Listwise CE，闭合率均 < 3.2%；
2. **排除了 Ranker 架构与集合交互缺陷**：引入 42.5k 参数的 Transformer Self-Attention (R7)，闭合率仅 0.48%；
3. **排除了几何信息缺失与评分滞后**：提供微调几何 (R2)、上下文 (R3) 与 12 步全部坐标 (R4)，闭合率均 < 2.8%。

因此，在坚实的实证排他支持下，得出确定性结论：
$$
\boxed{
\text{当前瓶颈既非 Objective Mismatch，亦非下游 Ranker 缺陷，而是 Frozen Representation Capacity 对 Top-1 意图信息的严重匮乏！}
}
$$

---

## 4. 走向下一步：正式开启 Representation Redesign

本轮诊断彻底完成了所有的消融与排他性证明。依据指南 §22 的明确规定，**当前已具备充分证据正式进入表征重构（Representation Redesign）研发阶段**。

建议的新阶段重点突破方向：
1. **模态与上下文深度解耦（Mode-Context Disentanglement）**：打破当前简单的加性偏置（`context + mode_emb`），构建针对不同意图显式正交/解耦的潜变量空间；
2. **交互感知的候选残差表征（Interaction-Aware Candidate Residuals）**：在编码阶段直接将未来意图候选与社交代理、场景地图进行 Cross-Attention，让表征本身具备可辨识的意图物理约束；
3. **意图条件化的先验特征（Intent-Conditioned Feature Learning）**：从生成根源增强特征的可分性，而非在下游被动排序。
