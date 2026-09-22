# F16 Frozen Candidate Rankability 综合审计与诊断报告 (P2 阶段修复版)

> **实验系列**: `F16_RANKABILITY_PROBE_SERIES`  
> **评估切分**: ETH 验证集 (N=1,516 样本，严格未访问 Test 测试集)  
> **冻结基准**: EXP-173 V2 / `f16_temporal_social_cross` / ETH / Seed 1  
> **基线事实**: 原生 Selected ADE = **`0.5389 m`** | Oracle minADE20 = **`0.2196 m`** | Selection Regret = **`0.3193 m`**  
> **裁决门禁状态**: **`[STATUS: LOW_RECOVERY_WITH_POINTWISE_PAIRWISE_PROBE]`**  
> **审计说明**: 本报告依据《F16 Rankability Probe 修正与补充实验执行文档》全面修正了前期指标定义错误（Pairwise Accuracy 方向纠偏）、降级了过度概括的结论，并对模型选择偏差进行了说明。

---

## 1. 核心实证对照总表 (Validation N=1,516)

所有探针面对**完全相同的固定 20 个候选集合**，特征归一化严格拟合自训练集。损失函数采用决策相关的 Pairwise RankNet 损失（过滤 2cm 微扰，加权 20cm 差距）：

| 方法标识 | 评分输入特征 (Feature Ladder) | 特征维度 | 验证集 Selected ADE | 验证集 Selected FDE | Selection Regret | Regret 闭合比例 (Closure) | 原生相比之配对 Bootstrap 差值 (95% CI) | Oracle Recall@1 | Oracle Recall@3 | 修正后 Pairwise Acc (差值>20cm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native R0** | `native_logits` (原有评分头) | - | **`0.5389 m`** | `1.1873 m` | `0.3193 m` | **`0.00%`** | 基准 (0.0000 m) | `20.18%` | `36.08%` | **`84.73%`** |
| **R1** | `[mode, source_anchors]` (原输入重训) | 70 | **`0.5323 m`** | `1.1745 m` | `0.3127 m` | **`+2.06%`** | `[+0.0008, +0.0120] m`* | `12.14%` | `36.41%` | **`84.93%`** |
| **R2** | `[mode, corrected_anchors]` (后微调几何) | 70 | **`0.5304 m`** | `1.1691 m` | `0.3108 m` | **`+2.66%`** | `[+0.0028, +0.0140] m`* | `11.35%` | `35.82%` | **`84.97%`** |
| **R3** | `[context, mode, corrected_anchors]` (加Context) | 134 | **`0.5301 m`** | `1.1678 m` | `0.3105 m` | **`+2.74%`** | `[+0.0022, +0.0148] m`* | `10.88%` | `27.90%` | **`84.35%`** |
| **R4** | `[context, mode, final_trajectory]` (完整几何) | 152 | **`0.5309 m`** | `1.1698 m` | `0.3113 m` | **`+2.50%`** | `[+0.0029, +0.0131] m`* | `13.13%` | `36.15%` | **`85.01%`** |
| **Oracle** | $k^* = \arg\min_k ADE_k$ (真值候选下界) | - | **`0.2196 m`** | `0.4578 m` | `0.0000 m` | **`100.00%`** | - | `100.0%` | `100.0%` | `100.00%` |

> *注：上述 Bootstrap 置信区间为在验证集最优检查点上的描述性配对 Bootstrap（descriptive paired bootstrap on the selected validation checkpoint），由于存在模型选择偏差（model-selection bias），不可直接等同于严格泛化显著性。

---

## 2. 统计检验与特征阶梯严格对比 (Pairwise Probe Comparisons)

为严格验证特征阶梯中各增量成分的独立贡献，计算了各探针之间的配对 Bootstrap 差值（1000 次重抽样）：

| 阶梯比较项 | 物理假设验证 | 改善量均值 (Mean $\Delta$) | 95% Bootstrap 置信区间 | 是否具有描述性显著性 |
| :--- | :--- | :---: | :---: | :---: |
| **R2 vs R1** | 后微调几何相对粗 Anchor 几何收益 | **`+0.0019 m`** | `[-0.0001, +0.0042] m` | **否 (跨零)** |
| **R3 vs R2** | 显式拼接 Context 特征收益 | **`+0.0003 m`** | `[-0.0033, +0.0040] m` | **否 (跨零)** |
| **R4 vs R2** | 完整轨迹 12 步坐标相对 Anchor 收益 | **`-0.0005 m`** | `[-0.0037, +0.0031] m` | **否 (跨零)** |
| **R4 vs R3** | 完整轨迹几何相对 Context+Anchor 收益 | **`-0.0008 m`** | `[-0.0053, +0.0036] m` | **否 (跨零)** |

### 关键审慎修正：
1. **R1 与 R2 之间差异不显著**：R2 相对 R1 仅取得 1.9 mm 的微弱改善，且其 95% 置信区间跨零（`p > 0.05`）。因此不能宣称“Score staleness 已被证实为造成选模失误的主要根因”。它仅是一个可观察到的局部时序错位现象。
2. **R3 的 Context 增量解释纠偏**：由于当前模型的 `mode_feature` 本身就是由 `context` 与 `mode_embedding` 通过 MLP 衍生而来，因此 R3 实际上是“显式重复拼接同一 context latent”，未带来统计增益并不代表“历史/交互 context 对 selection 无用”。

---

## 3. 指标修正：真实的 Pairwise Rankability 水平

在纠正了前期代码中 $\Delta_{ij} = ADE_j - ADE_i$ 的符号定义反转错误后，真实的候选两两排序准确率如下：
- **全对微扰过滤后 ($\Delta ADE > 2\text{cm}$)**：原生为 **`80.73%`**，探针为 **`80.1% ~ 81.1%`**；
- **决策高影响对 ($\Delta ADE > 20\text{cm}$)**：原生为 **`84.73%`**，探针为 **`84.4% ~ 85.0%`**；
- **Oracle Margin 准确率 (Oracle 对比差值 $>20\text{cm}$ 候选)**：原生为 **`72.79%`**，探针为 **`71.8% ~ 72.4%`**。

### 实证发现：
当前潜空间表征在进行**任意两根候选的粗粒度二元优劣比较**时，实际上具有约 **80%–85%** 的较高排序准确率；但在 20 选 1 的 Top-1 选择任务中，由于模态交织和局部相似候选缺乏高阶意图辨识度，Oracle Recall@1 仅停留在 11%–20%。

---

## 4. 科学推论降级与边界界定

根据当前的受控探针实证结果，我们对原报告中过度概括的表述进行严格降级：

1. **当前 pointwise frozen probes 对 selection gap 的恢复能力很低**（最佳探针仅闭合 2.74% 的 Regret）。
2. **这些结果削弱了“简单 score-head retraining 即可解决问题”的假设**：无论使用何种特征组合（粗几何、精微调几何、重复 context 或完整轨迹），低容量独立 MLP 探针均无法有效降低选模误差。
3. **因果排除尚不完整**：在完成 Top-1 对齐损失（如 Oracle-vs-Rest / Listwise CE）以及候选集感知架构（Set-Aware Ranker）前，**尚不能严格区分 objective mismatch、ranker architecture limitation 与 representation limitation**。

因此，当前状态严格标记为：
`[STATUS: INCONCLUSIVE_NEEDS_TOP1_ALIGNED_PROBE]`。
继续进入 Top-1 对齐与集合感知诊断（R5/R6/R7）。
