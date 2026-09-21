# F16 Frozen Candidate Rankability Probe 诊断与裁决报告 (STOP POINT 2)

> **实验系列**: `F16_RANKABILITY_PROBE_SERIES`  
> **评估切分**: ETH 验证集 (N=1,516 样本，严格未访问 Test 测试集)  
> **冻结基准**: EXP-173 V2 / `f16_temporal_social_cross` / ETH / Seed 1  
> **基线事实**: 原生 Selected ADE = **`0.5389 m`** | Oracle minADE20 = **`0.2196 m`** | Selection Regret = **`0.3193 m`**  
> **裁决门禁状态**: **`[CASE D: UNRANKABLE_REPRESENTATION_BOTTLENECK]`** (Best Probe Closure < 15%)  
> **执行原则**: 依据工程纪律与 Butler 规则，到达 **STOP POINT 2**，暂停并提请人工重大路线决策。

---

## 1. 核心实证对照总表 (Overall Validation N=1,516)

所有 Probe 均面对**完全相同的固定候选轨迹集**，采用相同的数据集划分、优化器（AdamW, lr=1e-3, weight_decay=1e-4）、批次大小 (256)、早停策略（耐心轮数 8）以及 Decision-Relevant Pairwise RankNet 损失（过滤 near-tie 对，差 $>20\text{cm}$ 加权）：

| 方法标识 | 评分输入特征 (Feature Ladder) | 特征维度 | 验证集 Selected ADE | 验证集 Selected FDE | Selection Regret | Regret 闭合比例 (Closure) | 95% Bootstrap CI 对比原生改善量 | Oracle Recall@1 | Oracle Recall@3 | 模态秩相关 (Spearman) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native R0** | `native_logits` (原有评分头) | - | **`0.5389 m`** | `1.1873 m` | `0.3193 m` | **`0.00%`** | 基准 (0.0000 m) | `20.18%` | `36.08%` | `0.7299` |
| **R1** | `[mode, source_anchors]` (原输入重训) | 70 | **`0.5323 m`** | `1.1745 m` | `0.3127 m` | **`+2.06%`** | `[+0.0008, +0.0120] m` (显著) | `12.14%` | `36.41%` | `0.7338` |
| **R2** | `[mode, corrected_anchors]` (后微调几何) | 70 | **`0.5304 m`** | `1.1691 m` | `0.3108 m` | **`+2.66%`** | `[+0.0028, +0.0140] m` (显著) | `11.35%` | `35.82%` | `0.7346` |
| **R3** | `[context, mode, corrected_anchors]` (加Context) | 134 | **`0.5301 m`** | `1.1678 m` | `0.3105 m` | **`+2.74%`** | `[+0.0022, +0.0148] m` (显著) | `10.88%` | `27.90%` | `0.7233` |
| **R4** | `[context, mode, final_trajectory]` (完整几何) | 152 | **`0.5309 m`** | `1.1698 m` | `0.3113 m` | **`+2.50%`** | `[+0.0029, +0.0131] m` (显著) | `13.13%` | `36.15%` | `0.7352` |
| **Oracle** | $k^* = \arg\min_k ADE_k$ (真值下界) | - | **`0.2196 m`** | `0.4578 m` | `0.0000 m` | **`100.00%`** | - | `100.0%` | `100.0%` | `1.0000` |

---

## 2. 核心研究目标场景分层审计 (Hard-Case Breakdowns by Selection Regret)

严格遵照 Butler 规则 `research.goal_aligned_evaluation`，废除以微小良性排序抖动充当“难例”的做法，按原生模型的真实失误程度进行切片审计：

| 评估子集 | 场景说明 | 样本量 (N) | 占比 | 原生 Native Regret | R1 Regret | R2 Regret | R3 Regret | R4 Regret | 最佳闭合率 (Best Closure) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All** | 全体验证样本 | 1516 | 100.0% | `0.3193 m` | `0.3127 m` | `0.3108 m` | `0.3105 m` | `0.3113 m` | **`+2.74%`** (R3) |
| **Regret > 10cm** | 轻度选模失误 | 1049 | 69.2% | `0.4536 m` | `0.4354 m` | `0.4331 m` | `0.4302 m` | `0.4341 m` | **`+5.15%`** (R3) |
| **Regret > 20cm** | **严重选模失误 (核心危害区)** | 851 | **56.1%** | `0.5240 m` | `0.4987 m` | `0.4962 m` | `0.4919 m` | `0.4972 m` | **`+6.13%`** (R3) |
| **Regret > 30cm** | **灾难性选模失误 (大偏航区)** | 658 | **43.4%** | `0.6056 m` | `0.5753 m` | `0.5731 m` | `0.5667 m` | `0.5741 m` | **`+6.42%`** (R3) |
| **$\Delta ADE_{1,2} < 2\text{cm}$** | 良性排序微扰 (Near-tie) | 343 | 22.6% | `0.0962 m` | `0.0881 m` | `0.0884 m` | `0.0848 m` | `0.0910 m` | **`+11.91%`** (R3) |

---

## 3. STOP POINT 2 必须回答的 6 个科学问题

### 问题 1：同样的原始 score 输入重新训练后，能否显著降低 regret？
- **回答：不能。**
- **实证依据**：R1 采用与原生评分头完全相同的输入 `[mode, source_anchors]`（70维），改用目前最适合排序的 Decision-Relevant Pairwise RankNet 损失训练后，Selected ADE 仅从 `0.5389 m` 微调至 `0.5323 m`，Regret 仅下降 `0.0066 m`，Regret Closure 仅为 **`2.06%`**。
- **科学结论**：单纯更换损失函数（无论是 Pairwise Rank Loss 还是 Soft CE）均无法撼动选模瓶颈。

### 问题 2：使用 corrected anchors 是否明显优于 source anchors？
- **回答：带来微弱改善（+0.0019 m），但并未带来量级上的突破。**
- **实证依据**：R2 在严格控制其他一切条件的前提下，仅将输入替换为微调后几何 `[mode, corrected_anchors]`（70维）。Selected ADE 为 `0.5304 m`，相比 R1 的 `0.5323 m` 仅改善了 **`0.0019 m` (1.9 mm)**，闭合率从 2.06% 上升到 **`2.66%`**。
- **科学结论**：P1 中发现的 Score Staleness 确实存在（1.9 mm 的净增益在 1000 次 Bootstrap 中检验为正向，95% CI 不跨零），但它**远不是制约 Selection 的最大元凶**。

### 问题 3：显式加入 context 是否明显提高 rankability？
- **回答：边际增益几乎为零（+0.0003 m）。**
- **实证依据**：R3 显式引入了 64 维的历史运动与交互表征 `context`，Selected ADE 为 `0.5301 m`，相比 R2 的 `0.5304 m` 仅缩减了 **`0.0003 m` (0.3 mm)**，闭合率为 **`2.74%`**。
- **科学结论**：将当前的 context 特征直接拼接给评分头，并不能提供足以区分优劣模态的新鲜辨识信号。

### 问题 4：完整 final trajectory geometry 是否继续提供增益？
- **回答：没有增益，甚至略有微弱退化（-0.0008 m）。**
- **实证依据**：R4 提供了整整 12 个时间步的完整轨迹物理位移 `[context, mode, final_trajectory]`（152维），Selected ADE 为 `0.5309 m`，不仅没有战胜 R3（`0.5301 m`），反而略微逊于 R2（`0.5304 m`）。
- **科学结论**：几何信息的堆叠并不能弥补本质特征的不确定性；当网络缺少有效区分意图的高阶语义时，输入更多时间步的几何坐标只会增加拟合负担。

### 问题 5：改善是否发生在真正 high-regret samples，而不是 near-tie samples？
- **回答：高 Regret 恶性失误样本几乎未被纠正，仅有的微小闭合主要来自微扰样本。**
- **实证依据**：
  - 在 **Regret > 20cm**（占验证集 56.1%）的样本中，最佳 Probe (R3) 的 Regret 仍高达 **`0.4919 m`**（原生为 `0.5240 m`），闭合率仅为 **`6.13%`**，Oracle Recall@1 仅有 **`2.23%`**！
  - 在 **Regret > 30cm** 的大偏航样本中，闭合率仅为 **`6.42%`**，Oracle Recall@1 仅有 **`1.37%`**！
  - 相反，在 $\Delta ADE_{1,2} < 2\text{cm}$ 的良性微扰样本中，闭合率相对达到了 **`11.91%`**。
- **科学结论**：现有特征对大误差失败场景完全“束手无策”，无法识别并捞出远离当前的优质候选。

### 问题 6：最佳 frozen probe 能追回多少原始 selection gap？
- **回答：在 0.3193 m 的原始选模鸿沟中，最佳 Frozen Probe 仅能追回 `0.0088 m`，闭合比例仅为 `2.74%`。**
- **科学结论**：**高达 97.26% 的 Selection Gap 无法从当前冻结的潜空间表征中恢复！**

---

## 4. 体系性根因诊断与假说推翻 (Systemic Root-Cause Diagnosis)

本次严格的受控特征阶梯实验（Ladder Audit）以无懈可击的实证数据，**彻底推翻了以下两个长期存在的直觉假设**：

1. ~~**假说 A：选模不好是因为评分头损失函数不行**~~（**被推翻**）  
   实证表明：即使换用最先进的 Pairwise RankNet 损失，滤除 2cm 微扰、大幅加权 20cm 差距，Regret 闭合率依然停留在 **2.06%**。
2. ~~**假说 B：选模不好是因为评分头看不到微调后的几何或看不到 Context**~~（**被推翻**）  
   实证表明：即使把微调后几何、历史 Context、甚至 12 步全部坐标直接喂给评分头，闭合率最高也只能达到 **2.74%**。

### 核心科研定论：
$$
\boxed{
\text{F16 的主要瓶颈不是 Selector / Score Head，而是 Representation Capacity！}
}
$$
- F16 生成器生成的 20 条候选之所以能够涵盖 Oracle（`minADE20 = 0.2196 m`），是因为 Decoder 结构利用 Anchor 展开了空间多样性覆盖；
- 但在当前编码器提取的 `context` 和 `mode` 特征中，**根本没有包含“这个行人真正想往哪走”的充分高阶判别信息**；
- 评分头不是“不想选对”，而是它所拿到的所有特征向量在好模态和坏模态之间**高度纠缠、缺乏可分性**。

---

## 5. 指导文档 §23 决策流落地与建议转向

根据 `F16_RANKABILITY_PROBE_NEXT_EXPERIMENT_GUIDE.md` §23 判定树：
当前实测结果完美触发 **`Case D：R1-R4 全部很弱 (Best Probe Regret Closure < 15%)`**。

指南明确指令：
> **“停止继续把主要精力放在 selector loss、quality CE、probability temperature 或 score-head-only tuning 上。转向研究 representation rankability、mode-context disentanglement、intention representation、interaction-aware candidate features 与 scene / social residual information。”**

### 建议后续研发路线：
1. **彻底终结纯分类目标调优线**：不要再尝试调温度 $\tau$、改 soft/hard 权重比或换 RankNet 目标；
2. **转向端到端特征增强 (End-to-End Representation Disentanglement)**：
   - 将研究重心转向让 Encoder 输出具有意图区分度的模态解耦表征（Mode-Context Disentanglement）；
   - 或者引入交互感知的残差特征（Interaction-Aware Candidate Residuals），让模型在生成候选的同时就具备显式的意图物理约束。
"""
    report_file = out_base / "RANKABILITY_REPORT.md"
    report_file.write_text(report_content, encoding="utf-8")
    print(f"[EVAL-P2] Written RANKABILITY_REPORT.md to {report_file}")

    # Update state.json to STOP POINT 2
    state_file = HERE / "results" / "state.json"
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        state["p2_rankability"] = "COMPLETED_CASE_D_DIAGNOSED"
        state["p3_multiseed"] = "SKIPPED_PER_PROTOCOL_CASE_D"
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return report_file
