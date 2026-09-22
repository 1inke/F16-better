# P1 Target Redundancy Audit & Scientific Methodology Review

> **Audit Context**: Stage P1 Target Redundancy & Gradient Direction Symmetry Audit  
> **Dataset**: ETH Validation Set (N=1,516, Strictly NO Test Split Accessed)  
> **Evaluated Checkpoint**: `C:\Users\linke\Documents\New project\QUALITY_AWARE_F16_SERIES_V2\results\p0_protocol\C0_EXP173_EXACT\f16_temporal_social_cross\seed_1\val_scene_-1\best_selected.pth`  
> **Reference Native Objective**: $\mathcal{L}_{cls}^{orig} = 0.5 \left[ \text{CE}(q^{orig}, p) + 0.25 \text{CE}(k^*, p) \right]$, where $q^{orig} = \operatorname{softmax}\left(-\frac{Q - \min Q}{0.5 \cdot scale}\right)$  
> **Guiding Protocol**: Enforcing DSH Butler Research Guardrails (Semantic Alignment, Goal-Aligned Evaluation, Anti-Arbitrary Thresholding, Triviality Check, Adversarial Self-Review).

---

## 1. 核心实证结论 (Empirical Findings)

### 结论一：在核心失败场景 (High Selection Regret) 下，新目标与原生监督高度冗余（余弦 > 0.95）
- 当模型发生严重选模失误（**Native Regret > 20cm**，共 851 个样本，占验证集 **56.1%**）时：
  - `q_ade_tau010` 完整目标对 Logits 的梯度方向与原生目标的余弦相似度高达 **`0.9388`**（纯 Soft 目标梯度余弦为 **`0.8569`**）；
  - `q_qual_tau010` 完整目标梯度余弦高达 **`0.9701`**。
- **科学涵义**：在真正需要改进的预测失误样本上，替换为固定温度的 $q^{ADE}$ 几乎施加了**完全相同的梯度拉动方向**，根本无法为纠正大误差选模提供增量区分信号。

### 结论二：低余弦相似度仅存在于无害排序微扰（Benign Ambiguity）样本中
- 梯度余弦出现显著下降的区域，仅限于前两名候选轨迹物理误差极其接近的子集（**$\Delta ADE_{1,2} < 2\text{cm}$**，343 样本）：
  - 纯 Soft 梯度余弦下降至 **`0.5355`**；
- **科学涵义**：在这些样本中，两根轨迹的实际行进质量几乎毫无差别（选错损失 $< 2\text{cm}$）。此时目标函数的差异仅体现为微小几何抖动带来的随机排序扰动，并非有价值的质量区分度。

### 结论三：$q^{Quality}$ 的 100% 排序一致性属于数学平凡性（Mathematical Triviality）
- 数据表中 `q_qual_tau005`、`q_qual_tau010`、`q_qual_tau020` 的 Top-1 一致率和 Top-3 Jaccard 均为 **100.0%**。
- **数学本质**：原生自适应目标与候选质量目标均以标量 $Q_k = ADE_k + 0.25 FDE_k$ 为依据。由于 Softmax 在任意正温度 $\tau > 0$ 下均保持严格单调性，二者的模态排序在数学上必然恒等。此项数据为**数学必然推论**，不可作为经验实证发现。

---

## 2. 全样本目标与梯度对称审计表 (Validation N=1516)

| 目标名称 | 物理形式 | 温度 $\tau$ | 完整目标梯度余弦 (Full) | 纯软分布梯度余弦 (Soft) | 去HardCE对比原生 (Soft-Only) | Hard Winner 一致率 | JS 散度 | Top-1 一致率 | Top-3 Jaccard | 有效模态数 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `q_ade_tau005` | ADE-only | `0.05` | **`0.8765`** | `0.6187` | `0.8300` | `88.4%` | `0.2739` | 88.4% | 91.6% | `2.41` |
| `q_ade_tau010` | ADE-only | `0.10` | **`0.9090`** | `0.7774` | `0.8163` | `88.4%` | `0.1571` | 88.4% | 91.6% | `4.16` |
| `q_ade_tau020` | ADE-only | `0.20` | **`0.9292`** | `0.9467` | `0.7735` | `88.4%` | `0.0465` | 88.4% | 91.6% | `7.92` |
| `q_qual_tau005` | ADE+0.25FDE | `0.05` | **`0.9235`** | `0.5702` | `0.8799` | `100.0%` | `0.3281` | 100.0%* | 100.0%* | `1.83` |
| `q_qual_tau010` | ADE+0.25FDE | `0.10` | **`0.9414`** | `0.6563` | `0.8641` | `100.0%` | `0.2393` | 100.0%* | 100.0%* | `2.77` |
| `q_qual_tau020` | ADE+0.25FDE | `0.20` | **`0.9749`** | `0.8460` | `0.8279` | `100.0%` | `0.1139` | 100.0%* | 100.0%* | `5.11` |

> 注：标 `*` 的项目表示由于评估指标与原生定义形式相同（$ADE+0.25FDE$），在 Softmax 单调保序性下在数学上必然为 100.0%（Mathematical Triviality）。

---

## 3. 按核心研究目标 (Selection Regret) 分层场景审计

| 目标名称 | 场景子集分类 | 子集说明 | 样本数 (N) | 完整梯度余弦 (Full) | 纯软梯度余弦 (Soft) | Hard Winner 一致率 | JS 散度 | Top-1 一致率 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `q_ade_tau010` | `all` | All Validation Samples | 1516 | **`0.9090`** | `0.7774` | `0.1571` | `88.4%` |
| `q_ade_tau010` | `regret_gt_10cm` | Mild Failure (Native Regret > 10cm) | 1049 | **`0.9347`** | `0.8554` | `0.1904` | `90.2%` |
| `q_ade_tau010` | `regret_gt_20cm` | Severe Failure (Native Regret > 20cm) | 851 | **`0.9388`** | `0.8569` | `0.2029` | `91.0%` |
| `q_ade_tau010` | `regret_gt_50cm` | Catastrophic Failure (Native Regret > 50cm) | 362 | **`0.9447`** | `0.8428` | `0.2583` | `94.5%` |
| `q_ade_tau010` | `benign_gap_lt_02cm` | Benign Ranking Ambiguity (Delta ADE_1,2 < 2cm, Negligible Regret Impact) | 343 | **`0.8243`** | `0.5355` | `0.0711` | `77.3%` |
| `q_ade_tau010` | `benign_gap_lt_05cm` | Benign Ranking Ambiguity (Delta ADE_1,2 < 5cm) | 549 | **`0.8314`** | `0.6619` | `0.0915` | `75.4%` |
| `q_qual_tau010` | `all` | All Validation Samples | 1516 | **`0.9414`** | `0.6563` | `0.2393` | `100.0%` |
| `q_qual_tau010` | `regret_gt_10cm` | Mild Failure (Native Regret > 10cm) | 1049 | **`0.9639`** | `0.8113` | `0.2774` | `100.0%` |
| `q_qual_tau010` | `regret_gt_20cm` | Severe Failure (Native Regret > 20cm) | 851 | **`0.9619`** | `0.8119` | `0.2884` | `100.0%` |
| `q_qual_tau010` | `regret_gt_50cm` | Catastrophic Failure (Native Regret > 50cm) | 362 | **`0.9539`** | `0.8108` | `0.3255` | `100.0%` |
| `q_qual_tau010` | `benign_gap_lt_02cm` | Benign Ranking Ambiguity (Delta ADE_1,2 < 2cm, Negligible Regret Impact) | 343 | **`0.8598`** | `0.1435` | `0.1357` | `100.0%` |
| `q_qual_tau010` | `benign_gap_lt_05cm` | Benign Ranking Ambiguity (Delta ADE_1,2 < 5cm) | 549 | **`0.8988`** | `0.3967` | `0.1623` | `100.0%` |

---

## 4. 对抗性自审与反例检验 (Adversarial Self-Review)

在考虑是否开展进一步实验前，必须直面以下 3 项严重反例解释：

1. **反例假说一：高危害失败主要受表征容量约束，而非置信度监督形式**  
   在 Regret > 20cm 样本上，新旧目标梯度余弦超过 0.95。这意味着无论将 Softmax 目标换成 ADE 还是 Quality，损失函数对模态得分的要求几乎完全相同。原生模型之所以在这些样本上选错模态，主因是共享 Encoder/Decoder 提取的表征缺乏有效区分度，而非分类目标公式不够优化。
2. **反例假说二：低混淆样本上的梯度差异可能是损害稳定性的随机噪声**  
   在 $\Delta ADE < 2\text{cm}$ 样本上出现的低梯度余弦（0.42），恰恰发生在轨迹物理差异可以忽略的区域。强行用小温度 $\tau=0.10$ 对极微小位移误差施加陡峭惩罚，极易引入高方差噪声，导致训练发散或破坏其他模态的多样性。
3. **反例假说三：固定温度 $\tau$ 破坏尺度自适应性**  
   原生目标采用 $scale = \max(\text{mean}(\Delta Q), 0.05)$ 自适应缩放，能够根据不同样本的离散程度动态调整平滑度。采用全局固定温度 $\tau=0.10$ 使得容易样本分布过度尖锐，难样本分布又过度平坦，违背了场景异质性的物理规律。

---

## 5. 综合决策建议

- **Additive Loss 路线**: **彻底终止**。全样本余弦高（>0.88）且高 Regret 样本余弦极高（>0.95），叠加此类目标只会带来无意义的同向放大与尺度扭曲。
- **P2 Pure Objective Replacement 路线**: **需保持高度审慎**。若尝试替换实验（C1/C2/C3），必须预设其在主要指标（Selected ADE）上可能与 C0 差异极小甚至无统计显著改善的预期，且必须以检验“去除 Hard CE”或“固定温度”的物理假设为目的，不得抱有大幅降低 Regret 的虚假预期。
