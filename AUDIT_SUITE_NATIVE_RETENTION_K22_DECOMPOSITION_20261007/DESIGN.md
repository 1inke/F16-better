# 实验设计协议：Native Slot Retention, K22/K24 扩增与场景迁移鲁棒性审计

**日期：** 2026-10-07  
**实验代号：** `AUDIT_SUITE_NATIVE_RETENTION_K22_DECOMPOSITION_20261007`  
**执行环境：** Python 3.10.20, PyTorch 2.11.0+cu128, CUDA 12.8  

---

## 一、研究背景与根本问题

在前期 Stage 02 Explicit Candidate Complete-Coverage Audit 中，发现了一个剧烈反常的现象：
- 原始 `native20` 在 DEV (N=1516) 上覆盖 **733** 个窗口；
- 当采用无 GT 的几何聚类与 logit 规则剪枝为 12 条候选（`native12_control`）时，覆盖暴跌至 **532**，瞬间净损失 **201** 个达标窗口；
- 加入 8 条 explicit proposals 构成的 `hybrid20` 仅救回 48 个窗口，却丢了 177 个窗口（净覆盖仅 604），而未删剪的 `augmented28` (native20 + explicit8) 则覆盖达 781（救回 48 个，丢 0 个）。

这引发了三个核心科学问题：
1. **实验 A（Native Slot Retention Audit）**：为什么删除 8 条 native 候选会引发如此剧烈的覆盖崩塌？成功窗口内部的有效好候选结构是多模态充裕还是极度脆弱的“单候选支撑”？被删掉的关键好候选具备何种统计、排序与几何特征？是否存在 GT-free 的规则予以挽回？
2. **实验 B（K22 / K24 渐进扩增与 Null Controls）**：是否必须执着于严格 $K=20$？如果保留全部 native20，仅基于无未来信息（GT-free）的观测机动后验概率增加 2 条（$K=22$）或 4 条（$K=24$）候选，能否以极低计算开销（+10% ~ +20%）捕获 $K=28$ 带来的绝大多数 rescue 收益？同时通过复制候选（$duplicate8$）与随机微扰（$jitter8$）建立严格的 Null Controls，检验收益究竟来自“候选数量本身”还是“确定性机动提案的轨迹正交支撑”。
3. **实验 C（Stage 03 场景迁移分解鲁棒性审计）**：前期 Scene Transfer Decomposition 中 Support Ratio 计算值为 1.248，与 1.25 启发式门槛极其接近。这是否是特定超参数（$k=5$, $\text{dim}=16$）的人工产物？通过全网格灵敏度分析（PCA dim 8~48, $k=1\sim20$）与外层行人轨迹/时间块 Bootstrap，彻底澄清泛化瓶颈是纯条件映射漂移（Conditional Mapping Shift），还是支撑集漂移与映射漂移兼具的混合漂移（Mixed Shift）。

---

## 二、三项子实验科学对照设计

### 实验 A：Native Slot Retention Audit

- **CONTROL**：原始冻结 F16 输出的 `native20` 候选集合（DEV N=1516）。
- **TREATMENT**：按照 Stage 02 协议的 `choose_native_subset`（保留 Top-4 Logit + 8 条贪心最远点几何多样性候选）剪枝产生的 `native12` 集合。
- **HYPOTHESIS**：
  1. $N_{good} = \sum_{k=1}^{20} \mathbf{1}(ADE_k \le 0.2 \land FDE_k \le 0.4)$ 在成功窗口中呈严重长尾偏态，大量窗口仅有 $N_{good}=1$（单候选悬崖效应）；
  2. 201 个丢失窗口中被删掉的好候选并非尾部噪点，而是具有中高 logit 排序（Rank 5~12）且高度集中在特定模式槽位（Slot/Mode Index）的骨干候选；贪心最远点聚类由于偏爱几何极值而“跨过了”这些关键候选。
- **FALSIFICATION**：
  - 若 $N_{good} \ge 3$ 占绝大多数，且被删候选在 20 个候选内部排名极低（Rank 17~20）且离散无规律，则假设被证伪。

---

### 实验 B：K22 / K24 渐进扩增与严格 Null Controls

- **CONTROL**：
  - 基础控制：`native20`（$K=20$）。
  - 数量控制（Null Control 1）：`native20 + duplicate8`（$K=28$，复制 Top-8 native 候选）。
  - 微扰控制（Null Control 2）：`native20 + jitter8`（$K=28$，在 Top-8 native 候选上施加无 GT 的平滑时序高斯扰动 $\sigma \in \{1\text{cm}, 2\text{cm}, 5\text{cm}\}$）。
- **TREATMENT**：
  - 基于冻结 `IntentClassifier`（仅输入过去观测 motion + social）输出的机动后验概率 $P(\text{maneuver})$ 排序：
    - `K22`（$K=20 + 2$）：保留 native20，仅加入 Top-2 预测机动的 explicit proposals。
    - `K24`（$K=20 + 4$）：保留 native20，仅加入 Top-4 预测机动的 explicit proposals。
    - `augmented28`（$K=20 + 8$）：保留 native20，加入全部 8 条 explicit proposals。
- **HYPOTHESIS**：
  1. $duplicate8$ 的 rescue 必然严格为 0（证明单纯增加 $K$ 无法提升覆盖）；
  2. $jitter8$ 仅能在极少数边界样例（$ADE \approx 0.205$）产生微弱扰动增益，无法提供大幅 rescue；
  3. $K22$ 与 $K24$ 能捕获 $K=28$ 所产生的 48 个 rescue 中的大半（$\ge 50\%$ 与 $\ge 80\%$），且 lost 保持为 0。
- **FALSIFICATION**：
  - 若 $duplicate8$ 或 $jitter8$ 也能提供与 explicit 相当的 rescue，或者 $K22/K24$ rescue 捕获率不足 30%，则 explicit 机动假设不成立。

---

### 实验 C：Stage 03 Scene Transfer Decomposition 鲁棒性审计

- **CONTROL**：基线设定（PCA $\text{dim}=16$, $k=5$, $\text{seed}=42$）。
- **TREATMENT**：
  - 超参数二维网格遍历：$\text{PCA dim} \in \{8, 16, 24, 32, 48\}$, $k \in \{1, 3, 5, 10, 20\}$（共 25 组参数组合）。
  - 层次化重抽样（Bootstrap B=1000）：
    - 外层行人轨迹重抽样（151 unique pedestrian tracks）；
    - 外层录制时间块重抽样（16 unique time blocks）。
- **HYPOTHESIS**：
  - 1.248 的 Support Ratio 是特定超参数点估计的随机截断；在完整的参数网格与 Bootstrap 分布中，$R_{support}$ 将在 1.18~1.35 范围内波动，置信区间跨越 1.25；真实物理现实为**支持集漂移与映射漂移并存的混合漂移（Support + Mapping Mixed Shift）**。
- **FALSIFICATION**：
  - 若在所有 PCA 维度和 $k$ 取值下，$R_{support}$ 始终稳定在 1.05~1.15，且 $R_{conditional}$ 稳定在 1.25~1.35，则支持纯条件映射漂移假设。

---

## 三、门禁与审计原则

1. **零 GT 泄露**：候选提案选择、剪枝与机动后验概率仅依赖过去 8 帧观测历史；测试阶段未来 12 帧 GT 绝不参与任何前向决策。
2. **零模型微调**：本阶段为纯审计与无损前向实验，所有 F16、decoder 与 intent 分类器权重严格冻结。
3. **逐项算术与 ID 核查**：所有样本计数、混淆矩阵、聚类距离、Bootstrap 百分位点经独立算术审核脚本交叉验证。
