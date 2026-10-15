# DESIGN: Motion Program Target Identifiability Audit

## 1. 科学问题与背景
在先前的连续运动程序长运行实验中，我们确立了以下核心事实：
1. **Stage 01A**: Motion Program 几何表达能力极强（Free4 Oracle 覆盖率达 95.45%，参数缺口占 80.11%）；
2. **Stage 03**: 当使用单点确定性网络回归 8 维参数时，出现了经典的“均值平滑陷阱”（Overall 覆盖率提升 +1.00 pp，但 Change 突变难例覆盖率下降 -2.37 pp，minADE 恶化 1.88%）。

为了精准判定 Change 难例平滑退化的真正根因，本诊断审计设计两个相互独立的零训练实验：
- **实验 A (Oracle 参数唯一性检验)**:
  检测同一样本、同一机动类别的连续参数解是否因内部耦合（如 onset 与 $\theta$ 补偿、速度比例与 persistence 补偿）存在多解/简并。若参数空间方差巨大而轨迹空间几乎重合，则证实将 Oracle $z$ 作为单点回归目标本身是 ill-posed 的；
- **实验 B (历史条件未来多模态检验)**:
  使用表现最好的 P2 表征，检验具有高度相似过去观测的训练集近邻样本（$k=20$），其未来 Oracle 运动程序是单峰集中还是本质多模态分叉（如转角正负分叉、骤停与直行分叉）。

---

## 2. 实验 A 设计与判据
- **样本范围**: Zara Outer 全部 Change 难例（$N=211$）及抽样 Stable 对照组（$N=211$）；
- **优化配置**: 每个样本、每个类别执行 **16 次独立随机 restarts**（160 steps，lr=0.06）；
- **近最优解筛选**: 保留所有满足 $Q(z) \le 1.01 Q_{\text{best}}$ 的解集合；
- **统计指标**:
  - $D_z$: 归一化参数空间 pairwise 距离；
  - $D_\tau$: 轨迹空间 pairwise ADE；
  - 参数间交叉相关性：$\text{Corr}(\text{onset}, \theta)$、$\text{Corr}(s, \text{persistence})$；
  - 简并度指数：$D_z / \max(D_\tau, 1e-4)$。

---

## 3. 实验 B 设计与判据
- **表征输入**: P2 终层时空编码（96 维），以训练集统计量归一化；
- **近邻检索**: 对 Zara Outer 每个 Change 难例，在合法训练集（$N=16,337$）中检索 $k=20$ 个余弦/欧氏最近邻；
- **统计指标**:
  - 邻居 GT 动作类别分布熵与主导类别占比；
  - 邻居 Oracle 未来累计转角 $\theta$ 的标准差、跨度及双峰系数（Bimodality Coefficient）；
  - 邻居终末速度比 $s_3$ 的离散度与分叉比例。

---

## 4. 后续方案分流判据（Pre-specified Routing）
1. **若 $D_z \gg 0$ 且 $D_\tau \approx 0$**（参数简并为主）：
   判定参数标签不可辨识，转向在轨迹空间通过 min-of-4 损失优化的 `BEST-OF-4 MOTION PROGRAM GENERATOR`；
2. **若邻居 future 显著多峰**（条件多模态为主）：
   转向输出参数分布的 `3-COMPONENT MDN` 并通过联合似然分配 4 个候选 slots；
3. **预算约束**:
   任何后续方案严格保持 $K=24$（Native 20 冻结 + 4 个新生成的 Program 候选）。
