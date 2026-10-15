# DESIGN: Stage 06 Internal Frozen Validation

## 1. 科学问题与研究背景
在经历了 Stage 04（Learned Correction: 04A, 04B1, 04B2）和 Stage 05（Scorer Redesign: 05P, 05B）的系统性探索与证伪后：
1. **生成端残差修正尝试穷尽**：04A/04B1/04B2 均因跨场景支持度偏移（Support Shift）出现 Over-Editing，破坏了原生候选；系统已预注册决策锁定生成端为 **Static K24**（Native 20 + Top-4 Explicit Proposals）。
2. **打分器路线穷尽**：05P（浅层探针）与 05B（Cross-Attention 深度打分器）在未见过的 Zara Outer 上均无法战胜原始 Native 先验（38.20% vs 40.23%），打分器路线正式终止，状态机锁定 `selected_scorer = None`。

本阶段（06_INTERNAL_VALIDATION）的科学目标是：
**在完全冻结生成端（Static K24）、完全冻结指标阈值（ADE $\le 0.2$m, FDE $\le 0.4$m, J $\le 1$）的前提下，执行严格的内部两级隔离确认（Level A: Leave-One-Scene-Family-Out & Leave-One-Recording-Out），全面审计 Static K24 在各内部隔离折叠上的增益稳定性、零丢失特性（$Lost \equiv 0$）与动力学合规性，最终形成待人工审阅的终审测试协议与研究总结。**

---

## 2. 验证协议设计
根据 `06_OUTER_VALIDATION_PROTOCOL.md`：
- **Level A 验证集划分**：
  1. **Leave-One-Scene-Family-Out**（3折）：
     - Fold 1: `hotel`（N=759）
     - Fold 2: `university`（N=15,578）
     - Fold 3: `zara`（N=1,561）
  2. **Leave-One-Recording-Out**（7折）：
     - Fold 1: `biwi_hotel`
     - Fold 2: `crowds_zara01`
     - Fold 3: `crowds_zara02`
     - Fold 4: `crowds_zara03`
     - Fold 5: `students001`
     - Fold 6: `students003`
     - Fold 7: `uni_examples`
  3. **Original CAL 描述性基准**（N=2,879）
- **CONTROL**: 原生 Native 20（F16 原生候选集合）。
- **TREATMENT**: 静态 Static K24（Native 20 + Top-4 Observation-Driven Proposals）。

---

## 3. 通过与证伪门槛
- **必须满足门槛**：
  1. **严格保持零丢失**：在所有验证折叠中，必须实现 $Lost \equiv 0$（由于 Static K24 是在 Native 20 基础上无损增补 4 条显式机动候选，任何破坏原生好候选均不可接受）；
  2. **广泛产生救援增量**：在绝大多数折叠上 $Rescued > 0$，总体覆盖率（Complete Coverage）单调非降；
  3. **运动学物理门槛全部通过**：速度、加速度、加加速度 p95 在所有折叠上均在物理合理区间内；
  4. **严禁触碰最终独立测试集**：本实验只对内部 Level A 数据执行只读评测，终审状态输出为 `READY_FOR_HUMAN_REVIEW_FOR_FINAL_TEST`。
