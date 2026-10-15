# DESIGN: Stage 03 LearnedParam-K24

## 1. 科学问题与研究背景
在 Stage 02P 中，我们通过跨协议三方隔离测试确定了 F16 的终层编码器时空向量（`P2`, 96 维）为最佳动力学表征。
在主协议 Zara scene-family 上，确定性参数预测器将好候选覆盖率从 67.97% 提升至 69.06%（净提升 +1.09 pp，净挽救 17 个窗口），且在 2/3 协议上击败了固定默认参数的静态基准。

在 Stage 03 中，我们将进行**严格受控的 3-seed 正式比对**：
**在保持完全相同的候选预算 $K=24$、完全冻结 native20 和 scorer、且完全使用相同 Top-4 动作类别的前提下，由 past-only P2 表征驱动的 LearnedParam-K24 能否在 3 个独立训练 seed 上稳健超越 Static-K24？**

---

## 2. 实验对照设计
- **CONTROL**:
  `native20 + Static Top4 proposals`（$K=24$）。
  使用在场景隔离训练集上训练的意图分类器选出的 Top-4 动作类别，并赋予固定的默认动力学参数。
- **TREATMENT**:
  `native20 + SAME Top4 classes + Learned motion program parameters`（$K=24$）。
  保持完全相同的 4 个动作类别与候选 slot，参数由基于 P2 表征训练的紧凑 MLP 确定性回归得到。
- **3 Paired Seeds**:
  使用 3 个独立随机种子（Seed 42, 43, 44）训练 3 组配对模型，逐 seed 独立评估与配对统计。
- **评估数据集**:
  Primary 测试集：Zara scene-family 隔离测试集（$N=1,561$）。
  同时监控 `stable` ($N=967$)、`transition` ($N=371$)、`change` ($N=223$) 与 `clear_event` ($N=469$) 分组。

---

## 3. 门禁与证伪标准（Preregistered Gates）
- **Promotion Gate (晋级门禁)**:
  1. `change` 组 minADE 或 minFDE 至少一项改善 $\ge 3\%$，或 `change` / `clear-event` 覆盖率提升 $\ge +1\text{ pp}$；
  2. `overall` minADE 与 minFDE 不允许出现 $> 1\%$ 的恶化；
  3. 至少 $2/3$ 的 seed 保持同向正收益；
  4. 运动学守护通过（速度 p95、加速度均值/p95、加加速度均值/p95 无严重异常）；
  5. Learned proposal 必须直接击败 Static proposal。
- **证伪处理**:
  若无法通过 Promotion Gate，则判定该连续运动程序在同类别控制下未能产生实质收益，状态机进入 `STOP_LEARNED_PARAM_NO_GAIN`。
- **若通过 Promotion Gate**:
  选定 `selected_generator = "LEARNED_PARAM_K24"`，状态机晋级至 **Stage 04: Counterfactual Semantic Audit**！
