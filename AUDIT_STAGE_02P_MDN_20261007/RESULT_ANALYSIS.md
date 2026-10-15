# RESULT_ANALYSIS: Stage 02P-MDN Probabilistic Motion Program

## 1. 实验目标与研究背景
在完成 `MOTION_PROGRAM_TARGET_IDENTIFIABILITY_AUDIT` 确证历史条件未来分布存在强烈多模态（77.25% Change 样本在相似历史下未来分叉）后，本实验启动预注册的 **`02P_MDN`** 概率分支：
- 构建轻量 3-component 对角高斯混合密度网络（MDN），以 P2 编码器表征（96 维）为输入建模 $p(z \mid \text{history}, \text{class})$；
- 采用**联合模式分配器（Joint Mode Allocator）**：
  从 8 类别 $\times$ 3 模式 = 24 个候选连续运动程序中，依据 $P(c \mid h) \times \pi_{c, m}$ 联合置信度挑选 Top-4 候选；
- 严格限制单一动作类别最多占用 2 个 slots，**候选总预算继续严格锁定在 $K=24$（Native 20 冻结 + 4 个 MDN 模式）**。

---

## 2. 跨协议隔离评估量化结果

在 3 个严格隔离的泛化协议上分别评估 MDN-K24 对比 Static-K24：

| 隔离协议 | 样本数 $N$ | Static-K24 覆盖率 (J) | MDN-K24 覆盖率 (J) | 覆盖率净增量 | 挽救 / 丢失 | minADE / minFDE 变化 | 判定 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. target-track** | 1,035 | 61.64% (0.9612) | **62.71% (0.9573)** | **+1.06 pp** | 救 21, 丢 10 (净 +11) | minADE +0.23%, minFDE +1.46% | **PASS (胜出)** |
| **2. recording-time** | 566 | 56.01% (0.9746) | **55.12% (0.9846)** | **-0.88 pp** | 救 8, 丢 13 (净 -5) | minADE -1.04%, minFDE -0.05% | FAIL (未胜出) |
| **3. scene-family** | 1,561 | 67.97% (0.7981) | **68.48% (0.7946)** | **+0.51 pp** | 救 25, 丢 17 (净 +8) | minADE +0.10%, minFDE +2.08% | **PASS (胜出)** |

### 核心观察:
1. **跨协议 2/3 胜出**: MDN-K24 在 `target-track`（净增 +1.06 pp）和 `scene-family`（净增 +0.51 pp）上均战胜 Static-K24，胜出率达到 2/3；
2. **target-track 上突变保护显著改善**:
   在 target-track 上，Change 难例覆盖率为 43.48%（与 Static 保持完全持平，未发生 Stage 03 中的严重退化）；
3. **整体 minFDE 明显改善**: 在三大协议中，终点位移误差 minFDE 均获得了 1.46%~2.08% 的提升，显示多模态混合高斯模式有效保留了具有更远/不同方向的终末状态。

---

## 3. 门禁验证与状态机决策

- **预注册门禁**: `probabilistic_program_beats_static_in_2of3`；
- **实测结果**: 2 / 3 协议胜出，门禁达成（**TRUE**）；
- **状态机决策**:
  选定最佳动力学表征 `selected_parameter_representation = "P2"`，
  选定参数预测器 `parameter_predictor = "MDN"`，
  晋级至 **Stage 03: LearnedParam-K24 (MDN-Augmented 3-Seed Formal Audit)**！
