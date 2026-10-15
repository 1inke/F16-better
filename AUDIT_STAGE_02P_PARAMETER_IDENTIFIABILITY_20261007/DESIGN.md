# DESIGN: Stage 02P Parameter Identifiability

## 1. 科学问题与研究背景
Stage 01A Oracle 分解给出了决定性结论：
- 自由连续 Motion Program 拥有巨大理论天花板（好候选覆盖率达到 **95.45%**，J 降低 **72.09%**）；
- **参数缺口占据总缺口的 80.11%**（仅优化 Top-4 相同类别的连续参数，覆盖率即可达到 **93.21%**）；
- 类别选择缺口仅占 5.30%，分类学缺口仅占 14.59%。

因此，后续研究的核心科学问题转变为：
**能否仅基于过去观测历史（Past-Only Information），利用 F16 的内部表征准确识别/预测连续运动程序的 8 维参数 $z = [s_1, s_2, s_3, \theta_1, \theta_2, \theta_3, \text{onset}, \text{persistence}]$？**
并且：
**使用确定性预测参数生成的 Learned Proposals，能否在不泄漏任何未来信息的前提下，在跨协议迁移测试中稳健超越固定默认参数的 Static Proposals？**

---

## 2. 实验对照设计
### 2.1 四臂表征对比（Representation Arms）
为了公平比较不同信息源的可辨识度，所有表征 arm 统一使用结构相同的紧凑 MLP 预测器（Linear $\rightarrow$ GELU $\rightarrow$ Linear $\rightarrow$ GELU $\rightarrow$ Linear(8)）：
- **P0**: 基础观测动力学特征（Raw Observed Kinematics，从 motion 提取速度、加速度等）；
- **P1**: F16 稳健运动状态（Robust State，从编码器抽取 32 维基础运动隐状态）；
- **P2**: F16 完整编码器时空向量（Encoder Output，96 维终层时序池化特征）；
- **P3**: 编码器向量 + 交互特征（Encoder Output + Social，112 维联合特征）。

### 2.2 三大跨场景/跨实体分组协议（Grouped Protocols）
必须在 3 个严格隔离的协议上分别评估泛化表现：
1. **target-track**: 按行人目标轨迹隔离（$N=1,035$ Outer 测试集）；
2. **recording-time**: 按录像时间块隔离（$N=566$ Outer 测试集）；
3. **scene-family**: 按场景家族隔离（Zara Outer，$N=1,561$ 测试集）。

---

## 3. 门禁与证伪标准（Falsification Criteria）
- **晋级 Stage 03 条件**:
  若最佳表征的确定性预测程序在**至少 2/3 个分组协议**上重建的提议轨迹优于 Static 程序（即重构 ADE 降低且 coverage 提升），则判定参数具备确定性可辨识性（`deterministic_program_beats_static_in_2of3 = True`），直接进入 Stage 03 组装 LearnedParam-K24。
- **允许 02P_MDN 条件**:
  若确定性预测在 outer 上因多模态均值回归失效，但 01A 证明 Oracle 参数缺口巨大（`oracle_parameter_gap_large = True`），根据预注册规则，允许一次唯一的对角高斯混合密度网络（3-component MDN）重试。
- **终止条件**:
  若 MDN 仍无法超越 Static，则判定参数无法从历史有效识别，安全停机输出 `STOP_PARAMETER_NOT_IDENTIFIABLE_FROM_PAST`。
