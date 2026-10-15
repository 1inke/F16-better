# DESIGN: Stage 03 MDN-LearnedParam-K24 (3-Seed Formal Audit)

## 1. 科学问题与背景
在经历 Stage 03 确定性单点预测因均值平滑陷阱受阻、`TARGET_IDENTIFIABILITY_AUDIT` 确证 77.25% 历史条件未来多模态、以及 `02P_MDN` 在 2/3 协议成功击败 Static-K24 后，状态机正式推进至：
**Stage 03: MDN-Augmented LearnedParam-K24 3-Seed 严格对照实验**。

核心科学问题：
**在严格固定预算 $K=24$（Native 20 冻结 + 4 个 MDN 模式提议）且不修改 Native/Scorer 的前提下，3-component 对角高斯 MDN 结合联合模式分配器（单类上限 2 个 slots），能否在 3 个独立随机种子（Seed 42, 43, 44）上在主协议 Zara scene-family 上全面通过预注册 Promotion Gate？**

---

## 2. 实验对照设计
- **CONTROL**: `native20 + Static Top4 proposals`（$K=24$）；
- **TREATMENT**: `native20 + Top-4 MDN joint mode proposals`（$K=24$）；
- **3 Paired Seeds**: Seed 42, 43, 44；
- **评估数据集**: Zara scene-family 隔离测试集（$N=1,561$）；
- **细分群组**: Stable ($N=979$), Transition ($N=371$), Change ($N=211$)；
- **联合分配机制**: 8 classes $\times$ 3 modes = 24 个候选连续运动程序，依据 $P(c \mid h) \times \pi_{c, m}$ 挑选 Top-4，单一动作类别最多占用 2 个 slots。

---

## 3. 门禁验证标准 (Promotion Gate)
1. `change` 组 minADE 或 minFDE 至少一项改善 $\ge 3\%$，或 `change` 覆盖率提升 $\ge +1\text{ pp}$；
2. `overall` minADE 与 minFDE 不允许出现 $> 1\%$ 的恶化；
3. 至少 $2/3$ 的 seed 保持同向正收益；
4. 运动学守护通过（速度 p95、加速度 p95、加加速度 p95 在合规界限内）；
5. 整体重构必须直接击败 Static-K24。
