# F16 多模态候选轨迹 Rankability Probe 系列 (F16_RANKABILITY_PROBE_SERIES)

> **基线体系**：EXP-173 V2 / `f16_temporal_social_cross` / ETH / Seed 1  
> **核心原则**：冻结 F16 候选生成器，面对完全相同的 20 个候选集合，严格评估“已有候选是否能够被正确排序/选出”。  
> **门禁裁决结论**：**`[STATUS: CASE D - UNRANKABLE_REPRESENTATION_BOTTLENECK]`**

---

## 1. 结构与模块组成

```text
F16_RANKABILITY_PROBE_SERIES/
├── cache_rankability_features.py      # 冻结特征与相对坐标系候选离线抽取器
├── audit_cache_identity.py            # P0 缓存数值一致性独立重算审计
├── audit_score_staleness.py           # P1 Refiner 修正后评分时序滞后性统计
├── train_rankability_probe.py         # P2 特征阶梯探针训练器 (R1 - R4)
├── evaluate_rankability_probe.py      # P2 全指标评测与 1000次 Bootstrap 检验
├── ranker_models.py                   # 轻量诊断探针结构 (~6.8k - 12.2k 参数)
├── metrics.py                         # 决策相关评测指标库
├── common/
│   ├── feature_schema.py             # 维度定义与 Schema 严格断言
│   └── pairwise_objective.py         # Decision-Relevant Pairwise RankNet Loss
├── tests/
│   └── test_no_future_leakage.py      # 未来真值无泄漏与物理隔离单元测试
├── results/
│   ├── p0_identity/
│   │   └── CACHE_IDENTITY_AUDIT.md   # P0 数值一致性对齐报告 (误差 0.00e+00)
│   ├── p1_staleness/
│   │   ├── staleness_metrics.csv     # 滞后性统计全局汇总表
│   │   ├── correction_quantiles.csv  # Anchor 修正分位数 Q1-Q4 切片表
│   │   └── POST_REFINEMENT_STALENESS_REPORT.md  # P1 时序滞后性诊断报告
│   ├── p2_rankability/
│   │   ├── probe_summary.csv         # R0-R4 全样本表现与闭合率汇总表
│   │   ├── hardcase_summary.csv      # 按 Selection Regret 真实失误切片表
│   │   ├── bootstrap_ci.csv          # 1000次 Bootstrap 95% 置信区间表
│   │   └── RANKABILITY_REPORT.md     # P2 诊断与裁决报告 (回答6大核心问题)
│   └── state.json                     # 全流程状态机记录
└── README.md
```

---

## 2. 核心实验结论速查

1. **P0 Cache Identity Gate (`PASS`)**：
   - 最终轨迹与分类 Logits 在独立重算下最大绝对偏差均为 **`0.00e+00`**；
   - 验证集基线事实完全复现：Selected ADE = `0.5389 m`，Oracle minADE20 = `0.2196 m`，Regret = `0.3193 m`。
2. **P1 Score Staleness Audit (`POSSIBLE_STALENESS`)**：
   - 验证集中 **`31.33%`** 的样本在 Refiner 微调修正后最优模态易位；
   - 原生评分头对最终轨迹质量的 Spearman 秩相关相比粗轨迹净衰减 **`0.0254`**；
   - 在 Anchor 修正最大的 Q4 分位组中，最优模态易位率高达 **`58.0%`**。
3. **P2 Frozen Rankability Probe (`CASE D: UNRANKABLE_REPRESENTATION_BOTTLENECK`)**：
   - **R1 (原输入重训, 70维)**：Selected ADE = `0.5323 m`，Regret 闭合率仅 **`2.06%`**；
   - **R2 (后微调几何, 70维)**：Selected ADE = `0.5304 m`，Regret 闭合率仅 **`2.66%`**；
   - **R3 (加 Context, 134维)**：Selected ADE = `0.5301 m`，Regret 闭合率仅 **`2.74%`**；
   - **R4 (完整轨迹几何, 152维)**：Selected ADE = `0.5309 m`，Regret 闭合率仅 **`2.50%`**；
   - **核心定论**：即使提供完整轨迹几何与 Context，重训排序损失，**高达 97.26% 的选模差距在冻结表征中仍不可恢复**！这证明 F16 的核心瓶颈不在于下游 Selector，而在于 **共享 Encoder 提取的隐空间表征缺乏对行人意图的充分辨识容量**。
