# F16 多模态候选轨迹 Rankability Probe 系列 (F16_RANKABILITY_PROBE_SERIES)

> **基线体系**：EXP-173 V2 / `f16_temporal_social_cross` / ETH / Seed 1  
> **核心原则**：冻结 F16 候选生成器，面对完全相同的 20 个候选集合，严格评估“已有候选是否能够被正确排序/选出”。  
> **门禁裁决结论**：**`[STATUS: CASE C - FROZEN_REPRESENTATION_SELECTION_BOTTLENECK]`**

---

## 1. 结构与模块组成

```text
F16_RANKABILITY_PROBE_SERIES/
├── cache_rankability_features.py            # 冻结特征与相对坐标系候选离线抽取器
├── audit_cache_identity.py                  # P0 缓存数值一致性独立重算审计
├── audit_score_staleness.py                 # P1 Refiner 修正后评分时序滞后性统计
├── train_rankability_probe.py               # P2 特征阶梯探针训练器 (R1 - R4)
├── train_top1_probes.py                     # P2 Top-1 对齐与集合感知探针训练器 (R5 - R7)
├── evaluate_rankability_probe.py            # 全指标评测、配对探针差值 CI 与统计检验
├── ranker_models.py                         # 探针模型库 (MLP 探针 ~6.8k-12.2k, Set-Aware 探针 42.5k)
├── metrics.py                               # 修正后 Pairwise 准确率与 Oracle Margin 指标库
├── common/
│   ├── feature_schema.py                   # 维度定义与 Schema 严格断言
│   └── pairwise_objective.py               # All-pair RankNet, Oracle-vs-Rest, Listwise CE
├── tests/
│   ├── test_no_future_leakage.py            # 未来真值隔离单元测试
│   ├── test_pairwise_metrics.py             # Pairwise 准确率符号正确性单元测试
│   └── test_oracle_vs_rest_loss.py          # Oracle-vs-Rest 目标单元测试
├── results/
│   ├── p0_identity/
│   │   └── CACHE_IDENTITY_AUDIT.md         # P0 数值一致性对齐报告 (误差 0.00e+00)
│   ├── p1_staleness/
│   │   ├── staleness_metrics.csv           # 滞后性统计全局汇总表
│   │   ├── correction_quantiles.csv        # Anchor 修正分位数 Q1-Q4 切片表
│   │   └── POST_REFINEMENT_STALENESS_REPORT.md  # P1 时序滞后性诊断报告 (审慎修正版)
│   ├── p2_rankability/
│   │   ├── probe_summary.csv               # R0-R7 全探针表现与闭合率汇总表
│   │   ├── hardcase_summary.csv            # 按 Selection Regret 真实失误切片表
│   │   ├── bootstrap_ci.csv                # 原生对比 1000次 Bootstrap 95% 置信区间表
│   │   ├── pairwise_probe_comparison_ci.csv # 探针间增量配对 Bootstrap 差值显著性表
│   │   ├── RANKABILITY_REPORT.md           # P2 特征阶梯诊断报告 (审慎修正版)
│   │   └── TOP1_ALIGNED_PROBE_REPORT.md    # STOP POINT 3 最终科学裁决报告
│   └── state.json                           # 全流程状态机记录
└── README.md
```

---

## 2. 核心实验结论速查

1. **P0 Cache Identity Gate (`PASS`)**：
   - 最终轨迹与分类 Logits 在独立重算下最大绝对偏差均为 **`0.00e+00`**；
   - 验证集基线事实完全复现：Selected ADE = `0.5389 m`，Oracle minADE20 = `0.2196 m`，Regret = `0.3193 m`。
2. **P1 Score Staleness Audit (`POSSIBLE_NOT_DOMINANT`)**：
   - 验证集中 **`31.33%`** 的样本在 Refiner 微调修正后最优模态易位；
   - 但后置几何探针 R2 相对 R1 仅取得 1.9 mm 微弱增益，置信区间跨零（`[-0.0001, +0.0042] m`），证实其非选模主要根因。
3. **指标符号纠偏与真实 Rankability**：
   - 修复前期 $\Delta_{ij} = ADE_j - ADE_i$ 符号颠倒错误，实测二元模态对排序准确率高达 **`80% ~ 85%`**，Oracle 对比恶性竞争模态的 Margin 准确率也达 **`72% ~ 73%`**。
4. **Top-1 对齐与集合感知诊断 (`CASE C: FROZEN_REPRESENTATION_SELECTION_BOTTLENECK`)**：
   - **R5 (Oracle-vs-Rest)**：聚焦 Top-1 边际，3 种子均值闭合率为 **`-0.26%`**；
   - **R6 (Listwise Top-1 CE)**：直接进行 20 选 1 多分类，3 种子均值闭合率为 **`+1.21%`**（最好种子为 3.19%）；
   - **R7 (Set-Aware Ranker)**：引入 1 层 Self-Attention 全局交互（42.5k 参数），3 种子均值闭合率为 **`+0.48%`**；
   - **核心定论**：在消除了“目标错位”、“评分滞后”、“单候选无交互”等一切可能因素后，全系列探针最大闭合率依然压制在 **`3.19%`** 以下，而在严重失误样本群（Regret > 20cm）中 Oracle Recall@1 仅为 **`1.8% ~ 3.2%`**。这充分排他性地证明了**冻结潜空间表征对于 Top-1 意图辨识的信息匮乏**，为进入端到端表征重构（Representation Redesign）奠定了坚实科学基础。
