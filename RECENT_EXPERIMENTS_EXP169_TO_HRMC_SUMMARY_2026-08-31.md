# 最近实验说明：EXP-169 至 Horizon-Resolved Motion Conditioning

> 日期：2026-08-31
> 数据集：ETH/UCY official five-fold
> 正式对照：EXP-164，`F16 + H3 + V2 + original score`，K=20
> 第一轮口径：五折、seed 1；`J = minADE20 + 0.5 × minFDE20`

## 1. 本阶段研究问题

EXP-168 已经发现 structured/free 两类候选存在明显互补，但固定 10/10 配额虽然改善 minADE，却损害 minFDE 与 same-path FDE。本阶段依次回答：

1. free 候选的价值发生在哪些预测时刻，现有 structured 候选能否通过小残差到达？
2. ETH 的 low-density × high-acceleration 困难是否由 social branch 过度注入造成？
3. 如果不是 social branch，运动信息是否在编码器中丢失？
4. 已编码的运动状态是否被 H3/V2 正确转换成候选几何？
5. 在冻结正式模型主体的条件下，运动条件残差能否产生稳定、足够大的提升？

## 2. 实验链与结论

| 阶段 | 类型 | 核心发现 | 决策 |
|---|---|---|---|
| EXP-169A Cross-Family Horizon Audit | 零训练审计 | structured/free 在 t4/t8/t12 均有强互补，但 t8/t12 的半预算可达率仅 25.90%/27.95% | 关闭 EXP-170A/B bounded dual-basis |
| EXP-169B Sparse-Social Counterfactual | 零训练审计 | Full Social 在 ETH-LDHA 的 J 优势为 4.21%，五折 LDHA macro 优势为 4.98% | 不削弱 social；关闭 EXP-169C/171A |
| Individual Motion Representation Audit | 零训练审计 | robust_state 的 acceleration outer R²=0.447、Spearman=0.850；最终 encoder Spearman=0.799 | 不是运动信息缺失，不增加 acceleration encoder |
| Motion-to-Candidate Transfer Audit | 零训练审计 | 高加速度候选最近距离约为普通样本 1.8–1.9 倍；t12 候选响应仅为 GT 的 66.4% | 瓶颈位于 representation-to-candidate transfer |
| H3→V2 Correction Attribution | 零训练审计 | V2 总体 J 改善 4.74%，但 high-acc 仅 2.57%，non-high 达 10.30%；ETH high-acc 为 -0.40% | 批准 proposal-only/correction-only 隔离训练 |
| P1/C1 Horizon-Resolved Motion Conditioning | 受控训练 | P1 比 C1 更有效，但总体 J 仅提升约 0.39%，且普通运动样本退化 | 不晋级，不补 seeds 2/3，停止小残差路线 |

## 3. EXP-169A：互补性真实，但局部不可达

| Horizon | Macro prefix oracle gain | Macro point oracle gain | Reach@0.5×V2 | Reach@1.0×V2 |
|---|---:|---:|---:|---:|
| t4 | 31.68% | 35.23% | 47.17% | 80.17% |
| t8 | 38.90% | 49.30% | 25.90% | 33.98% |
| t12 | 41.15% | 48.69% | 27.95% | 36.14% |

结论不是“两族没有互补”，而是有价值的 free 候选多数距离 structured 候选过远。尤其 ETH 的半预算 reach t4/t8/t12 只有 24.5%/10.6%/14.9%。因此不能通过扩大 V2 correction cap 或小型 dual-basis residual 强行吸收这些候选。

预注册结果：

- `EXP170A_ALLOWED = false`
- `EXP170B_ALLOWED = false`

## 4. EXP-169B：Full Social 不是 ETH 短板

同一 checkpoint、同一输入与同一 decoder 下，仅切断目标行人接收的他人信息。

| 范围 | Full Social 的 ADE 优势 | FDE 优势 | J 优势 |
|---|---:|---:|---:|
| 五折 ALL macro | 4.20% | 5.99% | 5.01% |
| 五折 LDHA macro | 4.59% | 5.51% | 4.98% |
| ETH LDHA | 3.90% | 4.64% | 4.21% |

Source-inner LDHA proxy 中 4/5 个 setup 方向为正，没有稳定的 Self-only 优势。由此排除 density gate、social suppression、Self-only fallback 与 support-calibrated social residual。

## 5. 编码信息没有消失，断点发生在候选几何

### 5.1 表征审计

| Representation | Macro outer acceleration R² | Pearson | Spearman |
|---|---:|---:|---:|
| robust_state | 0.447 | 0.834 | 0.850 |
| temporal | 0.030 | 0.805 | 0.754 |
| post_social | 0.284 | 0.786 | 0.753 |
| encoder_output | 0.336 | 0.824 | 0.799 |

加速度在最终 encoder 中仍高度可排序，因此继续增加运动编码器、速度尺度分支或范数放大器缺少依据。真正异常是 source→target 的绝对映射校准不稳定，而不是特征完全不可用。

### 5.2 候选传递审计

| Horizon | High-acc nearest | Non-high nearest | H3 response / GT | V2 response / GT |
|---|---:|---:|---:|---:|
| t4 | 0.136 | 0.076 | 87.6% | 93.0% |
| t8 | 0.281 | 0.153 | 74.4% | 78.0% |
| t12 | 0.468 | 0.246 | 64.4% | 66.4% |

预测时域越长，候选对运动强度的响应越不足。高加速度样本上的 correction norm 更小、cap-hit 更低，说明问题不是 correction budget 不够，而是 refiner 没有主动使用已有预算。

## 6. H3→V2 归因：V2 有效，但收益分配失衡

| Group | minADE improvement | minFDE improvement | J improvement |
|---|---:|---:|---:|
| ALL | 4.12% | 5.47% | 4.74% |
| High acceleration | 2.36% | 2.77% | 2.57% |
| Non-high acceleration | 7.08% | 14.85% | 10.30% |

V2 的主要机制不是持续细化原 H3 winner，而是重排 20 个 mode 的几何：部分原 winner 被牺牲，其他 mode 被修正后成为新 oracle winner。高加速度样本中原 H3 ADE winner 经 V2 后平均反而变差，ETH high-acc 的 J 也退化 0.40%。

该结果批准了最后一次受控训练：在 proposal 与 correction 两个位置分别加入连续、零初始化、分时域运动条件残差。

## 7. P1/C1 正式五折结果

### 7.1 结构

- P1：在 H3 source anchors 之后、冻结 V2 refiner 之前加入 mode-wise t4/t8/t12 二维残差。
- C1：H3 proposal 不变，仅对现有 V2 correction 增加运动条件残差。
- F16、social branch、H3、V2 主体与 score 全部冻结。
- 每个变体新增 8,580 个可训练参数；模型总参数 330,568。
- 不使用 hard high-acc gate、scene ID、CV reference、全局缩放或扩大 correction cap。

### 7.2 Macro 指标

| Model | minADE20 | minFDE20 | same-path FDE | J | 结论 |
|---|---:|---:|---:|---:|---|
| EXP-164 V2 seed1 | 0.222457 | 0.364897 | **0.433931** | 0.404905 | 正式对照 |
| P1 proposal residual | **0.221492** | **0.363671** | 0.434807 | **0.403328** | ADE +0.44%，FDE +0.34%，J +0.39% |
| C1 correction residual | 0.222315 | 0.364399 | 0.434184 | 0.404515 | ADE +0.06%，FDE +0.14%，J +0.10% |

### 7.3 P1 逐折变化

| Fold | minADE change | minFDE change | 判断 |
|---|---:|---:|---|
| ETH | -0.00% | -0.28% | 无效/轻微退化 |
| Hotel | -0.28% | -0.27% | 退化 |
| Univ | +1.70% | +0.75% | 明确有效但单折主导 |
| Zara01 | +0.44% | +0.11% | 微弱 |
| Zara02 | +0.31% | +2.03% | 主要改善终点 |

### 7.4 运动分组

P1 对 high-acc 的 macro J 改善 0.84%，但对 non-high-acc 的 macro J 退化 1.77%；后者的 minFDE 退化 3.10%。Hotel non-high-acc 尤其明显，J 退化 9.15%。模块确实学习了随 horizon 增大的修正，但它会扰动本来容易预测的样本。

P1 的平均 adapter norm：

- t4：0.0176
- t8：0.0430
- t12：0.0819

这说明失败不是 adapter 没有更新，而是小残差只能移动已有候选，无法形成所需的新候选族；同时 same-path FDE 未改善，终点最佳候选与整条路径最佳候选的矛盾仍然存在。

## 8. 当前正式决策

1. 保留 EXP-164 `F16 + H3 + V2 + original score` 作为正式模型。
2. C1 淘汰。
3. P1 保留为正向但不足的消融证据，不替代 V2，不补 seeds 2/3。
4. 停止继续搜索 motion residual 的 cap、深度、学习率、hard/soft threshold。
5. 不启动 EXP-170A/B、EXP-169C 或 EXP-171A。
6. 如果继续追求接近 10% 的提升，需要建立新的候选生成假设，使运动条件直接决定候选模板/几何，而不是继续后置小残差；该方向需要新的实验指导和独立预注册。

## 9. GitHub 归档说明

本次同步包含：

- 实验脚本与运行驱动；
- README、RESULT_ANALYSIS；
- 五折聚合 CSV/JSON；
- P1/C1 每次训练的 history、metrics 与锁定配置。

不同步：

- 原始或处理后数据集；
- `.pth/.pt/.ckpt` 权重；
- `.npy/.npz` 数据与 split indices；
- console logs；
- 数十 MB 的逐样本 CSV。逐样本表可由已提交脚本和本地 checkpoint 重新生成，仓库保留对应聚合结果。
