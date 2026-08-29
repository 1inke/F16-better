# FCRA1 实验结果分析

## 结论

FCRA1 **未达到 Seed1 晋级标准，不建议继续多种子复验，也不建议围绕当前后置 residual correction 继续调权重**。

它确实修复了 CSRS1 的一部分机制问题：候选责任分配明显优于随机，正样本修正量也终于高于负样本，而且候选多样性没有塌缩。然而，最终 General 与 Challenge 的 overall minADE/minFDE 均退化；actual-social-needed 子集的改善也远低于 2% 门槛。更关键的是，预测 residual 与真实 candidate residual 的平均余弦为负，说明修正方向仍没有可靠学成。

因此，本轮最重要的研究结论不是“再调一下 FCRA1”，而是：

> social information 有价值，但在 Full candidates 生成后再把它提取为显式候选修正，当前证据表明收益上限很低。下一步应转向 interaction-aware mode allocation / matching loss，让交互信息直接决定不同 mode 在训练中负责哪些未来分支。

## 1. Overall 指标

数值越低越好；变化率为负表示改善。

| Split | 模型 | minADE20 | minFDE20 | Same-path FDE | Top1 ADE | Top1 FDE |
|---|---:|---:|---:|---:|---:|---:|
| General | B0 | 0.17868 | 0.31407 | 0.37851 | 0.46739 | 1.02615 |
| General | FCRA1 | 0.17892 | 0.31602 | 0.37665 | 0.46763 | 1.02867 |
| General | 变化 | **+0.138%** | **+0.619%** | -0.493% | +0.052% | +0.245% |
| Challenge | B0 | 0.27807 | 0.47784 | 0.56940 | 0.75006 | 1.61948 |
| Challenge | FCRA1 | 0.27846 | 0.47920 | 0.57019 | 0.75086 | 1.62034 |
| Challenge | 变化 | **+0.138%** | **+0.284%** | +0.138% | +0.106% | +0.053% |

General 只有 Same-path FDE 改善 0.493%，但两个主要 oracle 指标都退化。Challenge 五项指标全部退化，说明 General 的局部改善没有跨分布复现。

相较 CSRS1，FCRA1 把 General minADE 的退化从 +0.824% 压到 +0.138%，说明修复 target/base mismatch 与增加显式 assignment 是有效的“止损”；但它没有转化为净收益，而且 General minFDE 仍退化 0.619%。

## 2. Actual-social-needed 子集

| Split | 样本数 | minADE20 变化 | minFDE20 变化 | Same-path FDE 变化 | Top1 ADE 变化 | Top1 FDE 变化 |
|---|---:|---:|---:|---:|---:|---:|
| General | 628 | -0.293% | +0.478% | -0.991% | -0.839% | -0.637% |
| Challenge | 1412 | -0.135% | -0.557% | -0.139% | +0.186% | +0.139% |

这说明 FCRA1 在真正需要社会信息的样本上并非完全无效，但改善幅度只有约 0.1%–1.0%，且 ADE/FDE、oracle/Top1 之间方向不一致。它远未达到预设的“至少一个主要指标改善 2%”标准，也不足以支撑接近 10% 的总体提升目标。

## 3. Candidate responsibility 是否学会

| 指标 | General | Challenge | 随机基准 |
|---|---:|---:|---:|
| Top1 accuracy | 12.99% | 9.33% | 5% |
| Top3 recall | 38.72% | 26.12% | 15% |
| Soft CE | 2.657 | 2.929 | 约 2.996（20 类均匀） |
| Top3 gate entropy | 1.094 | 1.097 | ln(3)=1.099 |

显式 assignment **学到了弱信号**，明显优于随机；但强度不够：General 仍漏掉 61.3% 的 oracle candidate，Challenge 漏掉 73.9%。Top3 gate entropy 几乎等于三类均匀分布的最大熵，说明模型在选出的三个 mode 内基本无法判断谁更应该承担修正。

所以 assignment 没有完全失败，但距离可用于精确 candidate correction 还很远，尤其在 Challenge 上接近弱分类器。

## 4. Need/no-op 与修正幅度

| 指标 | General | Challenge |
|---|---:|---:|
| Need AUROC | 0.636 | 0.639 |
| Need AUPRC | 0.525 | 0.580 |
| Need probability 与真实 gain 的 Spearman | 0.084 | 0.128 |
| Positive correction norm | 0.000581 | 0.000394 |
| Negative correction norm | 0.000515 | 0.000366 |
| Positive / Negative | 1.129 | 1.077 |

正样本修正量终于高于负样本，满足了最基本的 no-op 方向要求，但差距仅 7.7%–12.9%。Need 仍只适合作为粗粒度开关，和真实收益强度的相关性非常弱。

同时，绝对修正量只有约 4e-4–6e-4，相比 0.18–0.48 的轨迹误差尺度极小。FCRA1 实际学成了一个“谨慎但几乎不动”的修正器，这解释了为什么所有指标变化都很细微。

## 5. Residual alignment 是核心失败点

| 指标 | General | Challenge |
|---|---:|---:|
| Assigned Top3 residual cosine（positive） | **-0.082** | **-0.184** |
| Alignment vs ΔminADE Spearman | -0.054 | +0.092 |
| Alignment vs ΔminFDE Spearman | -0.093 | +0.038 |

平均余弦为负，表示预测修正方向总体上与 `GT anchor - current Full anchor` 相反；Challenge 更差。General 中“更对齐对应更小退化”的方向勉强正确，但相关性接近零；Challenge 方向反而错误。

这解释了一个看似矛盾的现象：assignment 高于随机、positive correction 也大于 negative，但精度仍不升。模型知道“哪些样本可能要修、哪些候选可能负责”，却没有学会“应该往哪里修”。

## 6. Coverage 与 diversity

- General endpoint coverage 退化 0.619%，与 minFDE 完全一致；第 8 步仅改善 0.050%。
- Challenge 第 4/8 步分别改善 0.202%/0.138%，但 endpoint 退化 0.284%，局部锚点移动没有形成终点收益。
- General 的 endpoint/trajectory/anchor diversity 分别变化 -0.115%/-0.089%/-0.090%。
- Challenge 分别变化 +0.108%/+0.152%/+0.108%。

多样性基本保持不变，因此失败不是候选塌缩造成的；但也说明 FCRA1 没有重新组织候选覆盖，只施加了非常微弱的局部扰动。

## 7. 轻量化代价

- B0：355,048 parameters。
- FCRA1：361,232 parameters。
- 增加：6,184 parameters，约 **+1.74%**。

参数代价不大，但在精度没有提升的情况下仍不具备保留价值。

## 8. 晋级判定

| 标准 | 结果 |
|---|---|
| Overall minADE 或 minFDE 改善 ≥0.5% | **失败**：两 split 均退化 |
| Needed 子集至少一个主要指标改善 ≥2% | **失败**：最大仅 0.557%（minFDE） |
| Challenge 复现改善方向 | **失败** |
| Assignment Top3 明显高于随机 | 通过，但仍弱 |
| Positive correction > Negative | 弱通过 |
| Residual alignment 与 gain 方向正确 | **失败** |
| Diversity 不明显下降 | 通过 |

综合判定：**不晋级，不跑 Seed2/3。**

## 9. 下一步研究方向

不建议继续扩大 residual head、调整 Top-M/温度或扫 loss 权重。这些操作可能让 0.1% 级变化发生波动，但缺少获得大幅提升的机制依据。

下一轮应改为 **Interaction-Aware Mode Allocation / Matching**：

1. 不在候选生成后追加 residual correction。
2. 让 social/interaction representation 参与训练期的 mode-to-GT 分配成本。
3. 分配成本同时考虑完整轨迹、终点/锚点和交互一致性，使不同 mode 直接学习不同的社会行为分支。
4. 推理时仍一次性输出 20 条候选，不引入 CV 外推，也不牺牲当前稀疏注意力和轻量主体。

这条方向直接作用于候选形成和覆盖，而不是对已生成候选做极小后处理，更符合当前需要寻找较大提升来源的目标。

## 数据来源

- `seed1_summary.json`：严格匹配的 B0/FCRA1 总体指标与参数量。
- `diagnostic/combined_summary.json`：assignment、needed 子集、residual alignment、coverage、diversity 诊断。
- 诊断脚本已修复多锚点 residual cosine 未展平的问题，并完成重新运行。
