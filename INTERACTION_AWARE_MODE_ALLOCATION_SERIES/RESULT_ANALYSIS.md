# MA1 实验结果分析

## 核心结论

MA1 **明确失败，不进入 Seed2/3，也不建议继续扫描 Top-M、temperature 或 social strength**。

本轮确实产生了约 10% 的指标变化，但方向与目标相反：General minADE 退化 10.47%，minFDE 退化 5.08%；Challenge 也分别退化 1.79% 和 1.61%。机制诊断表明，Top-5 soft regression 让多个候选同时追逐同一条 GT，导致候选集合收缩，而 social compatibility 并没有让 social-needed 样本形成不同的责任分布。

一句话概括：

> MA1 改变了 mode formation，但改变方式是把候选拉得更相似，而不是形成更有社会语义的职责分工。

## 1. Overall 指标

数值越低越好；变化率为负才是改善。

| Split | 模型 | minADE20 | minFDE20 | Same-path FDE | Top1 ADE | Top1 FDE |
|---|---:|---:|---:|---:|---:|---:|
| General | B0 | 0.17868 | 0.31407 | 0.37851 | 0.46739 | 1.02615 |
| General | MA1 | 0.19738 | 0.33003 | 0.40168 | 0.48204 | 1.03094 |
| General | 变化 | **+10.467%** | **+5.080%** | +6.120% | +3.136% | +0.466% |
| Challenge | B0 | 0.27807 | 0.47784 | 0.56940 | 0.75006 | 1.61948 |
| Challenge | MA1 | 0.28305 | 0.48553 | 0.56970 | 0.74781 | 1.61249 |
| Challenge | 变化 | **+1.791%** | **+1.609%** | +0.053% | -0.299% | -0.432% |

General 所有指标均退化。Challenge 的 Top1 ADE/FDE 小幅改善，但 oracle minADE/minFDE 明显退化。这种“Top1 稍好、best-of-20 更差”的分离说明模型生成了更接近平均解的候选，却损失了候选集合的覆盖范围。

## 2. Social-needed 子集

| Split | 样本数 | minADE20 变化 | minFDE20 变化 | Same-path FDE 变化 | Top1 ADE 变化 | Top1 FDE 变化 |
|---|---:|---:|---:|---:|---:|---:|
| General | 628 | **+9.988%** | **+8.997%** | +11.076% | +0.675% | +0.003% |
| Challenge | 1412 | **+3.312%** | **+4.494%** | +4.265% | -0.347% | -0.406% |

真正需要社会信息的样本没有受益，反而比 Challenge overall 退化得更严重。因此不能把 overall 失败解释为 non-needed 样本稀释；MA1 没有把社会信息转化为有利的 mode specialization。

## 3. 候选多样性发生明显收缩

| Split | Endpoint diversity | Trajectory diversity | Anchor diversity |
|---|---:|---:|---:|
| General | **-13.64%** | **-13.25%** | **-12.71%** |
| Challenge | **-6.51%** | **-7.31%** | **-6.77%** |

这是本轮 oracle 指标退化的直接原因。Top-5 responsibility 的平均熵为：

- General：1.550；
- Challenge：1.551；
- 五个候选完全均匀时为 `ln(5)=1.609`。

也就是说，五个被选中的 mode 获得了近似均匀的回归梯度。对单一 GT 同时回归五个 mode，会自然地把它们拉向同一轨迹。现有 diversity loss 权重不足以抵消这股收缩力量。

因此问题不是普通超参数波动，而是 **Top-5 soft trajectory regression 与 best-of-20 多模态覆盖目标存在结构性冲突**。

## 4. Anchor coverage 全时段退化

| Split | Step 4 | Step 8 | Step 12 | Endpoint |
|---|---:|---:|---:|---:|
| General | +8.964% | +7.350% | +6.189% | +5.080% |
| Challenge | +1.215% | +1.993% | +3.548% | +1.609% |

从第 4 步到终点没有任何一致改善。General 的早期覆盖退化最严重；Challenge 的退化随预测时域增加而扩大。MA1 并非只损伤终点或某一个锚点，而是整体候选几何覆盖变差。

## 5. Mode usage：没有死模式，但出现职责集中

### General

- Winner entropy：2.763 → 2.751，略下降。
- 占比超过 5% 的 modes：9 → 7。
- Never-win modes：0 → 0。
- Mode 10 独自承担 20.5% 的 oracle winners。

### Challenge

- Winner entropy：2.939 → 2.965，略改善。
- 占比超过 5% 的 modes：8 → 9。
- Never-win modes：0 → 0。

所以 MA1 没有造成离散意义上的完全 mode collapse，特别是 Challenge 的 winner 使用率仍均衡。但几何 diversity 已明显下降。这说明“每个 mode 偶尔仍能获胜”和“不同 modes 保持足够远的轨迹覆盖”不是一回事；只统计 winner entropy 会漏掉这种连续空间收缩。

## 6. Assignment 没有可靠学成

| 指标 | General | Challenge | 参考值 |
|---|---:|---:|---:|
| Predicted Top1 accuracy | 7.65% | 6.33% | 随机 5% |
| Mean Top1 mass | 13.01% | 9.66% | 均匀 5% |
| Mean Top3 mass | 34.92% | 27.72% | 均匀 15% |
| Predicted entropy | 2.683 | 2.802 | `ln(20)=2.996` |
| Target → predicted KL | 1.064 | 1.317 | 0 为完全一致 |

Assignment 只略高于随机，Challenge 尤其弱。模型没有形成“给定社会状态后稳定选择某类 mode”的能力。

## 7. Social compatibility 几乎没有形成组间差异

| 指标 | General | Challenge |
|---|---:|---:|
| Needed vs non-needed responsibility JS | 0.00665 | 0.00079 |
| Needed vs non-needed winner JS | 0.02261 | 0.00369 |

JS divergence 接近零，Challenge 几乎完全相同。这表明 social-needed 和 non-needed 样本触发的责任分布没有实质区别。所谓 interaction-aware matching 最终主要表现为 geometry-driven Top-5 soft matching，社会条件化部分没有建立可测量的 specialization。

## 8. 训练曲线排除 checkpoint 偶然性

- General 从第 3 轮开始 minADE 已为 0.19961，之后最好仅恢复到 0.19738，始终远差于 B0 的 0.17868。
- Challenge 第 3/6/9/12 轮 minADE 分别为 0.28290/0.28396/0.28337/0.28305，始终差于 B0 的 0.27807。
- 训练 loss 持续下降，但 oracle coverage 没有恢复。

因此失败从训练早期就已形成，不是最终 checkpoint 选错，也不是训练轮数不足。继续训练更久没有合理的改善依据。

## 9. 参数代价

- B0：355,048 parameters。
- MA1：357,961 parameters。
- 新增：2,913 parameters，约 **+0.82%**。

结构仍然轻量，但精度和多样性明显退化，因此不能保留。

## 10. 晋级判定

| 标准 | 判定 |
|---|---|
| Overall minADE 或 minFDE 改善 ≥0.5% | **失败** |
| Social-needed 子集改善 ≥2% | **失败，反向退化 3%–10%** |
| Challenge 复现改善方向 | **失败** |
| Winner usage 不更集中 | General 失败，Challenge 通过 |
| Social-needed responsibility 有可测差异 | **失败** |
| Diversity 不明显下降 | **失败** |
| 中后段 anchor coverage 一致改善 | **失败** |

综合判定：**MA1 不晋级，不跑 Seed2/3。**

## 11. 后续研究判断

不建议直接执行原建议中的 MA2（去掉 social）、MA3（endpoint-only）或继续扫 Top-M/temperature。当前证据已经指出主要矛盾：让多个 modes 对同一个 GT 承担轨迹回归，会牺牲 set coverage；而 social compatibility 本身又没有学出组间差异。

如果仍要对该路线保留一次最小机制验证，只应考虑：

> **单 winner 回归保持不变，social information 只在几何近邻候选之间做 tie-breaking / mode identity regularization，绝不让 Top-5 同时回归同一 GT。**

但从寻找接近 10% 正向提升的目标出发，更合理的主线是转向：

1. **Candidate-set optimization**：直接优化整组候选的覆盖，而不是复制单 GT 梯度。
2. **Distribution-level objective**：让候选拟合未来分布，并显式保留候选间排斥或 coverage budget。
3. **Mode prototype / template specialization**：通过跨样本聚类或行为原型建立稳定 mode identity，而不是在单样本内部用 Top-5 soft matching 临时分配。

MA1 给出的最有价值结论是：

> 当前瓶颈不是“责任分配还不够软”，而是多候选训练必须保护集合覆盖；任何同时拉动多个 mode 追逐单一 GT 的目标都会与 minADE/minFDE 的 best-of-K 优势相冲突。

## 数据来源

- `seed1_summary.json`：严格匹配的 B0/MA1 总体指标和参数量。
- `diagnostic/combined_summary.json`：social-needed、winner usage、assignment、coverage 和 diversity 诊断。
- General/Challenge `console.log`：各验证轮训练曲线。
