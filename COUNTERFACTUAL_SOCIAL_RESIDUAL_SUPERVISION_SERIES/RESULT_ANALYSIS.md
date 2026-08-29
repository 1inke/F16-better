# CSRS1结果分析

## 1. 最终结论

CSRS1未达到seed1晋级标准，**不进行seed2/3复验，也不通过调整辅助损失权重继续搜索**。

实验不是简单地表明“Need和Residual都学不会”。Need分类明显优于随机，Residual方向也能很好拟合；真正失败的是将监督信号转化为多模态候选修正的环节。具体存在两个结构问题：

1. `GT-best Self-Only candidate`残差被叠加到已经使用Full Social编码的候选上，可能重复补偿已有社会修正。
2. mode gate没有候选分配监督，最终反而在non-needed样本上施加更大的修正，并在Challenge中接近模式塌缩。

因此失败层级位于：**target/base一致性 + interaction-aware mode allocation**，而不是继续扩大编码器或残差头。

## 2. 主指标

负变化率代表改善。

| 划分 | 方法 | minADE20 | minFDE20 | same-path FDE | Top-1 ADE | Top-1 FDE |
|---|---|---:|---:|---:|---:|---:|
| General | B0 | 0.17868 | 0.31407 | 0.37851 | 0.46739 | 1.02615 |
| General | CSRS1 | 0.18015 | 0.31692 | 0.37816 | 0.46815 | 1.02869 |
| General | 变化 | **+0.824%** | **+0.907%** | -0.093% | +0.164% | +0.247% |
| Challenge | B0 | 0.27807 | 0.47784 | 0.56940 | 0.75006 | 1.61948 |
| Challenge | CSRS1 | 0.27882 | 0.47772 | 0.56834 | 0.74840 | 1.61734 |
| Challenge | 变化 | **+0.269%** | -0.026% | -0.186% | -0.221% | -0.132% |

General两个主要oracle指标均退化；Challenge只有FDE等指标极小改善，minADE仍退化。变化远低于整体0.5%或social-needed子集2%的晋级要求。

## 3. Actual social-needed子集

| 划分 | positive样本 | minADE变化 | minFDE变化 | same-path FDE变化 |
|---|---:|---:|---:|---:|
| General | 628 | +0.816% | +1.134% | +1.009% |
| Challenge | 1412 | -0.001% | -0.300% | -0.175% |

CSRS1在真正根据反事实收益定义的正样本上也没有获得目标中的2%改善。General甚至比整体结果更差，因此问题不能归结为负样本稀释。

## 4. Need分类：能区分，但强度不足

| 指标 | General | Challenge |
|---|---:|---:|
| 正样本率 | 41.42% | 46.57% |
| AUROC | 0.627 | 0.643 |
| AUPRC | 0.515 | 0.580 |
| Positive recall | 0.315 | 0.513 |
| Negative recall | 0.798 | 0.665 |
| Need概率与真实gain Spearman | 0.065 | 0.125 |

AUROC/AUPRC明显高于随机基准，说明social token包含一部分“是否需要社会信息”的线索。但概率与连续gain的相关性很弱，尤其General正类召回只有31.5%。Need不是完全失败，但不足以承担精细控制。

正负样本平均need probability分别为：

- General：0.453 / 0.400。
- Challenge：0.496 / 0.428。

存在分离，但间隔不大，概率集中在0.4–0.5附近，并没有形成可靠的开关。

## 5. Residual：高余弦不等于有效社会修正

| 指标 | General | Challenge |
|---|---:|---:|
| positive residual cosine | 0.792 | 0.944 |
| non-needed residual cosine | 0.719 | 0.952 |
| cosine vs minADE变化相关 | -0.044 | +0.064 |
| cosine vs minFDE变化相关 | -0.068 | +0.084 |

Challenge中non-needed样本余弦甚至略高于positive样本；方向越准确也没有带来更低的ADE/FDE，Challenge中的相关方向还是相反的。

教师目标进一步说明了原因：

- General residual target范数：positive 1.73、non-needed 1.40；范数与gain相关仅0.034。
- Challenge：positive 2.856、non-needed 2.861；范数与gain相关仅0.005。

`GT-best Self-Only candidate`仍然主要描述一般候选剩余误差，而不是由社会信息造成的增量。它比IR1的惯性残差容易预测，但没有成为有效的社会控制量。

## 6. Mode gate发生错误分配

| 划分 | positive修正量 | non-needed修正量 | mode gate均值 | mode gate标准差中位数 |
|---|---:|---:|---:|---:|
| General | 0.0149 | **0.0200** | 0.121 | 0.0074 |
| Challenge | 0.0148 | **0.0159** | 0.040 | 0.0036 |

虽然Need概率在positive样本上更高，但mode gate在positive样本上更低，最终抵消了Need head。Challenge中20个模式间的gate差异几乎消失，说明所谓mode-specific correction实际接近一个很弱的共享修正。

根本原因是mode gate没有直接监督。Residual标签对应一个由GT选出的Self-Only赢家，但训练没有告诉mode gate该残差应该分配给哪个候选，只依赖间接轨迹梯度自行发现匹配关系。

## 7. Anchor coverage

General第4/8步覆盖改善0.48%/0.39%，但终点退化0.91%；Challenge第8步改善0.25%，第12步退化0.13%，终点基本不变。说明局部锚点移动存在，但没有形成一致的整条路径收益。

## 8. 下一步研究判断

不继续CSRS1原结构。下一轮应同时修复以下两点：

1. **Residual相对Full Social候选定义。** 对B0 Full候选选择联合oracle赢家，学习`GT anchor - best Full anchor`的剩余修正，避免把Self-Only所需修正重复叠加到Full候选上。
2. **显式候选分配监督。** 使用Full oracle赢家或基于候选联合代价的soft responsibility监督mode assignment；修正只作用于被分配的Top-1/Top-M候选，其余模式保持不变以保护多样性。
3. 对non-needed样本增加no-op约束，直接惩罚锚点修正幅度，防止mode gate绕过Need head。
4. 第一轮仍只做B0/新方案seed1，先验证positive样本修正量必须高于negative、mode assignment具有明显准确率、alignment与ADE/FDE收益建立负相关，再看整体指标。

这对应建议中的情况C与部分情况B：模型能识别一部分need并预测残差方向，但当前target和mode assignment无法把它转化为更好的多模态候选集。
