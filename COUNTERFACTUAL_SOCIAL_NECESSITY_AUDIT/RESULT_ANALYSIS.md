# CSNA结果分析

## 1. 核心结论

CSNA证明：**当前F16社会上下文确实提供了有意义的预测信息，不能停止社会交互主线。** Full Social相对Self-Only在General与Challenge整体上均取得统计稳定的优势，尤其Challenge的minADE/minFDE分别改善2.54%和3.78%。

但实验同时否定了另一项假设：**当前手工interaction potential不能可靠表示“社会信息必要性”。** 收益没有随potential单调增加，逐样本potential与社会收益的Spearman相关几乎为零。因此不能按照原建议直接用potential高分样本构造或筛选CSRS监督。

脚本的自动结论为`do_not_enter_csrs`，原因是它严格要求“最高交互子集超过2%且十分位单调”。这里没有通过的主要是potential排序，而不是社会上下文本身。专业判断应为：**有条件进入改造后的CSRS seed1验证，但必要性标签和置信度必须直接来自Full-vs-Self实际收益，不能来自interaction potential。**

## 2. 整体指标

正的Full advantage表示Full Social误差更低。

| 划分 | 指标 | Full Social | Self-Only | Full优势 |
|---|---:|---:|---:|---:|
| General | minADE20 | 0.17952 | 0.18353 | 2.23% |
| General | minFDE20 | 0.31769 | 0.32454 | 2.16% |
| General | same-path FDE | 0.38101 | 0.38601 | 1.31% |
| General | Top-1 ADE | 0.46989 | 0.48182 | 2.54% |
| General | Top-1 FDE | 1.03276 | 1.05678 | 2.33% |
| Challenge | minADE20 | 0.27874 | 0.28583 | 2.54% |
| Challenge | minFDE20 | 0.48038 | 0.49852 | 3.78% |
| Challenge | same-path FDE | 0.57057 | 0.59414 | 4.13% |
| Challenge | Top-1 ADE | 0.75200 | 0.77252 | 2.73% |
| Challenge | Top-1 FDE | 1.62268 | 1.66530 | 2.63% |

聚类bootstrap的平均配对差95% CI均支持主要整体收益：

- General minADE `[0.00159, 0.00680]`，minFDE `[0.00112, 0.01435]`，组合代价 `[0.00241, 0.01386]`。
- Challenge minADE `[0.00519, 0.00922]`，minFDE `[0.01316, 0.02439]`，组合代价 `[0.01320, 0.02311]`。

这些区间均不跨零，说明平均收益不只是重叠窗口造成的随机波动。

## 3. 收益是稀疏且非均匀的

Overall social-help win rate约为49%–50%，但存在2.4%（General）和10.5%（Challenge）的完全平局样本。排除平局后：

- General组合代价条件胜率为49.97%，收益来自正向样本较大的平均改善，而不是多数样本小幅改善。
- Challenge组合代价条件胜率为56.30%，说明困难样本中社会上下文具有更一致的帮助。

5%截尾后，组合代价平均收益仍为General `0.00501`、Challenge `0.01640`，证明总体结果不是少量极端样本驱动。

这恰好支持“need classification”的必要性：社会信息不是对每个样本都应该以相同强度注入。

## 4. 现有interaction potential失效

interaction potential与逐样本社会收益的Spearman相关：

| 划分 | minADE收益 | minFDE收益 | 组合代价收益 |
|---|---:|---:|---:|
| General | 0.027 | -0.002 | 0.009 |
| Challenge | 0.059 | 0.014 | 0.041 |

所有相关性都接近零。十分位结果也明显非单调：Challenge的主要收益集中在第3–8十分位，最高第10十分位反而只有0.74% minADE和0.45% minFDE优势；General也在第3、4和第8十分位出现峰值。

因此，“距离较近/closing较强”不等于“模型必须进行社会修正”。极高potential中可能包含模型已能通过个体运动解释、稳定同行、或手工风险分数误判的样本。

## 5. 分场景结果

Challenge四个场景的组合代价平均社会收益全部为正：`0.04410 / 0.01208 / 0.03482 / 0.02102`。General前三个场景为正，第四个场景为`-0.00534`。

这说明社会分支在Challenge上的价值具有跨场景一致性，但General仍存在域相关的负迁移，后续need classifier必须有能力在不需要或有害的样本上关闭修正。

## 6. 对后续CSRS的修改要求

下一轮可以做CSRS seed1，但应采用以下版本：

1. 在训练集离线生成同checkpoint的Full/Self严格配对教师缓存。
2. 必要性标签只使用真实组合代价收益 `g=C_self-C_full`，不使用interaction potential。
3. 设置正类阈值`δ≈0.01`以过滤数值平局和极弱收益。本轮该阈值对应General约41.4%、Challenge约46.6%的正样本率，不会造成严重类别失衡。
4. 连续置信度建议从`c=clip(max(g,0)/0.10,0,1)`开始，使明确受益样本获得强监督。
5. social residual仍使用`GT-best self-only learned candidate`，但只在`g>0`样本上监督；负收益样本训练need gate关闭社会修正，而不是学习反向残差。
6. 推理阶段不运行Self-Only教师，不读取GT，只使用need probability控制社会残差注入。
7. seed1晋级仍要求整体minADE或minFDE至少改善0.5%，或者实际social-needed子集改善2%；另一个oracle指标不能明显退化。

## 7. 最终决定

- 不停止当前社会交互研究方向。
- 停止使用现有interaction potential作为必要性排序或监督权重。
- 进入一次经过上述修正的CSRS seed1验证，而不是直接多种子验证。
