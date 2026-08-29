# Tail-Candidate / Candidate-Family Audit正式结果

## 1. 最终结论

TCA给出了预设的**情况B**：tail候选具有很高的实际价值，但tail-needed future没有形成比普通future更清晰、可直接替换现有tail的全局簇。

因此：

- `hcb1_training_allowed = false`；
- 不编写、不训练当前建议形式的HCB1；
- 不用4–6个训练集cluster medoid替换现有tail candidates；
- 当前decoder/candidate family应冻结，研究重心回到encoder、场景/社会条件信息、轻量化和完整稳健性验证。

这一结论不表示tail没有用；恰恰相反，现有K=11–20在困难样本上非常重要。失败的是“用少量静态rare-branch模板概括tail”的假设。

## 2. Tail真实边际价值

### Validation

| Split | 指标 | tail产生正收益的样本比例 | 全样本平均收益 | 命中样本平均收益 |
|---|---|---:|---:|---:|
| General | ADE@10→20 | 19.92% | 0.0605 | 0.3037 |
| General | FDE@10→20 | 20.45% | 0.1500 | 0.7337 |
| Challenge | ADE@10→20 | 34.83% | 0.1323 | 0.3798 |
| Challenge | FDE@10→20 | 33.84% | 0.3200 | 0.9457 |

General的median和P75均为0，说明tail属于“约20%样本的大收益”，不是所有样本的小修正。Challenge中tail需求升到约35%，说明运动困难分布对完整20候选尤其依赖。

### Train→Challenge分布偏移

- Challenge训练折tail正收益率约19%–20%；
- motion-challenge validation升到约34%–35%。

这说明最需要tail的未来在训练折中相对稀少，在困难验证集显著富集。它首先是稀有行为泛化问题，而不是简单的候选数量问题。

## 3. Social-needed

| Split | 子集 | ADE tail正收益率 | 命中ADE收益 | FDE tail正收益率 | 命中FDE收益 |
|---|---|---:|---:|---:|---:|
| General | social-needed | 20.86% | 0.3407 | 21.82% | 0.8071 |
| General | non-needed | 19.26% | 0.2753 | 19.48% | 0.6756 |
| Challenge | social-needed | 35.20% | 0.3916 | 34.14% | 0.9896 |
| Challenge | non-needed | 34.51% | 0.3693 | 33.58% | 0.9067 |

social-needed对tail略更依赖，General差异更明显。但tail并非只由社会交互触发；non-needed同样有大量tail winner，说明运动变化和场景/domain shift也很关键。

## 4. 聚类结果

| Split | material tail训练样本比例 | 最佳cluster数 | silhouette | 多初始化ARI | 最小簇比例 | 随机全future silhouette |
|---|---:|---:|---:|---:|---:|---:|
| General | 21.02% | 4 | 0.298 | 0.964 | 16.03% | **0.436** |
| Challenge | 20.23% | 6 | 0.291 | 0.992 | 12.70% | **0.409** |

K-means结果在不同初始化之间很稳定，但tail silhouette明显低于随机抽取的普通future。稳定性来自大尺度方向/速度划分，而不是tail形成了更紧密、更独特的rare-behaviour类别。

cluster统计呈连续变化：

- history speed约0.15–0.38；
- future turn约0.31–1.15 rad；
- future speed change从轻微减速到加速；
- 各簇大量由scene 2样本主导。

因此它们更像连续的速度—转向谱和数据集场景不平衡，而不是可以稳定替代tail的离散语义模板。

## 5. 锁定Validation HCB oracle proxy

Proxy保持总候选数20：保留score top-10，用训练tail cluster medoid替换部分原tail，其余位置保留原tail。

| Split | 组成 | B0 minADE/FDE | Proxy minADE/FDE | 相对变化 |
|---|---|---:|---:|---:|
| General | core10 + 4 medoid + 6原tail | 0.17868 / 0.31407 | 0.18623 / 0.32510 | **退化4.23% / 3.51%** |
| Challenge | core10 + 6 medoid + 4原tail | 0.27807 / 0.47784 | 0.31217 / 0.53265 | **退化12.26% / 11.47%** |

这是禁止HCB1的决定性证据。即使不训练模型、直接把训练集真实future medoid作为候选，仍远不如现有learned tail。Challenge退化尤其严重，因为训练集中占主导的中心簇无法覆盖motion-challenge中的稀有尺度、转向和速度组合。

## 6. 为什么“聚类稳定”仍不能训练HCB1

1. 人类轨迹天然能按左右、速度和曲率聚类，稳定K-means并不等于这些簇对tail有新增价值。
2. tail future的silhouette反而低于普通future，说明tail内部更连续、更复杂。
3. 全局medoid丢失当前输入决定的运动尺度、交互响应和场景条件。
4. 替换原tail会牺牲少数样本上的大幅oracle收益；这些收益正是K=20优于K=10的主要来源。
5. Challenge的train/validation tail需求明显漂移，静态训练基底难以外推。

## 7. 对建议的最终决策

### 不执行

- 不实现10 Core + 10静态Tail Basis的HCB1；
- 不调整cluster数后反复测试validation；
- 不用手工左转/右转/加减速模板代替数据簇；
- 不继续decoder loss和candidate basis的局部扫描。

### 保留

- TCA作为论文中的候选边界诊断；
- CDA1作为“改善K≤10但不改善K=20”的机制消融；
- 现有20候选配置，因为tail在20%–35%的样本上提供不可替代的大收益。

### 下一阶段

当前证据支持的重点是：

1. 保留现有decoder和K20；
2. 回到F16 sparse social encoder及其严格dense/simple对照；
3. 对困难样本增加更有条件性的输入信息，尤其是场景、运动状态和社会交互，而不是静态tail模板；
4. 完成多seed、ETH/UCY five-fold、SDD大规模agent测试；
5. 完成参数、FLOPs、latency、VRAM、edge ratio和N-scaling轻量化证据。

## 8. 最重要的研究判断

**当前tail不是缺少四到六种可命名的运动模板，而是缺少对连续、条件相关、训练中稀有的困难未来的泛化能力。**

现有tail已经比静态训练簇更强。若未来再次修改candidate family，必须引入新的条件信息来源或真正的多未来监督，而不能只重排单GT训练得到的全局basis。
