# H3-to-V2 Correction Attribution Audit 结果分析

## 1. 完整性

- 五折、seed 1、零训练审计完整。
- H3 coarse 在 source-inner 上逐折复现 H3 official metrics，最大绝对误差 `8.94e-8`。
- V2 corrected 在 outer-test 上逐折复现 EXP-164，最大绝对误差 `5.96e-8`。
- H3/V2 的 20 个 mode index、三锚点与完整轨迹坐标通过运行时一致性断言。

## 2. V2总体有效，但收益明显偏向普通样本

全体样本上，H3→V2 的 macro 改善约为：

- minADE：4.12%
- minFDE：5.47%
- J：4.74%

分组后：

| Group | minADE improvement | minFDE improvement | J improvement |
|---|---:|---:|---:|
| High acceleration | 2.36% | 2.77% | **2.57%** |
| Non-high acceleration | 7.08% | 14.85% | **10.30%** |

高加速度 J 收益只有普通样本的约四分之一。自动判定为 `high_acc_correction_underbenefit`。

分折 high-acc J 改善：

- ETH：-0.40%
- Hotel：+1.12%
- Univ：+1.97%
- Zara01：+5.44%
- Zara02：+4.70%

ETH high-acc 中 V2 同时使 minADE 退化 0.14%、minFDE 退化 0.74%，不是统计口径造成的单指标冲突。

## 3. V2通过“重排mode几何”获得oracle收益

五折 macro same-mode 归因：

| Group | 修正原H3 ADE winner | 修正最终V2 ADE winner | ADE winner changed | 改善mode比例 |
|---|---:|---:|---:|---:|
| High acceleration | **-0.0061** | +0.0200 | 18.4% | 72.6% |
| Non-high acceleration | +0.0001 | +0.0216 | 28.9% | 84.7% |

在高加速度样本中，原H3最佳ADE mode经过V2修正后平均反而变差。V2的正收益来自另一部分mode被修好，随后成为新的oracle winner。

ETH high-acc最典型：

- 原ADE winner same-mode gain：-0.0161
- 最终V2 winner same-mode gain：+0.0162
- 原FDE winner same-mode gain：-0.0555
- 最终V2 FDE winner same-mode gain：+0.0560

因此不能把V2描述成“对H3最佳轨迹精细化”。更准确的机制是：

> V2对整个候选集合做有偏几何重排，牺牲部分原winner，同时把其他mode推成新winner。

## 4. 高加速度上的分时域收益不足

point-oracle gain五折macro：

| Group | t4 | t8 | t12 |
|---|---:|---:|---:|
| High acceleration | 0.0000 | 0.0072 | 0.0117 |
| Non-high acceleration | -0.0020 | 0.0169 | 0.0272 |

t4几乎没有oracle收益；主要收益来自t8/t12，但高加速度收益仍不到普通样本的一半。

高加速度模式被改善的比例在各时刻约为70%，看似不低，但修正没有集中到当前最有价值的winner附近。ETH high-acc 的t4/t8/t12 point-oracle gain全部为负，与上一轮候选距离审计一致。

这说明当前训练目标能让大量mode获得小幅改善，却没有充分保护或优化困难样本的关键候选。

## 5. correction后评分发生系统性错位

score logits由修正前的`mode + source_anchors`计算，V2 correction之后没有同步更新。

五折macro：

| Group | Source ADE winner score rank | Corrected ADE winner score rank | Top1 ADE gain |
|---|---:|---:|---:|
| High acceleration | 6.11 | 6.57 | -0.0040 |
| Non-high acceleration | 3.55 | 4.05 | -0.0011 |

修正后的oracle winner平均评分名次更差，top1 ADE也总体恶化。ETH high-acc 的top1命中率从8.0%降至6.2%。

评分错位是真实结构问题，但它不是当前minADE/minFDE的主要原因，因为min指标不使用score选候选。此前独立score calibration/reranker实验也没有产生足够收益，所以本轮不重新回到评分主线。

## 6. same-path与min指标为何在ETH出现分离

ETH high-acc：

- minADE：-0.14%（退化）
- minFDE：-0.74%（退化）
- same-path FDE：+3.33%（改善）

V2改善了一部分ADE winner对应轨迹的终点一致性，但候选集合中最好的ADE和独立最好的FDE反而略差。这正是mode重排和winner切换造成的结果，不代表同一个候选同时改善了三个指标。

## 7. 机制结论

结合前两轮审计：

1. H3 source proposal的长期运动响应不足，t12只有GT响应斜率的64.4%。
2. V2 corrected只提高到66.4%，没有修复动态范围压缩。
3. V2对普通样本收益很大，对高加速度收益明显不足。
4. correction不是cap受限，而是没有把修正预算分配给困难样本的关键mode。
5. 原winner常被牺牲，新winner又与冻结score不同步。

因此短板同时存在于proposal和refiner，但更上游、更根本的问题是proposal对中长期运动强度响应不足；refiner只能有限补偿，并引入winner/score错位。

## 8. 下一训练实验建议

审计链已经足以允许一次受控训练实验，不需要继续增加零训练诊断。

优先实验应是 `Horizon-Resolved Motion-Conditioned Proposal Residual`，而不是直接改score或扩大refiner：

- 冻结F16、social branch、H3主体、现有V2 refiner和score；
- 在source anchors之后、V2 refiner之前加入零初始化小型连续residual；
- 显式输入无量纲运动特征，例如加速度/速度比、short-long velocity差、turn/nonlinearity和现有decoder context；
- 分别输出t4/t8/t12的小型二维残差，t4最小、t8/t12更有表达力；
- 不使用high-acc hard gate、不使用scene id、不扩大现有cap；
- 不做统一radial scaling，而是mode-wise二维方向修正；
- 第一轮只训练该adapter，score保持冻结，首先验证minADE/minFDE oracle几何。

必须同时保留一个`correction-only motion conditioning`对照，但不应第一步直接组合两者。若proposal adapter无法提高high-acc和macro min指标，再停止该路线，而不是继续加大模块。
