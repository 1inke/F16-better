# Motion-to-Candidate Transfer Audit 结果分析

## 1. 完整性

- ETH、Hotel、Univ、Zara01、Zara02 五折，seed 1，正式审计完整。
- 未训练或修改模型，复用 EXP-164 official F16+H3+V2。
- minADE、minFDE、same-path FDE 逐折复现，最大绝对误差 `5.96e-8`。
- corrected anchor 与最终轨迹 t4/t8/t12 坐标在运行时执行一致性断言。

## 2. 高加速度样本的候选最近距离显著增大

五折 macro：

| Horizon | High-acc nearest | Non-high nearest | High-acc outside box | Non-high outside box |
|---|---:|---:|---:|---:|
| t4 | 0.136 | 0.076 | 19.4% | 10.9% |
| t8 | 0.281 | 0.153 | 7.7% | 5.0% |
| t12 | 0.468 | 0.246 | 7.7% | 2.7% |

高加速度样本在三个 horizon 的最近候选误差约为普通样本的 1.8–1.9 倍。五折方向一致，说明前一轮发现的高加速度误差问题确实传递到了候选几何。

但 t8/t12 大部分 GT 仍处在 20 个候选的纵横轴向包络内。因此主要问题不是整个 envelope 完全不够宽，而是有限候选在包络内部的方向、落点和密度配置不理想。

## 3. V2 correction 对困难样本系统性欠响应

| Horizon | High-acc correction norm | Non-high norm | High-acc cap hit | Non-high cap hit | High-acc oracle gain | Non-high gain |
|---|---:|---:|---:|---:|---:|---:|
| t4 | 0.077 | 0.114 | 2.3% | 6.2% | 0.0000 | -0.0020 |
| t8 | 0.151 | 0.206 | 23.2% | 49.4% | 0.0072 | 0.0169 |
| t12 | 0.262 | 0.341 | 17.5% | 35.7% | 0.0117 | 0.0272 |

correction norm 与加速度在所有 inner/outer fold、所有 horizon 中均为负相关。outer macro Spearman 约为：

- t4：-0.469
- t8：-0.487
- t12：-0.404

也就是说，运动越剧烈，refiner 使用的修正通常越小。与此同时，高加速度候选离 GT 更远，说明这种缩小不是因为困难样本已经被 source proposal 解决。

更重要的是，高加速度 cap-hit 反而显著更低。因此问题不是 correction budget 太小，也不能通过继续扩大 `[0.20, 0.25, 0.50]` 上限解决。当前 refiner 没有主动使用已有预算。

ETH 是最强失败案例：高加速度样本 t4/t8/t12 的 correction oracle gain 分别约为 -0.015/-0.002/-0.005，修正后最近候选反而略差。

## 4. 候选对加速度的响应随 horizon 被压缩

将 outer-fold 中“候选平均半径对加速度的斜率”与“GT 半径对加速度的斜率”比较：

| Horizon | H3 source / GT | V2 corrected / GT |
|---|---:|---:|
| t4 | 87.6% | 93.0% |
| t8 | 74.4% | 78.0% |
| t12 | 64.4% | 66.4% |

五个 outer fold 的 t12 corrected/GT 比值分别为：

- ETH：72.2%
- Hotel：66.2%
- Univ：59.6%
- Zara01：78.0%
- Zara02：56.1%

方向完全一致：预测 horizon 越长，候选几何对运动强度的响应越不足。V2 correction 相比 source proposal 只把 t12 macro 比值从 64.4% 提高到 66.4%，没有修复长期动态范围压缩。

## 5. 不是简单的“半径太小”

t12 五折 macro：

- high-acc candidate mean radius：3.842
- high-acc GT radius：3.928
- non-high candidate mean radius：2.497
- non-high GT radius：1.955

候选对普通样本整体偏远，而对高加速度样本接近或略短。这是从普通运动到剧烈运动的动态范围被压缩到中间区域，而不是所有样本统一缩短。

同时，高加速度 GT 大多仍在宽包络内，但最近候选距离很大。因此仅乘一个 global scale、扩大 lateral spread、增加 endpoint repulsion 或扩大 correction cap 都不对症；这些方向也与此前已关闭的实验重复。

## 6. 机制定位

结合前三轮审计：

1. Full Social 在 ETH-LDHA 中有益，不是交互过强。
2. robust/temporal/final encoder 中可以解码加速度，不是运动信息完全缺失。
3. H3 proposal 和 V2 correction 都没有把该信息充分转换成 horizon-resolved candidate geometry。
4. V2 refiner 的主要收益反而集中在普通样本，在困难样本上修正更小、收益更弱。

当前短板可定位为：

> decoder 对已编码运动状态的条件响应发生动态范围压缩，尤其集中在 t8/t12；同时离散模式的局部方向/落点配置不足。

## 7. 下一步边界

在训练新模块前，还需用一次零训练组件归因明确回答：

- H3 source trajectory 与 V2 corrected trajectory 在 high-acc/non-high 上各自的 minADE、minFDE、J；
- correction 改变的是 oracle winner 还是同一路径质量；
- ETH 高加速度上的负 anchor gain 是否会转化为完整轨迹指标退化；
- t4/t8/t12 中哪一个 correction 应保留、削弱或重新条件化。

该实验应称为 `H3-to-V2 Correction Attribution Audit`，直接复用一次推理同时计算 coarse 与 corrected 全轨迹，不训练模型。

若归因确认 V2 在 high-acc 上系统性无效或有害，下一训练实验才允许使用显式、连续、零初始化的 horizon-resolved motion conditioning。它必须：

- 冻结 F16、social branch 和 score head；
- 不使用 hard high-acc gate；
- 不扩大 cap；
- 不做全局 radial scale；
- 只重构 proposal/refiner 对现有运动特征的连续响应；
- 先做 proposal-only 与 correction-only 隔离对照。
