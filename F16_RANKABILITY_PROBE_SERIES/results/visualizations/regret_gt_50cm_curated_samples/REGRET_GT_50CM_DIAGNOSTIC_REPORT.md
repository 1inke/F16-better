# F16 重度选择失败样本（Regret > 0.50m）结构化诊断报告

- **分析数据源**：`F16_RANKABILITY_PROBE_SERIES/results/cache/val_seed1.npz` (ETH Validation)
- **验证集总样本数**：1516
- **Regret > 0.50m 总样本数**：362（占比 23.88%）
- **精选渲染样本数**：25（去重唯一数：20）

## 核心模式分布与样例索引

### 1_Extreme_Regret_Disasters

**特征描述**：Highest regret samples in the entire dataset (Regret 1.86m ~ 2.21m)

| Sample ID | Cache Index | Regret (m) | Selected ADE (m) | Oracle minADE (m) | Spread (m) | Oracle Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 30274 | 1483 | 2.213 | 2.976 | 0.763 | 4.17 | 19/20 |
| 30272 | 1481 | 2.100 | 2.912 | 0.813 | 4.28 | 19/20 |
| 30263 | 1472 | 1.927 | 4.336 | 2.408 | 2.92 | 20/20 |
| 29408 | 617 | 1.876 | 2.428 | 0.551 | 2.63 | 16/20 |
| 30185 | 1394 | 1.861 | 2.787 | 0.926 | 3.97 | 17/20 |

### 2_Clean_Selector_Failures

**特征描述**：Pattern C: Generator covers GT cleanly (Oracle ADE < 15cm), yet selector fails severely

| Sample ID | Cache Index | Regret (m) | Selected ADE (m) | Oracle minADE (m) | Spread (m) | Oracle Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 842 | 9 | 1.077 | 1.206 | 0.129 | 2.40 | 18/20 |
| 29452 | 661 | 0.998 | 1.142 | 0.144 | 3.05 | 15/20 |
| 30129 | 1338 | 0.894 | 1.020 | 0.126 | 3.47 | 9/20 |
| 29640 | 849 | 0.819 | 0.860 | 0.041 | 2.27 | 12/20 |
| 29832 | 1041 | 0.735 | 0.851 | 0.115 | 3.88 | 7/20 |

### 3_High_Spread_FanOut

**特征描述**：Pattern B: Diversity pushes endpoints far apart (Spread > 3.8m); oracle hit is a lucky outlier

| Sample ID | Cache Index | Regret (m) | Selected ADE (m) | Oracle minADE (m) | Spread (m) | Oracle Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 30267 | 1476 | 1.323 | 2.862 | 1.539 | 4.32 | 13/20 |
| 30272 | 1481 | 2.100 | 2.912 | 0.813 | 4.28 | 19/20 |
| 30270 | 1479 | 1.442 | 3.059 | 1.617 | 4.24 | 13/20 |
| 30274 | 1483 | 2.213 | 2.976 | 0.763 | 4.17 | 19/20 |
| 30120 | 1329 | 0.676 | 0.971 | 0.295 | 4.16 | 9/20 |

### 4_Inverted_Confidence_Bottom_Ranks

**特征描述**：Selector inversion: Oracle mode is ranked at the very bottom (Rank 19 or 20 out of 20)

| Sample ID | Cache Index | Regret (m) | Selected ADE (m) | Oracle minADE (m) | Spread (m) | Oracle Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 30274 | 1483 | 2.213 | 2.976 | 0.763 | 4.17 | 19/20 |
| 30272 | 1481 | 2.100 | 2.912 | 0.813 | 4.28 | 19/20 |
| 30263 | 1472 | 1.927 | 4.336 | 2.408 | 2.92 | 20/20 |
| 30262 | 1471 | 1.703 | 3.627 | 1.925 | 2.93 | 20/20 |
| 9069 | 434 | 1.466 | 2.176 | 0.709 | 2.60 | 19/20 |

### 5_Compact_Cluster_Confusion

**特征描述**：Pattern A: Compact candidate bundle (Spread < 2.0m), but misaligned with actual maneuver

| Sample ID | Cache Index | Regret (m) | Selected ADE (m) | Oracle minADE (m) | Spread (m) | Oracle Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 29581 | 790 | 0.606 | 1.256 | 0.650 | 1.49 | 14/20 |
| 29664 | 873 | 0.541 | 0.678 | 0.137 | 1.56 | 17/20 |
| 29384 | 593 | 0.574 | 0.698 | 0.124 | 1.57 | 11/20 |
| 29370 | 579 | 0.530 | 0.593 | 0.063 | 1.60 | 11/20 |
| 9057 | 422 | 0.668 | 0.944 | 0.275 | 1.81 | 15/20 |

## 关键实证发现与研究指引

1. **高发散度扇形盲撒（Pattern B）占主导**：在 Regret > 0.5m 的样本中，有 106 个样本的候选终点 Spread > 3.0m。这印证了此前推测——强行通过多样性损失扩散终点，使 Oracle 往往只是极端离群点恰巧撞线，原模型本身对其毫无置信度（Score Rank 常在 15 名之后）。
2. **纯选择器失误（Pattern C）真实存在**：有 35 个样本的 Oracle ADE 小于 15cm（候选生成极为完美），但模型选择器依然选错了大于 50cm 的轨迹，这才是后处理重排与效用选择最有价值攻克的真实场景。
3. **置信度倒挂（Rank 19~20）**：在部分高危害样本中，真实最佳候选不仅没被选中，反而被原生分类器排在最底层，表明原生 logits 存在系统性偏见。
