# 文献机制到本地实验的映射

## 检索范围

本轮首先读取用户 Obsidian 中的 BiFormer、Attending to Graph Transformers、MART、Near-Aware Attention、Sparse Trajectory Prediction、随机非自回归 Transformer、Forecast-MAE 等笔记，再通过 OpenAlex/CrossRef 聚合检索和论文官方页面核对。选择标准不是“方法新”，而是能否保持 K20、稀疏图、运动学先验和约 0.3M 参数规模。

## 已转化并实测的机制

| 文献机制 | 本地转化 | 实验 | 结论 |
|---|---|---|---|
| Bi-Level Routing Attention：粗粒度内容路由后精细注意力 | 低维节点+关系路由选择 Top-k 行人，再执行关系注意力 | EXP-022/034 F13 | seed=1 有信号，三种子不稳定，淘汰 |
| Relational Transformer / MART：关系进入 Q/K、节点边协同更新 | 关系 Q/K 位移；节点—边—节点二次传播 | EXP-024 F15、EXP-027/037 F18 | 二次传播失败；Q/K 单种子好但多种子性价比不足 |
| α-entmax 稀疏注意力 | sparsemax 与 entmax-1.5 | EXP-026/028/031/033/036 | 稳定改善 minADE 覆盖，但最终排序不如 F16 |
| Agent Attention：少量代理 token 中转全局信息 | 两个群体代理，O(NP) 汇聚与广播 | EXP-030 F21 | 全局均值稀释具体邻居关系，失败 |
| Graph Transformer structural encoding | 稀疏度数、平均距离、风险和接近速度节点编码 | EXP-029 F20 | 与已有边 token 信息重复，失败 |
| 跨帧/时空联合交互 | agent Query 直接读取稀疏邻居全部历史时刻 | EXP-025/035 F16 | **三种子稳定显著提升，锁定** |
| DropKey | 训练期随机删除 10% 邻居—时刻 key | EXP-038 F24 | 8 帧历史冗余不足，失败 |

## 关键来源

- [BiFormer: Vision Transformer With Bi-Level Routing Attention](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_BiFormer_Vision_Transformer_With_Bi-Level_Routing_Attention_CVPR_2023_paper.html)，CVPR 2023。核心是内容感知的两级动态稀疏路由。
- [EfficientViT: Memory Efficient Vision Transformer With Cascaded Group Attention](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html)，CVPR 2023。用于评估多头冗余与轻量化分组设计；结合 F14 结果，没有继续堆叠级联头。
- [Agent Attention: On the Integration of Softmax and Linear Attention](https://doi.org/10.1007/978-3-031-72973-7_8)，ECCV 2024。启发 EXP-030 的低秩全局代理通路。
- [Attending to Graph Transformers](https://arxiv.org/abs/2302.04181)，2023。用于区分节点、边、相对和全局结构编码，并启发 EXP-029。
- [MART: MultiscAle Relational Transformer Networks for Multi-agent Trajectory Prediction](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08288.pdf)，ECCV 2024。启发关系进入 Q/K 以及节点—边协同更新。
- [Stochastic Non-Autoregressive Transformer-Based Multi-Modal Pedestrian Trajectory Prediction](https://ieeexplore.ieee.org/document/10367756)，IEEE T-ITS 2024。其 α=1.5 entmax 是 EXP-028/031/033 的直接依据。
- [Sparse Trajectory Prediction](https://pubmed.ncbi.nlm.nih.gov/41171662/)，IEEE TPAMI，在线发表于2025、卷期为2026。其动态不规则交互和 early-sparsity 支持保持稀疏结构的总体方向。
- [Forecast-MAE](https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_Forecast-MAE_Self-supervised_Pre-training_for_Motion_Forecasting_with_Masked_Autoencoders_ICCV_2023_paper.html)，ICCV 2023。其遮挡训练思想经过任务适配形成 Temporal DropKey，但本地结果表明短轨迹不适合随机删除关键时刻。

## 迁移结论

跨领域方法必须按数据结构重新解释。视觉中的区域路由和代理 token 面对大量冗余 patch 很有效，但 ETH/UCY 的节点少、历史仅 8 帧、有效交互高度局部，因此硬路由、全局代理和随机删 key 均容易损失关键信息。

最成功的迁移并不是替换一种“更新的 attention”，而是修正信息组织方式：保留原稀疏邻居图，让目标摘要按内容查询邻居每个历史时刻。它保留全部物理边先验，同时避免过早时间压缩。
