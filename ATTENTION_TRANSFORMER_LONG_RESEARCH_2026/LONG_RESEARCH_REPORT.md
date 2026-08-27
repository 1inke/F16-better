# ASTRA 注意力与 Transformer 改良长调研报告

> 形成日期：2026-08-23  
> 目标：在尽量保持轻量化、运动学先验、稀疏交互图和固定 K20 解码协议的前提下，从目标检测、视觉 Transformer、视频理解、点云、图学习、NLP 和时间序列中筛选尚未被 ASTRA 实质验证、且可能改善 ETH/UCY minADE/minFDE 的机制。

## 一、结论先行

这轮调研最重要的结论不是“还有很多新型注意力可以换”，而是：**ASTRA 已经覆盖并淘汰了相当大一部分通用高效注意力路线，下一步必须从任务结构出发做定制迁移。**

当前值得正式进入实验的方向按优先级排列为：

1. **运动学引导的可变形时间—社会注意力（首选）**：针对每个“目标行人—邻居”对，预测少量连续历史采样时刻，始终保留最近状态，再只对这些时刻做稀疏注意力。它不同于已经失败的单行人加速度事件采样，因为采样位置由目标—邻居关系共同决定。
2. **Q/K 归一化的缩放余弦注意力（最低风险）**：只改变相似度度量，不改稀疏图、不删 token、不压缩 V。它可能缓解不同场景、不同速度尺度导致的注意力 logit 尺度漂移，增加参数几乎可以忽略。
3. **运动关系分组向量注意力（表达力候选）**：借鉴点云的 grouped vector attention，让不同运动通道组对同一邻居拥有不同权重，而不是每个 head 内所有通道共用一个标量权重。
4. **差分稀疏注意力（抗无关邻居候选）**：用两张 softmax 图之差抵消共同噪声；需采用半维双分支控制 QK 计算量，否则与轻量化目标冲突。
5. **自适应局部时间平滑（稳健性候选）**：在不减少 8 帧 token 的情况下，先让 K/V 吸收邻近帧信息，降低轨迹测量噪声和 Masked History 的不稳定性。

以下方向目前不应继续作为 ETH/UCY 主线：BiFormer 式粗到细路由、Agent tokens、NAT/固定窗口、GQA/MQA、Cascaded Group、线性注意力、普通 token merging、固定/全局 landmark、DropKey、关系类型 bias、二阶邻居摘要、扩大邻域、长序列专用的 Mamba/Hyena/NSA/HyperAttention。它们或者已经被等价/近似验证且失败，或者其收益条件与当前 `N≤9、T=8` 完全不匹配。

一个必须强调的判断是：**即使新注意力有效，合理预期也应是小到中等幅度提升，而不是靠换 attention 一次性解决当前 minADE/minFDE 的全部差距。** 已有实验表明，编码器信息最终仍受 K20 候选几何基底约束。本文只筛选“解码器外仍值得做”的方向，不否定以后处理候选几何瓶颈的必要性。

## 二、当前模型与证据边界

### 2.1 当前应作为参照的模型

当前精度主线为：

`F16 稀疏时间—社会 cross attention + Kinematic K20 + Coverage Loss 0.20 + EMA`

F16 的社会交互并不是普通的全局 self-attention。每个目标行人用聚合后的 agent 表示作为 Query，只查询稀疏图中邻居的 8 帧历史 K/V；关系 token 同时进入 attention bias 和 value message。当前主要尺度为 `N≤9、T=8、d=96、H=4`。

当前可确认的指标包括：

| 配置 | 参数量 | 主要内部结果 |
|---|---:|---|
| F16 + 50% 几何/速度增强 | 284,642 | scene 1 三种子 minADE20 `0.130107±0.000938`，minFDE20 `0.214219±0.002456` |
| F16 + 航向规范化 + 镜像/速度增强 | 284,642 | 内部五场景加权 minADE20 `0.174011`，minFDE20 `0.332367` |
| F47 多尺度低秩时间混合器 | 216,828 | 参数比 F16 少 23.8%，scene 1 单种子接近精度主线；尚缺正式多种子跨场景验证 |

这些五场景是 `univ_data_train.npy` 内部 scene ID，并非公开论文常用的 ETH/Hotel/Univ/Zara1/Zara2 官方 leave-one-dataset-out 五折。因此，本报告的实验优先级服务于内部筛选，不能把现有数字直接当成公开 SOTA 对比结论。详细证据见已有的[三部分编码实验报告](../THREE_PART_ENCODER_SERIES/FINAL_REPORT.md)、[非解码优化报告](../NON_DECODER_OPTIMIZATION_SERIES/FINAL_REPORT.md)和[注意力计算报告](../ATTENTION_COMPUTE_SERIES/FINAL_REPORT.md)。

### 2.2 当前真正暴露出的短板

已有实验共同说明了四点：

1. **当前不是“邻居看得不够多”**。扩大邻域、双尺度远邻、二阶摘要、TTC/DCA 图都没有稳定改善 Oracle 指标；ETH/UCY 的小规模局部交互中，加入更多关系更容易引入噪声。
2. **Q/K/V 的表达不能随便压缩**。GQA、MQA、低维 Q/K、Identity-Value、Graph Additive 和 Cascaded Group 均有明显精度损失。
3. **历史确实有冗余，但压缩方式非常敏感**。前后两段池化能减少注意力核心计算，早期池化+最近两帧也有过强信号；按速度相似度合并、按单行人加速度选事件则失败。
4. **跨场景尺度与坐标系仍影响学习**。航向规范化给五场景 minADE 带来 4.4% 加权改善，说明表征一致性比继续堆叠复杂关系 bias 更重要。

因此，新方法应满足至少一项：让注意力更稳定、让历史采样真正依赖交互事件、或提高同一稀疏边内部的关系表达力。只追求渐近复杂度、增加远邻、或换 softmax 稀疏函数，不再是合适主线。

## 三、检索范围与判断标准

本次优先核验 CVPR、ICCV、NeurIPS、ICLR、ICML、ACL/EMNLP 等正式论文页和论文原文，覆盖：

- 目标检测与密集视觉：Deformable Attention、BiFormer、NAT、Swin V2、EfficientViT、动态 token；
- 视频：运动对应驱动的可变形采样；
- 点云：向量注意力与几何位置调制；
- 图学习：Graphormer、GraphGPS、TokenGT、Exphormer；
- NLP：Differential Transformer、GQA、FlashAttention、NSA、HyperAttention；
- 时间序列与替代序列模块：PatchTST、iTransformer、Mamba、Hyena。

筛选时采用五个标准：

1. 是否解决 ASTRA 已观察到的真实问题；
2. 是否与 `N≤9、T=8` 的小张量条件相容；
3. 是否保持稀疏图与运动学关系；
4. 是否能做单变量消融；
5. 是否可能同时改善 minADE/minFDE，而不仅仅减少理论 FLOPs。

## 四、跨领域论文梳理与对 ASTRA 的含义

### 4.1 目标检测/视觉：内容自适应采样比固定稀疏更值得迁移

#### Deformable Attention / Deformable Video Transformer

[Vision Transformer with Deformable Attention（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Xia_Vision_Transformer_With_Deformable_Attention_CVPR_2022_paper.html)指出，密集全局注意力会引入无关区域，而固定窗口等数据无关稀疏模式又限制长程关系；其核心是在内容条件下选择少量 K/V 位置。[Deformable Video Transformer（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Deformable_Video_Transformer_CVPR_2022_paper.html)进一步把采样与跨帧运动对应联系起来，只关注少量动态预测的视频 patch。

这类思想对 ASTRA 有直接价值，但不能照搬二维图像坐标采样。行人编号没有连续空间意义，不能在“第 2.3 个行人”上插值；**可连续插值的是每个真实邻居的时间轴**。因此正确迁移形式是：稀疏图仍决定允许交互的真实邻居，网络只在每条允许边的 8 帧历史中预测 2–3 个连续时刻。

已有 F39 不能代表这条路线已经失败。F39 依据“来源行人自身最大加速度”选一个离散事件，同一个邻居面对不同目标时仍选择同一时刻；它没有使用目标—邻居相对位置、相对速度、closing、TTC/DCA 来决定采样。可变形的核心恰好是 **query-aware / pair-aware**，两者机制不同。

#### BiFormer

[BiFormer（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_BiFormer_Vision_Transformer_With_Bi-Level_Routing_Attention_CVPR_2023_paper.html)先在区域级路由，再在被选区域中执行细粒度 token attention，兼顾内容自适应和 GPU 友好的矩阵运算。

这一机制在大图像或未来 SDD 密集场景仍有价值，但 ETH/UCY 当前入口最多 9 人，预处理本身已经形成局部候选，F13 动态路由复验失败，F42 扩大/双尺度邻域也显著退化。当前再做 agent 区域路由，粗筛成本和路由错误很可能大于收益。结论是：**ETH/UCY 排除；SDD 场景分支建立后重新评估。**

#### NAT、Swin、CSWin 和局部窗口

[Neighborhood Attention Transformer（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/papers/Hassani_Neighborhood_Attention_Transformer_CVPR_2023_paper.pdf)以滑动邻域保持局部性，并依靠定制 kernel 获得真实效率；[Swin V2（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html)使用窗口体系并引入 scaled cosine attention；[CSWin（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_CSWin_Transformer_A_General_Vision_Transformer_Backbone_With_Cross-Shaped_Windows_CVPR_2022_paper.html)沿横纵条带扩大感受野。

对 ASTRA 而言，窗口/条带形态已经被“时间轴 + 社会轴因子化”和运动学 Top-K 图更自然地替代；行人并不位于规则栅格中。NAT 的 kernel 经验也与 F41 一致：只有稀疏模式配套融合 kernel 才可能真实加速，普通 gather/scatter 在小张量上反而变慢。因此窗口本身排除，但 Swin V2 的 **scaled cosine similarity** 尚未被独立验证，应该保留。

#### 动态 token pruning / merging

[A-ViT（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/papers/Yin_A-ViT_Adaptive_Tokens_for_Efficient_Vision_Transformer_CVPR_2022_paper.pdf)让 token 自适应停止计算；[Token Merging（ICLR 2023）](https://openreview.net/pdf?id=JroZRaRw7Eu)通过轻量匹配合并相似 token，而不是直接丢弃信息。

ASTRA 已试过固定两段池化、landmark、最近精确保留、速度相似度 token squeezing。F40 的失败尤其说明：运动速度相似不等于轨迹阶段等价，跨时间合并会破坏顺序。动态 pruning/merging 对当前精度主线优先级低；如果以后以部署为目标，应采用固定输出长度、保护最后两帧、并以目标—邻居相关性决定合并，而不能再次只用来源轨迹相似度。

#### Less-Attention Vision Transformer

[LaViT（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_You_Only_Need_Less_Attention_at_Each_Stage_in_Vision_CVPR_2024_paper.html)减少每个 stage 中重复计算 attention 的次数，并用先前 attention 做后续特征对齐。ASTRA 当前时间 Transformer 只有一层，核心 temporal-social attention 也只执行一次，没有足够的跨层冗余可复用。因此它是很好的未来加深模型时的约束原则，但不是当前可直接产生收益的实验。

### 4.2 点云：标量 head 权重可能不足以描述运动关系

[Point Transformer V2（NeurIPS 2022）](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d78ece6613953f46501b958b7bb4582f-Abstract-Conference.html)提出 grouped vector attention：注意力权重不再只是每个 head 一个标量，而是在通道组层面调制 value，并加强相对位置编码。

这与行人交互高度相关。当前 F16 每个 head 对某个“邻居—时刻”产生一个标量，之后同一 head 的 24 个通道共享权重；但速度、横向位移、纵向 closing、转向等信息可能需要对同一邻居采用不同响应。F18 的 relation-modulated Q/K 是给 Q/K 向量加关系位移，F43/F45 是标量 bias，二者都不是通道组权重。因此 grouped vector attention 是目前少数真正没有被等价实验覆盖的表达机制。

不能直接使用 96 维逐通道注意力，否则容易增参和过拟合。建议只使用 `G=4` 或 `G=8` 个通道组，并让权重编码器输入低维运动关系；同时保持现有稀疏 allowed mask。这样既保留点云方法的几何优势，也符合轻量化约束。

### 4.3 图 Transformer：大部分结构编码路线已经被覆盖

[Graphormer（NeurIPS 2021）](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html)通过中心性、最短路和边编码把图结构注入 attention；[GraphGPS（NeurIPS 2022）](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html)把局部真实边消息传递和全局 attention 解耦；[TokenGT（NeurIPS 2022）](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d84236751fe6d25dc06db055a3180b0-Abstract-Conference.html)把节点和边都作为 token；[Exphormer（ICML 2023）](https://proceedings.mlr.press/v202/shirzad23a.html)使用虚拟全局节点与 expander sparse edges 扩展到大图。

ASTRA 已有 node structure encoding、关系 token/关系 value、关系类型 bias、二阶邻居、edge iteration、agent proxy、稀疏真实边，并试过扩大图范围。对 `N≤9` 而言，最短路/expander/global node 的主要优势不会出现，而图偏置系列已经显示 minADE 与 minFDE 难以同步改善。

GraphGPS 仍给出一个方法论提醒：局部真实边信息与全局上下文不必由同一个注意力完成。未来加入 SDD 场景分支时，可以让 agent graph 保持局部稀疏、场景 tokens 做低频全局上下文；但不建议现在给 ETH/UCY 再增加图谱位置编码。

### 4.4 NLP：抗噪声机制比长上下文加速更相关

#### Differential Transformer

[Differential Transformer（ICLR 2025 Oral）](https://openreview.net/forum?id=BMOkEH35gE)计算两张独立 softmax attention map 的差，以抵消无关上下文中的共同噪声，并促进更稀疏的关注模式。

这与 ASTRA 的“扩大邻居无益、无关关系容易污染表示”诊断吻合。它不同于 sparsemax/entmax：后者改变单张分布的归一化，Differential Attention 则通过两张分布相减产生有符号的校正。建议采用计算受控版本：每个 head 的 Q/K 维度一分为二，分别生成两张图，使总 QK 点积维度不高于基线；value 仍保留完整 head 维度。风险在于 ETH/UCY 数据量小、负权重可能破坏运动消息的稳定性，所以优先级低于可变形采样和 cosine QK。

#### FlashAttention、HyperAttention、NSA

[FlashAttention（NeurIPS 2022）](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)的关键是 IO-aware 精确 attention kernel，而不是改变模型语义；[HyperAttention（ICLR 2024）](https://openreview.net/pdf?id=Eh0Od2BJIM)针对长上下文近线性近似；[Native Sparse Attention（ACL 2025）](https://aclanthology.org/2025.acl-long.1126/)组合压缩、选择和局部窗口，并面向 64K 上下文做硬件对齐。

当前 F16 的一条 query 最多只面对 9×8 个候选位置，且已有实测表明 attention 核心 MAC 减少 56%–67% 仍没有稳定端到端延迟改善。长上下文方法的索引、分块和调度开销不会摊薄。NSA 的“压缩 + 选择 + 局部精确”概念与 F35/F38/动态路由的组合高度相似，但其成功依赖长序列和定制 kernel。因此这些方法应排除出 ETH/UCY 精度实验，只作为未来大规模 SDD/场景 token 的工程参考。

#### GQA/MQA 与 DropKey

[GQA（EMNLP 2023）](https://aclanthology.org/2023.emnlp-main.298/)主要减少自回归大模型反复读取 K/V 的带宽开销；ASTRA 不是长序列自回归推理，F32/F33 又已直接显示 K/V 头共享会伤害交互精度。[DropKey（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Li_DropKey_for_Vision_Transformer_CVPR_2023_paper.html)在 softmax 前随机丢 Key，但 F24 已做 temporal DropKey，Masked History 也暴露出跨种子不稳定。因此二者均排除。

### 4.5 时间序列与序列替代模块：适用条件不匹配

[PatchTST（ICLR 2023）](https://openreview.net/pdf?id=Jbdc0vTOcol)通过长时间片 patch 和 channel independence 处理长时预测；[iTransformer（ICLR 2024）](https://openreview.net/pdf?id=JePfAI8fah)反转 token 组织，让变量成为 token；[Hyena（ICML 2023）](https://proceedings.mlr.press/v202/poli23a.html)与[Mamba（COLM 2024）](https://openreview.net/pdf?id=tEYskw1VY2)都面向长序列，以亚二次或线性序列混合替代注意力。

ASTRA 的观测只有 8 帧。F47 已表明低秩多尺度 mixer 可以在短历史上实现明显减参，这比引入完整 Mamba/Hyena 更符合规模。PatchTST 的 patch 价值也已部分被 F35/F38 覆盖。iTransformer 的“变量 token”有一定新意，但当前状态特征中高阶运动量很容易受 8 帧差分噪声影响，EXP-103 已验证直接扩展 jerk 等变量失败。结论是：这些方法可以作为“替换时间编码器”的远期分支，但不应排在注意力候选之前。

## 五、与既有实验的机制级排除矩阵

| 文献机制族 | ASTRA 对应实验 | 覆盖程度 | 实验结论 | 本轮判断 |
|---|---|---|---|---|
| Bi-Level / dynamic routing | F13、F23、F27、F42 | 高 | 动态路由复验失败，扩大/双尺度邻域降低 Oracle | ETH/UCY 排除；SDD 再看 |
| Agent / proxy tokens | F21 | 直接 | 精度不占优 | 排除 |
| Sparsemax / Entmax | F17、F19、F22 | 直接 | 不稳定或落后 F16 | 排除 |
| Relation-modulated QK | F18 | 直接 | 复验未进入主线 | 排除原形式 |
| Graph structural bias | F20、F25、F43、F45 | 高 | minFDE 偶有微增，minADE/同路径 FDE 退化 | 排除继续堆 bias |
| Node-edge-node / higher-order graph | F15、F44 | 高 | 增参且 Oracle 不稳 | 排除 |
| Agent-time cross attention | F16 | 直接 | 当前胜者 | 保留基线 |
| DropKey / masked history | F24、EXP-105/110 | 高 | 有局部收益但跨种子不稳 | 不继续加大 dropout |
| Landmarks / token compression | F28、F29、F35、F36、F38 | 高 | 固定两段和 recent-exact 有效率潜力；跨场景不稳定 | 保留对照，不再做全局 landmark |
| Additive / linearized attention | F30，文献复杂度分析 | 中高 | F30 明显退化；短序列无真实复杂度优势 | 排除 |
| Cascaded Group | F31 | 直接 | 参数更少但精度/速度均不理想 | 排除 |
| GQA / MQA | F32、F33 | 直接 | 精度显著退化 | 排除 |
| Identity Value | F34 | 直接 | 精度退化 | 排除 |
| Low-dimensional QK | F37 | 直接 | 精度退化 | 排除单纯降维 |
| Motion event sampling | F39 | 部分 | 单来源加速度事件失败 | 不排除 pair-conditioned deformable sampling |
| Token squeezing / merging | F40 | 部分 | 速度相似度合并破坏时间阶段 | 仅保留 query-aware 固定长度版本 |
| Edge-list exact sparse kernel | F41 | 直接 | 数学等价但实际更慢 | 小 N 排除 |
| Dual-scale / local-global | F42、F46、F47 | 中高 | 社会双尺度失败；时间低秩替换成为 Pareto 候选 | 只保留 F47 |
| Vector attention | 无直接对应 | 未覆盖 | — | 正式候选 |
| Scaled cosine QK | 无直接对应 | 未覆盖 | — | 正式候选 |
| Differential attention | 无直接对应 | 未覆盖 | — | 正式候选 |
| Pair-conditioned deformable time sampling | 无直接对应 | 未覆盖 | — | 首选候选 |
| Adaptive local token smoothing | 无直接对应 | 未覆盖 | — | 正式候选 |

## 六、建议正式实现的五个方法

### P1：Kinematics-Guided Deformable Temporal-Social Attention（KDTA）

#### 设计

对目标 `i` 与稀疏邻居 `j`，从目标表示、邻居表示和已有运动学 relation 中预测 `M=3` 个时间采样位置：

`offset_ij = MLP([agent_i, agent_j, relation_ij])`

其中一个位置固定为最后一帧，另外两个位置限制在 `[0, T-1]`，用线性插值得到 K/V。然后沿“邻居 × 采样时刻”做现有 softmax attention。稀疏 allowed graph、relation bias、relation value、K20 解码器全部不变。

建议两个单变量版本：

- `KDTA-2+last`：预测两个位置并固定最后一帧，总计 3 个 token；
- `KDTA-residual`：以 F38 的“早期中心 + 最近两帧”为初始位置，只学习有界残差，优化更稳定。

#### 为什么最有潜力

- 直接修复 F39 的根本缺陷：事件必须由目标—邻居关系决定；
- 保留最近状态和运动学先验；
- K/V 时间长度 8→3，attention 核心理论主要 MAC 可接近 F38 的 `-56.1%`；
- 内容自适应来自顶会视觉/视频证据，但迁移形式针对轨迹数据重新定义，有形成方法创新的空间。

#### 风险

8 帧很短，完全自由的 offset 容易塌缩到最后一帧。必须使用有界残差、采样分散正则或最小间隔，且不能让正则权重盖过轨迹损失。第一轮只用极小 offset MLP，控制总参数低于 290k。

### P2：Normalized Scaled-Cosine Sparse Attention（NSCA）

#### 设计

将

`q·k / sqrt(d_h)`

替换为

`exp(clamp(s_h)) × cosine(q,k)`

每个 head 只增加一个可学习 logit scale；现有 relation bias 在 cosine score 后相加。建议初始温度使初始 logit 方差接近当前点积版本，并对 scale 做上界约束。

#### 价值

- 不减少任何 Q/K/V 表达；
- 不改变 sparse mask 和运动学边；
- 参数只增加 4 个左右，MAC 基本不变；
- 针对跨场景速度尺度、轨迹幅值和 feature norm 漂移，和航向规范化的成功证据一致。

这是最适合作为第一批低风险实验的机制。如果连它都显著退化，说明现有点积幅值本身携带了有用的运动置信度，后续可改为“norm confidence 独立门控 + cosine similarity”，而不是直接放弃。

### P3：Relation-Grouped Vector Attention（RGVA）

#### 设计

保留现有多头 Q/K 相似度作为基础，再从低维 relation 和 `q-k` 生成 `G=4` 个通道组校正量。每个“邻居—时刻”最终拥有 4 个组权重，分别调制 value 的 4 个通道组。不要做 96 维逐通道权重。

可采用残差初始化：向量校正分支初始为零，使模型起点严格接近 F16，再逐渐学习横向、纵向、速度/转向等组的差异。

#### 价值与风险

它可能改善“同一个邻居对不同运动因素作用不同”的表达，是当前关系 bias/QK shift 没有覆盖的方向。代价是 score 元素和 softmax 数量增加，且小数据可能过拟合；因此总组数不超过 4，关系编码器应共享，参数预算建议 `<5k` 增量。

### P4：Compute-Neutral Differential Sparse Attention（CDSA）

#### 设计

每个 head 的 Q/K 通道一分为二：

`A = softmax(Q1 K1^T) - λ_h softmax(Q2 K2^T)`

再用 `A` 聚合完整 V。两次点积各使用半维，使 QK MAC 总量接近基线；score 张量数量会翻倍，但当前候选位置很少。`λ_h` 采用论文推荐的稳定参数化与接近恒等的初始化。

#### 价值与风险

它瞄准无关邻居/时刻的共同噪声，机制上不同于 entmax 和 risk bias。但负权重会改变当前“凸组合消息”的物理直觉，可能让运动学 value 产生过度校正。建议把它放在 P1/P2/P3 之后，并同时记录 attention 熵、负质量比例和梯度范数。

### P5：Motion-Adaptive Temporal Key Smoothing（MATKS）

#### 设计

受 ICCV 2023 token attention 稳健性工作启发，在计算 K/V 前对每帧做局部三点平滑：

`x'_t = x_t + α_t^- x_{t-1} + α_t^+ x_{t+1}`

`α` 由该行人的速度变化、观测置信度或低维 temporal feature 预测，并归一化。最后一帧保留强残差，防止最新状态被抹平。它不删 token、不改 sparse graph。

#### 价值与风险

高阶运动状态实验失败和 Masked History 不稳定都说明 8 帧差分含噪。局部自适应平滑比新增 jerk 特征更保守，也不同于 F35 的四帧平均和 F47 的整体替代 mixer。风险是快速转弯被平滑；因此 α 应受加速度/转向门控，在事件剧烈时减弱平滑。

## 七、实验优先级与停止规则

### 7.1 第一阶段：严格单变量筛选

统一固定：

- F16 其余结构；
- 当前 Kinematic K20 和 Coverage 0.20；
- EMA 0.995；
- 航向规范化主线与稳定增强口径分别作为明确记录的底座，不能混用结果；
- 30 epochs、同一 checkpoint 选择规则；
- 首筛 `seed=1, scene=1/3`。

推荐顺序：

1. `P2 NSCA`：实现最小、风险最低；
2. `P5 MATKS`：检查稳健性是否能改善跨场景；
3. `P1 KDTA-residual`：最高潜力；
4. `P1 KDTA-2+last`：验证自由采样与残差采样差异；
5. `P3 RGVA-G4`；
6. `P4 CDSA`。

### 7.2 晋级标准

不能因为单个 seed 的某个指标变好就晋级。建议使用：

- scene 1 与 scene 3 至少一个 minADE 明显改善，另一个不得退化超过 1%；
- `minADE20 + 0.5 × minFDE20` 优于同批重跑基线；
- same-path FDE 不得出现明显反向；
- 参数量原则上不超过 300k；
- 若声称轻量化，必须同时报告理论 MAC、峰值显存和交替 CUDA latency，不能只报 FLOPs。

通过首筛后再做三种子 scene 1；三种子均值有效且方差可接受，再跑五个内部场景。最终论文结论必须补官方 ETH/Hotel/Univ/Zara1/Zara2 leave-one-dataset-out。

### 7.3 组合实验

只有单模块通过后才组合：

- 首选 `KDTA + NSCA`：一个决定看哪些时刻，一个稳定相似度；机制正交；
- 次选 `KDTA + RGVA`：一个做时间位置选择，一个做通道组关系表达；但计算和过拟合风险更高；
- 不建议首先组合 `RGVA + CDSA`，因为二者同时改变权重语义，失败后难以归因；
- 不建议将 F35/F38 压缩和 KDTA 同时开启，它们都操作时间 token，属于重复变量。

## 八、预期收益与现实边界

从既有数据看，注意力结构单独带来的稳定收益通常小于数据坐标规范化：F16 的确是早期显著赢家，但后续大量 attention 变体没有超过它；航向规范化却带来了 4.4% 的五场景加权 minADE 改善。因此，对新机制的合理目标应是：

- P2/P5：争取 1%–3% 的稳定 minADE 改善或明显降低种子方差；
- P1：若交互关键时刻假设成立，争取 2%–5% 的 minADE/minFDE 联合改善，同时降低 attention 核心规模；
- P3/P4：属于高风险表达力实验，成功时可能改善密集交互子集，但全局平均未必同步。

这些百分比是基于现有实验趋势给出的研究目标，不是文献可以保证的效果。

最关键的边界仍然是：编码器最后通过有限上下文调节固定的 K20 运动学候选基底。如果五个新方向都只能改变 Top-1 或局部 FDE，而 minADE 长期停滞，那么应接受证据：瓶颈已经转移到候选几何覆盖，而不是继续无休止替换 attention。

## 九、面向 SDD 的保留方向

SDD 的行人数、空间范围和场景语义都更大，届时一些在 ETH/UCY 被排除的机制会重新获得成立条件：

1. BiFormer/NSA 式粗到细路由：先选局部人群或场景区域，再精细交互；
2. GraphGPS 式局部 agent graph + 全局 scene tokens；
3. 真正的 edge-list/block-sparse kernel：只有大 N 才可能摊薄 gather/scatter；
4. 场景分支中的多尺度线性注意力：高分辨率 scene tokens 才适合 EfficientViT 一类机制；
5. Agent proxy/global tokens：可用于人群—场景中介，而不是在 9 个行人之间再压缩。

因此当前实验代码最好把“时间采样策略”“score 函数”“value 权重形式”和“稀疏邻居选择”做成独立开关，避免以后迁移 SDD 时重新拆模型。

## 十、最终决策

下一系列实验不应再从论文列表中任选模块替换。建议正式建立一个新的 `PAIR_AWARE_ATTENTION_SERIES`，按以下主线执行：

1. 先做 NSCA，验证跨场景尺度稳定性；
2. 再做 KDTA 两个版本，验证 pair-conditioned 关键时刻假设；
3. 再做 RGVA，验证通道组关系表达；
4. MATKS 作为稳健性支线；
5. CDSA 作为最后的高风险抗噪声支线；
6. 只有通过单变量三种子复验的模块才进入组合。

如果只选择一个最有论文方法价值、又与既有实验不重复的方向，应选 **KDTA**；如果只选择一个最便宜、最适合先验证的方向，应选 **NSCA**。

