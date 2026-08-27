# 注意力计算机制轻量化实验总报告

## 1. 结论先行

本轮直接修改注意力计算形态，而不是继续改变 bias、mask 或 decoder。共实现并正式训练了时间 landmark、图加性注意力、Cascaded Group Attention、GQA、MQA、Identity-Value、无参数双段池化和低维 Q/K 等机制。

最重要的实验结论是：

1. **当前轨迹任务不能轻易压缩 Q/K heads、V 投影或取消 pairwise matching。** 这些方向虽减少 3%–7% 参数，但精度全部明显退化。
2. **唯一可行的压缩轴是邻居历史时间 token。** 将 8 个历史状态压成“早期趋势 / 近期状态”两个摘要，能够保留甚至改善 scene 1 精度。
3. **F35/F36 将 temporal-social attention score 和 K/V 时间轴减少 75%，核心主要操作约减少 67%，峰值前向显存减少 6.6%，参数不增加。**
4. **完整模型延迟没有稳定改善。** 交替 CUDA 基准中各版本差异落在约几个百分点的运行波动内，说明当前端到端瓶颈还包括时间编码、关系编码、图构建和 K20 解码器。
5. **F35 有明显场景依赖。** scene 1 三种子优于 F16，但全场景加权后仍略弱于 F16。因此 F16 保持跨场景默认主干，F35/F36 作为场景适配和低显存候选。

## 2. 文献筛选如何转化成实验

用户笔记中的 BiFormer、Agent Attention、Focused Linear Attention、EfficientViT、SwiftFormer、CAS-ViT 等方法经过正式论文核验后，没有直接照搬：

- BiFormer 与 Agent proxy 已在上一轮 F13/F21 中失败，不重复；
- Focused/Performer 式线性注意力在 `N≤9,T=8,d=96` 下常数项和 `d²` 项未必更省，未列入首批；
- SwiftFormer 的 additive attention 被改成运动学图约束版本；
- EfficientViT 被拆成 Cascaded Group 和低维 Q/K 两个可归因实验；
- GQA/MQA 从 NLP 迁移到社会 cross-attention；
- Linformer/token compression 被改成邻居历史 landmark；
- ICLR 2024 Simplifying Transformer Blocks 被改成 Identity-Value。

主要来源包括 [SwiftFormer（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Shaker_SwiftFormer_Efficient_Additive_Attention_for_Transformer-based_Real-time_Mobile_Vision_Applications_ICCV_2023_paper.html)、[EfficientViT（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html)、[FLatten（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Han_FLatten_Transformer_Vision_Transformer_using_Focused_Linear_Attention_ICCV_2023_paper.html)、[GQA（EMNLP 2023）](https://aclanthology.org/2023.emnlp-main.298/)、[Simplifying Transformer Blocks（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/24fd58f52ff8d0496add8da3991644e9-Abstract-Conference.html) 和 [FlashAttention（NeurIPS 2022）](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)。

## 3. 首轮结果

均为 scene 1、seed=1、30 epochs、固定 K20。

| 变体 | 参数量 | Original ADE/FDE | Fused ADE/FDE | 判断 |
|---|---:|---:|---:|---|
| F16 完整时间—社交 cross | 284,642 | 0.4984 / 0.8786 | 0.4289 / 0.7834 | 精度基线 |
| F28 Landmark-4 | 284,674 | 0.5841 / 1.0281 | 0.5148 / 0.9140 | 淘汰 |
| F29 Landmark-2 | 284,658 | 0.5034 / **0.8721** | 0.4403 / 0.7763 | 进入复验 |
| F30 Graph Additive | 266,503 | 0.5895 / 1.0060 | 0.5050 / 0.8586 | 淘汰 |
| F31 Cascaded Group | **263,906** | 0.5615 / 0.9755 | 0.4757 / 0.8361 | 参数小但更慢，淘汰 |
| F32 GQA-2 | 275,330 | 0.5725 / 0.9554 | 0.5252 / 0.8806 | 淘汰 |
| F33 MQA-1 | 270,674 | 0.5600 / 0.9384 | 0.5120 / 0.8661 | 淘汰 |
| F34 Identity-Value | 275,330 | 0.5531 / 0.9393 | 0.4898 / 0.8481 | 淘汰 |
| F35 双段时间池化 | 284,642 | **0.4786 / 0.8729** | **0.4215 / 0.7732** | 三种子与跨场景 |
| F37 低维 Q/K | 272,226 | 0.5608 / 0.9413 | 0.5089 / 0.8736 | 淘汰 |

## 4. F29 与 F35 三种子

| 方法 | Original ADE/FDE | Fused ADE/FDE | minADE20/minFDE20 |
|---|---:|---:|---:|
| F4 | 0.5400 / 0.9342 | 0.4593 / 0.8109 | 0.1658 / 0.2553 |
| F16 | 0.4948 / 0.8770 | **0.4338 / 0.7833** | **0.1641 / 0.2616** |
| F29 learned landmark-2 | 0.5122 / 0.8730 | 0.4587 / 0.7950 | 0.1687 / **0.2525** |
| F35 fixed half-pool | **0.4887 / 0.8653** | 0.4429 / 0.7910 | 0.1685 / 0.2650 |

F35 相对 F16 的 Original ADE/FDE 分别改善约 1.2%/1.3%。旧 verifier 融合后略弱，oracle 也略差，说明 F35 主要改善内部表示与模式选择稳定性，而不是候选覆盖。最终模型应针对 F35/F36 重新训练 verifier。

## 5. 为什么双段池化有效

F29 三个种子学到的两个 landmark 权重高度一致：第一个近似平均前四帧，第二个近似平均后四帧，几乎没有形成复杂动态投影。这说明轨迹历史的有效低秩结构不是任意 learned tokens，而是：

- 前半段：长期运动方向和稳定趋势；
- 后半段：最近速度、转向和即时交互状态。

F35 将这一规律显式化，用无参数均值获得更稳定的优化。F36 再把相同均值写成固定 2×8 矩阵；同一 checkpoint 下轨迹和 logits 最大差均为 0，数学严格等价。

## 6. 复杂度与显存

在 `N=9,T=8,d=96,H=4` 下：

| 指标 | F16 | F35/F36 | 变化 |
|---|---:|---:|---:|
| temporal-social score 元素 | 2,592 | 648 | **-75%** |
| K/V 历史 token | 8/邻居 | 2/邻居 | **-75%** |
| temporal-social 核心主要操作 | 约 1.62M | 约 0.53M | **约 -67%** |
| 总参数 | 284,642 | 284,642 | 0 |
| batch=64 峰值前向显存 | 58.44 MiB | 54.58 MiB | **-6.6%** |

完整公式见 `F16_F35_COMPLEXITY_ANALYSIS.md`。

## 7. 实际延迟

最终使用 18 轮交替、每轮旋转模型顺序的 CUDA 测试：

| 方法 | batch=1 mean | batch=64 mean |
|---|---:|---:|
| F16 | 10.56 ms | 13.20 ms |
| F29 | 10.56 ms | 13.40 ms |
| F35 | 10.55 ms | 12.94 ms |
| F36 | 10.56 ms | 13.84 ms |

差异均落在运行波动范围，不能宣称稳定端到端加速。attention 核心已经变小，但它不是当前完整模型唯一或绝对的延迟瓶颈。下一轮若以真实速度为目标，应优化关系特征的 `N×N×T` 构建、动态图/结构矩阵和 decoder，而不是继续压缩 attention heads。

## 8. 跨场景结果

样本加权 Original ADE/FDE：

| 方法 | ADE/FDE | 加权 minADE20 |
|---|---:|---:|
| F4 | 0.4864 / 0.9944 | **0.1984** |
| F16 | **0.4737 / 0.9802** | 0.1996 |
| F35 | 0.4838 / 0.9830 | 0.2019 |

F35 在 scene 1 最好，在 scene 4 的 FDE 更好，但在 scene 0/2/3 不占优。存在 verifier 的场景加权 Fused FDE 为 F4 0.9137、F16 0.9097、F35 **0.8950**。因此 F35 更偏向长时终点与特定场景，而 F16 仍是跨场景 ADE 默认主干。

## 9. 最终推荐

### 默认跨场景模型

继续使用 **F16 + K20**。它的跨场景加权 ADE 最好，泛化证据最完整。

### 低显存 / scene 1 / 长时 FDE 优先

使用 **F35 或数学等价的 F36 + K20**：不增加参数，attention 核心计算规模显著下降，峰值显存降低 6.6%，scene 1 三种子 Original 指标优于 F16，并在 scene 4 的 FDE/Fused FDE 上有优势。

### 不建议继续投入

Graph Additive、Cascaded Group、GQA、MQA、Identity-Value、低维 Q/K，以及在短序列上直接使用 Focused/Performer 式线性注意力。当前证据表明它们压缩了交互表达中不可替代的部分。

### 下一步真正值得做

1. 为 F35/F36 重训 verifier，验证 Fused ADE 能否追上其 Original 改善。
2. 将 scene 0 的优化转向单体运动编码和运动学外推，不再依赖社会 attention。
3. 对关系特征和动态图构建做 profiler；当前 attention 缩小 67% 但延迟不变，说明下一瓶颈在 attention 外部。
4. 若部署需要固定模型，可根据场景密度/域选择 F16 或 F35，但门控应在训练外做模型级策略，避免此前 F27 场景人数硬切换的表征不连续。
