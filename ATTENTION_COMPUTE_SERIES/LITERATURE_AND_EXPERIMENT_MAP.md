# 注意力计算机制轻量化：文献与实验映射

## 目标边界

本轮只研究能直接减少注意力矩阵、K/V token 数、QKV 投影量或成对点积的机制。固定 F10 运动编码、运动学稀疏图、K20 解码器、损失函数和训练协议。注意力偏置、稀疏概率归一化和单纯的掩码变化不属于本轮主变量。

## 用户笔记与正式文献核验

| 方法 | 正式来源 | 原始计算思想 | 对当前任务的判断 |
|---|---|---|---|
| Bi-Level Routing Attention | BiFormer, CVPR 2023 | 路由后仅在选中区域做精细注意力 | 已由 F13/F23 验证；当前实现仍先算路由矩阵，且多种子退化，不重复 |
| Agent Attention | ECCV 2024 / arXiv | 少量代理 token 中转全局信息 | 已由 F21 验证并退化；小 N 下代理瓶颈过强，不重复 |
| Focused Linear Attention | FLatten, ICCV 2023 | 线性注意力 + focused mapping + rank restoration | 当 token 数小于通道维时未必省算；当前 N≤9、T=8，暂缓 |
| Efficient Additive Attention | SwiftFormer, ICCV 2023 | 以线性逐元素运算替代二次 QK/KV 矩阵乘 | 适配成运动学图约束的 source-salience additive attention，进入 F30 |
| Cascaded Group Attention | EfficientViT, CVPR 2023 | 通道分组 QKV，head 间级联，减少 QKV 投影参数/FLOPs | 适配时间—社交 cross attention，进入 F31 |
| Linformer / token compression | arXiv 2020；STViT, CVPR 2023 | 将 K/V token 投影或压缩成较少 landmark | 8 个历史 K/V 压到 4/2，进入 F28/F29 |
| GQA / MQA | EMNLP 2023；Shazeer 2019 | 多个 query heads 共享较少 K/V heads | 4 个 Q heads 配 2/1 个 KV heads，进入 F32/F33 |
| Simplified Transformer | ICLR 2024 | 移除或固定 value/projection 等冗余组件 | 直接使用 temporal feature 作为 V，进入 F34 |
| FlashAttention | NeurIPS 2022 | IO-aware tiling，不显式落盘完整注意力图 | 保持精确注意力但不降 FLOPs；当前短序列和任意图 mask 很可能无法获益，先做后续基准而非首轮结构 |
| CAS-ViT | IEEE TIP 2026（论文预印本 2024） | 卷积加性 token mixer，移除矩阵乘和 softmax | 图轨迹没有规则二维卷积邻域；先用更直接的 SwiftFormer 式加性适配 |

## 首批实验

| 实验 | 变体 | 轨迹任务适配 | 预期节省 |
|---|---|---|---|
| EXP-043 | F28 temporal landmark-4 | 共享可学习时间投影，把每个邻居 8 个历史 K/V 压为 4 个 | temporal-social K/V 投影和分数规模约减半 |
| EXP-044 | F29 temporal landmark-2 | 同上，压为 2 个 | 对应部分约减少 75%，更激进 |
| EXP-045 | F30 graph additive | 每个源行人只计算一次时序显著性和 source score；目标差异由稀疏图与关系偏置提供 | 去除成对 QK 点积；不建立 N×N×T dot-product scores |
| EXP-046 | F31 cascaded group | 每个 head 只投影自己的通道组，上一 head 输出注入下一 head | QKV 投影参数和 MAC 理论降至约 1/H |
| EXP-047 | F32 GQA-2 | 4 个 Q heads 共享 2 组 K/V | K/V 投影与未展开缓存减半 |
| EXP-048 | F33 MQA-1 | 4 个 Q heads 共享 1 组 K/V | K/V 投影与未展开缓存减少 75% |
| EXP-049 | F34 identity value | 不学习 V 投影，直接拆分 temporal feature 为 values | 删除一次 K/V token 上的线性投影 |
| EXP-052/053/054 | F35 temporal half-pool | 由 F29 三种子权重反推，显式平均前四帧/后四帧 | score/KV时间轴减75%，核心操作约减67%，三种子通过但跨场景有依赖 |
| EXP-055 | F36 fixed half projection | F35 的固定矩阵等价实现 | 输出严格相同；端到端延迟无稳定显著变化 |
| EXP-056 | F37 low-dimensional Q/K | EfficientViT 式小 Q/K、全维 V，保持 heads 并行 | 参数减4.4%但精度退化，淘汰 |

## 预筛原则

1. 单种子 30 epochs 先看精度和稳定训练。
2. 不能只看理论大 O；同步记录总参数、训练耗时，并增加统一推理延迟/显存基准。
3. 只有相对 F16 精度损失可控且轻量化真实可测的候选进入三种子。
4. 小 N 任务中，低秩/线性方法的常数项可能大于收益，因此实测优先于论文中的长序列结论。

## 主要正式来源

- SwiftFormer, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Shaker_SwiftFormer_Efficient_Additive_Attention_for_Transformer-based_Real-time_Mobile_Vision_Applications_ICCV_2023_paper.html
- EfficientViT (Cascaded Group Attention), CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html
- FLatten Transformer, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Han_FLatten_Transformer_Vision_Transformer_using_Focused_Linear_Attention_ICCV_2023_paper.html
- GQA, EMNLP 2023: https://aclanthology.org/2023.emnlp-main.298/
- Linformer: https://arxiv.org/abs/2006.04768
- Simplifying Transformer Blocks, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/24fd58f52ff8d0496add8da3991644e9-Abstract-Conference.html
- FlashAttention, NeurIPS 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html
- STViT token sparsification, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Chang_Making_Vision_Transformers_Efficient_From_a_Token_Sparsification_View_CVPR_2023_paper.html
- Do Efficient Transformers Really Save Computation?, ICML 2024: https://proceedings.mlr.press/v235/yang24a.html
