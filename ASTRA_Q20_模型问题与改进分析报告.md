# ASTRA/Q20 模型问题与后续改进分析报告

> 日期：2026-08-16  
> 分析范围：用户在 Obsidian 中的行人轨迹预测论文笔记、当前 ASTRA/Q20 代码、已有实验报告及 Q20 逐样本输出。  
> 证据标记：**已验证**表示已有本地实验或代码直接支持；**文献支持**表示论文提出并验证过相近机制；**待验证**表示本文据此形成的下一步假设。

## 一、结论先行

当前模型并不是“20 模态设计没有效果”，而是存在三个层次不同的问题：

1. **最高优先级是实验协议问题。** 训练代码把 `mode='testing'` 的目标场景直接作为每轮验证集，并用它保存最佳模型；多模态验证又只监控 minADE/minFDE，而最终不满意的是 Top-1 ADE/FDE。换言之，当前开发过程既反复接触测试场景，又没有用最终目标选择模型。修复这一点之前，继续比较小幅结构增益很容易得到过度乐观或不可复现的结论。
2. **Q20 的首要模型瓶颈是候选排序，而非候选覆盖。** 三随机种子下，Q20 的 `minADE20=0.2613±0.0137`，但 Top-1 ADE 为 `0.6505±0.0178`，平均 selection regret 为 `0.3892±0.0041`。这说明好轨迹经常已经生成，却被评分器压在低分支。
3. **候选生成仍有第二层瓶颈。** `minFDE20≈0.454` 并非足够低；当前 20 个自由查询没有数据驱动的运动原型、明确的“终点—路径形状”层次，也缺少对所有候选的中间关键点约束。因此不能只修评分器后就停止。

建议的主线不是再叠加 RWPE、motif、LapPE 或更大的 reranker，而是：

> **保留已验证有效的状态编码、风险感知稀疏图和 E24 轨迹头；先建立无测试泄漏的验证协议，再训练一个与 Q20 分支标签解耦的轻量 stimulus verifier；随后把自由 Mode Query 改成数据驱动的“终点 × 路径形状”分层原型，并用少量中间关键点约束完整轨迹。**

如果目标是论文中的标准多模态集合预测，应把 Q20 视为已经有价值的候选生成器，主要报告 minADE20/minFDE20；如果目标是部署时只输出一条轨迹，则 Top-1 评分器就是必须单独解决的核心任务，二者不能继续混为一个指标。

## 二、当前模型的准确重建

当前主模型位于 [ASTRA_ACCURACY_Q20](C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20)，实际运行组合为：

- 8 个历史点、12 个未来点，2.5 Hz；
- 不使用图像/场景 U-Net，不使用 VAE；
- 位置与时间编码，加速度、速度、航向构成的六维自状态编码；
- 基于相对位置、相对速度、TTC、DCA 的自适应有向 Top-K 图；
- RBF 运动学边偏置和两层时空 Transformer；
- 20 个固定可学习 Mode Query，经模式自注意力和历史交叉注意力并行解码；
- 每个模式预测终点和 22 维 endpoint-zero PCA 残差，再进行保持终点不变的残差修正；
- 评分头读取 query、PCA/终点参数及其跨模式中心化值，输出 20 个 logits。

评分发生在轨迹重建和残差修正之后，但输入并不直接包含修正后的完整路径，也没有显式的社会/场景可行性似然，见 [trajectory_decoder.py](C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/models/trajectory_decoder.py:627)。

训练时先用

\[
C_k=\operatorname{ADE}_k+0.25\operatorname{FDE}_k
\]

选出 WTA 分支，只对赢家进行轨迹、终点和短期误差回归；评分目标由 soft target 与 hard CE 组合。soft target 先把每个样本的候选代价差除以该样本的平均 gap，见 [losses.py](C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/utils/losses.py:150)。这使评分目标主要表达“相对顺序”，却弱化了绝对质量差：一个所有候选都很差的样本和一个候选彼此都很接近的样本，经过归一化后可能产生相似的监督强度。

## 三、哪些设计已经得到支持

| 模块 | 现有证据 | 判断 |
|---|---|---|
| 自状态编码 | 三种子 ADE `0.6397→0.5962`，FDE `1.3080→1.2300`，所有主要风险分组 3/3 改善 | **保留，是当前最可靠的精度贡献** |
| 自适应有向 Top-K 图 | 相对全连接图，三种子 ADE `-0.0016`、FDE `-0.0048` | **保留为效率和归纳偏置，但不作为主要精度创新** |
| E24 endpoint + PCA residual | 单种子相对直接头 ADE `-0.0079`、FDE `-0.0098` | **保留；需补三种子验证** |
| Q20 并行生成 | 三种子 minADE20 `0.2613±0.0137`、minFDE20 `0.4538±0.0290` | **候选覆盖明显有效** |
| motif/RWPE/替代结构编码 | 多次实验没有稳定 ADE 收益 | **暂停** |
| DCT/恒速残差、P6/P24 | 总体或长期 FDE 退化 | **不回到这些低维强约束路线** |
| 大型 trajectory-set reranker | 增加约 117k 参数后 Top-1 反而恶化 | **当前方案否定，但不能据此否定所有验证器** |

上述本地证据分别见 [状态多种子报告](C:/Users/linke/Desktop/DRSE/ASTRA_MOTIF_RBF/results/state_multiseed_summary/report.md)、[稀疏图多种子报告](C:/Users/linke/Desktop/DRSE/ASTRA_MOTIF_RBF/results/full_vs_sparse_state_rbf_multiseed/report.md)、[E24 报告](C:/Users/linke/Desktop/DRSE/ASTRA_ENDPOINT_PCA_DECODER/results/endpoint_pca_seed1_summary/report.md) 和 [Q20 多种子报告](C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/results/q20_multiseed_validation/report.md)。

一个需要澄清的文献结论是：P6/P24 的负结果**不能等价为“EigenTrajectory 已被否定”**。EigenTrajectory 同时在输入和输出 ET 空间建模、对齐并聚类系数锚点、预测锚点残差，并同时施加系数、物理路径和终点损失；当前实验只验证了部分输出参数化。[EigenTrajectory 原文](https://openaccess.thecvf.com/content/ICCV2023/html/Bae_EigenTrajectory_Low-Rank_Descriptors_for_Multi-Modal_Trajectory_Forecasting_ICCV_2023_paper.html)支持“数据驱动锚点 + 残差”的思想，但当前结果明确反对重新采用过低秩的完整路径瓶颈。

## 四、最关键的故障诊断

### 4.1 测试集被当成验证集，且模型选择目标与最终目标错位

代码直接构造 `training` 和 `testing` 两个数据集，然后把 `testing` 传入训练函数，见 [main.py](C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/main.py:319)。训练过程中每 5 轮在这个 loader 上验证并据此保存最佳模型，见 [train_ETH.py](C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/train_ETH.py:245)。

对多模态模型，验证函数计算的是每个样本 20 条轨迹中的 minADE 和 minFDE，而不是评分器选出的 Top-1，见 [train_ETH.py](C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/train_ETH.py:702)。因此存在三种目标：

- 训练 WTA 赢家：`ADE + 0.25 FDE`；
- checkpoint 选择：minADE 和 minFDE 同时改善；
- 最终不满意的指标：评分 logits 的 Top-1 ADE/FDE。

基础 Q20 seed 1 的最佳 minADE/minFDE checkpoint 恰好出现在最后一轮，所以“评估 last checkpoint”没有改变这次主数字；但评估协议仍不成立，尤其前期的模块筛选、阈值选择和多次观察已经使用同一目标测试场景。

**修复要求：**

1. 从训练场景内部建立真正的 validation；测试场景只在方案冻结后运行一次。
2. 因滑动窗口高度重叠，validation 必须按场景或连续时间块切分，不能随机打散窗口。
3. 同时记录四组指标：单轨迹 Top-1 ADE/FDE、minADE20、与 best-ADE 同一路径的 FDE、minFDE20；checkpoint 按预先声明的主目标选择。
4. 若论文主张多模态集合能力，以 minADE20 为主；若主张单轨迹部署，以 Top-1 ADE 为主，不能事后切换。

### 4.2 排序器几乎退化成固定模式先验

对 seed 1 的 24,334 个逐样本结果重新分析得到：

| 指标 | 训练特征集 | 测试集 |
|---|---:|---:|
| Top-1 命中 best-ADE | 31.98% | 15.66% |
| Top-1 ADE | 0.4291 | 0.6418 |
| minADE20 | 0.1741 | 0.2545 |
| 平均最大模式概率 | 0.1443 | 0.1333 |

测试集平均概率熵为 2.7365，而 20 类均匀分布的最大熵为 `ln(20)=2.9957`，说明整体置信度很低。更重要的是，20 个分支都曾成为 best-ADE 分支，但 Top-1 只大量选择少数分支；分支 0、1、2、7、9、12、14、15、16、18 在测试中各自成为数百到上千次 oracle 赢家，却从未被选为 Top-1。

作为一个**仅用于诊断、不能作为有效测试基线**的检查：如果事后在测试集上固定选择平均 ADE 最小的第 4 号分支，其 ADE 为 0.6406，甚至略低于 learned Top-1 的 0.6418。这说明当前评分器的 argmax 几乎没有超过一个全局分支先验。

按最大概率五等分后，只有最高置信度 20% 的样本具有明显选择信息：

| 置信度组 | 平均 max-p | best-ADE 命中率 | selection regret | Q20 Top-1 ADE | E24 ADE |
|---|---:|---:|---:|---:|---:|
| Q1 | 0.0801 | 6.72% | 0.4215 | 0.6955 | 0.6677 |
| Q2 | 0.0874 | 5.47% | 0.5239 | 0.8491 | 0.7643 |
| Q3 | 0.0985 | 7.64% | 0.4785 | 0.7697 | 0.6896 |
| Q4 | 0.1502 | 10.70% | 0.3803 | 0.6323 | 0.5367 |
| Q5 | 0.2504 | 47.75% | 0.1326 | 0.2625 | 0.2644 |

这说明评分器不是完全没有信号，而是只在一小部分容易样本上可靠。已有 robust gate 跨三种子平均只切换 5.28%，ADE 改善 `0.0008`、FDE 基本不变，也与这个判断一致。

### 4.3 为什么现有评分损失容易产生这种现象

主要原因不是评分头参数太少，而是监督形式和数据结构共同导致：

1. **单一真实未来造成标签噪声。** 对一个本来就有多种合理未来的历史，仅有一条观测 GT；把“离这条 GT 最近的分支”当作真实行为类别，会把合理但未发生的模式都视为负类。
2. **WTA 分支不平衡。** dominant motion 获得更多赢家监督，评分头逐渐学成分支先验；未被频繁选择的分支即使在新场景中产生最佳轨迹，也难以获得高分。
3. **样本内 gap 归一化丢失绝对难度。** 每个样本都除以自身平均 gap，使排序器不能明确区分“赢家真正明显更好”与“20 条都差不多”。
4. **优化和评价的 utility 不统一。** 训练用 `ADE+0.25FDE`，最终又分别检查 Top-1 ADE、FDE 和 best-ADE hit。
5. **验证阶段不看 Top-1。** 即使评分器已开始过拟合，训练流程也不会据此早停。
6. **评分器没有独立学习真实轨迹分布。** 它只学习当前生成器产生的候选及其分支身份，容易记住生成器的偏差；冻结后再训练更大的 reranker 仍会受到同一问题影响。

Q6 的小型质量 reranker 曾把 ADE 从 0.7620 改善到 0.7267，说明排序损失有帮助，但模式仍高度集中；Q20 的大型集合 reranker 反而退化。因此下一步不应简单扩大 MLP/Transformer，而应改变评分器的训练数据和概率建模方式。

### 4.4 候选生成器的结构仍不够可识别

20 个查询都从随机可学习向量开始，加上相同的 pooled history，再通过共享 parameter head 解码。模式自注意力能产生差异，但每个分支没有稳定可解释的几何职责。EMP 也使用并行 predictor 和 WTA，但其核心是每个 predictor 的独立输出映射与后验稀疏选择；当前实现使用共享 parameter head，分支差异更依赖 query 表示，分支身份更容易漂移。[EMP 原文](https://www.sciencedirect.com/science/article/pii/S1077314224001887)确实验证了并行 predictor 和 model-error selector，但其成功不能自动迁移到当前共享头设计。

另一方面，现有多样性损失要求所有样本的候选终点彼此至少保持固定距离，没有根据样本的不确定性调整。它有助于覆盖，却可能在本来近似单峰的样本上制造不必要的远离候选，随后把困难全部交给评分器。

### 4.5 交互编码不是当前精度瓶颈

状态编码带来显著增益，而全连接图换成风险感知稀疏图只有约 0.0016 ADE 改善；motif 和其他结构位置编码又没有稳定收益。因此当前数据已经表明：

- “有没有运动状态”很重要；
- “怎样再复杂地编码图结构”暂时不是主要误差来源；
- 稀疏图的价值更多是轻量化、可解释性和在高密度数据上的可扩展性。

继续增加高阶图结构，很可能只是在一个已非瓶颈的模块上花参数和实验预算。

## 五、论文启示与当前模型的对应关系

| 论文/方法 | 可迁移的核心 | 对当前模型的真实启示 | 优先级 |
|---|---|---|---:|
| EMP | 并行 predictors、WTA 后验选择、mode-to-mode attention | 证明并行多分支路线合理；可测试“少量独立分支头”而非完全共享参数头 | 中 |
| STP | 从训练集压缩共享 sparse motion modes，再注入场景条件 | 用数据驱动原型初始化/约束 mode query，减少自由分支漂移 | 高 |
| EigenTrajectory | 对齐后的轨迹/系数锚点 + 残差 | 借鉴锚点与残差，不恢复低秩 P6 瓶颈 | 高 |
| G2LTraj | 全局关键步 + 局部生成；多粒度选择 | 只引入 2–3 个中间关键点作为路径骨架，避免完整自回归 | 高 |
| SICNet | 终点和完整实例之间的 sparse instance | 支持“终点之外还需要少量中间点”的判断 | 高 |
| Stimulus Verification | 用真实轨迹训练条件似然 verifier，再对生成候选评分 | 避免把 Q20 分支身份当类别；是当前评分器最直接的替代方案 | **最高** |
| EdgeVTP | 半径 + hard Top-K、一次性低维曲线解码 | 支持稀疏图与非自回归；但当前 DCT/E6 结果反对更强曲线瓶颈 | 低 |
| Forecast-MAE/CASPFormer | 场景预训练、多尺度或可变形场景采样 | 仅在 SDD 第二阶段给 endpoint/verifier 做晚融合，不应先重建大场景分支 | 中（SDD） |
| Stochastic NAR Transformer | 随机并行 queries、稀疏 agent-time 交互 | 当前模型已覆盖大部分思想；再加随机噪声不是首要问题 | 低 |

STP 的“共享稀疏运动模式”由训练集轨迹压缩而来，再变成样本条件查询，正好对应当前自由 query 缺少几何身份的问题。[STP 官方摘要](https://pubmed.ncbi.nlm.nih.gov/41171662/)同时强调 irregular interaction 和 early-sparsity，但你的交互侧已不是首要瓶颈，因此只借用 motion prototype 部分。

G2LTraj 在标准 ETH/UCY 协议中并行运行多个粒度并学习选择，但不同场景/基线并非所有 FDE 都改善；因此不建议整体复制递归模块，而建议只吸收“稀疏关键步作为全局骨架”。[G2LTraj 原文](https://www.ijcai.org/proceedings/2024/285)

SICNet 明确指出：仅用终点会忽略中间过程，而完整未来实例又会削弱随机性；其 learned sparse instance 是两者之间的折中。[SICNet 原文](https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Sparse_Instance_Conditioned_Multimodal_Trajectory_Prediction_ICCV_2023_paper.html)

最重要的外部补充是 Stimulus Verification：它不是拿候选中的 best-ADE 分支训练分类器，而是直接用训练集真实轨迹最大化条件似然；推理时按历史社会/场景刺激与候选未来的一致性评分，并用 NMS 保持多样性。[CVPR 2023 原文](https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Stimulus_Verification_Is_a_Universal_and_Effective_Sampler_in_Multi-Modal_CVPR_2023_paper.html) 这与当前“生成器覆盖强、分支分类器弱”的现象高度匹配。

## 六、推荐的模型改进路线

### P0：先重建可信的实验协议

这是所有模型改动的前置条件，不是额外工作。

1. 对每个 ETH/UCY leave-one-scene-out 任务，只从训练场景中划 validation；推荐 scene-aware 或 contiguous-block validation。
2. 所有损失权重、checkpoint、阈值、融合系数和结构选择只看 validation。
3. 目标测试场景在方案冻结后一次性评估。
4. 预先声明主指标：
   - 集合预测：minADE20，辅助报告 same-path FDE 与 minFDE20；
   - 单轨迹：Top-1 ADE，辅助报告 Top-1 FDE、regret、hit@1/3/5。
5. 以后不要再给 24,334 个重叠窗口直接做普通独立样本置信区间。训练随机种子才是随机训练效应的独立实验单位；场景内不确定性用连续时间块 bootstrap 或 scene-level 汇总。

### P1：用独立条件似然 verifier 替换分支分类器主监督

推荐先冻结现有 Q20 生成器，训练一个小型 verifier：

\[
\log p_\phi(Y\mid H,S)
=\sum_{t=1}^{12}\log\sum_{m=1}^{M}
\pi_{t,m}(H,S)\,\mathcal N(Y_t;\mu_{t,m},\Sigma_{t,m}),
\]

其中 `H` 为目标历史，`S` 为邻居历史/当前风险图；`M` 可先用 4 或 8，不必照搬论文的 12。网络只用真实训练轨迹做最大似然，不使用“第几个 Q20 分支是赢家”作为标签。推理时给每条候选计算 log-likelihood，并在 validation 上决定是否与原始 logit 做线性融合。

这一设计的优势是：

- 不受 Mode Query 的分支置换影响；
- 每条真实轨迹都提供监督，不存在只有 WTA 分支回归的问题；
- 直接学习“轨迹与历史/社会刺激是否一致”，而不是学习哪个固定 query 常胜；
- 可独立早停和校准，失败时也不会破坏生成器。

第一版只使用过去社会状态，保持轻量；若在 SDD 上加入场景，只把局部 ROI 场景特征送入 verifier，不立刻改变整个编码器。

不建议第一版再次加入复杂的未来邻居冲突 Transformer。已有 trajectory-set reranker 说明“特征更多、参数更大”不能替代正确监督。可以在条件似然有效后，再加一个小权重的碰撞能量：

\[
s_k=s_k^{\text{likelihood}}-\lambda_{col}E_{col}(\hat Y_k).
\]

### P2：把 20 个自由模式改成数据驱动的分层运动原型

推荐在 agent-centric 坐标中对训练未来轨迹做平移和航向对齐，并适当按历史速度分层。不要直接恢复 P6，而是聚类以下两级信息：

1. **5 个 coarse endpoint/intent 原型**：直行、左右偏转、快慢等由数据自动形成；
2. **每个 endpoint 下 4 个 path-shape 原型**：区分相同终点下不同中间弯曲/避让路径。

这样仍是 20 个候选，但获得 `5×4` 的可解释身份。每个 query 初始化为对应原型编码，模型预测原型残差；可保留 E24 的 full-rank endpoint-zero residual，因此不会重复低秩 P6 的失败。

为了避免 dominant mode 再次吞噬训练信号，分配阶段加入以下二选一机制：

- batch/EMA 逆频率加权；或
- 带容量约束的 Sinkhorn/optimal-transport assignment。

目标不是强迫每个 batch 完全均匀，而是防止长期从不被选择的死分支。

### P3：加入少量中间关键点，不采用完整递归解码

当前 E24 对终点约束明确，但对同一终点下的中间过程缺少显式结构。建议每条候选同时预测 `t=4,8,12` 三个关键点，或者只增加 `t=4,8`，并让完整 E24 路径对这些点保持一致：

\[
L_{key}=\sum_{r\in\{4,8,12\}}w_r
\left\|\hat Y_r-Y_r\right\|_1.
\]

关键点由 mode query 一次性并行生成，完整路径仍由 endpoint-PCA + residual 一次性输出。这样吸收 G2LTraj/SICNet 的“全局骨架”优点，同时避免已有 DCT/低秩曲线与长自回归带来的长期退化。

### P4：把 E24 稳定预测变成 Q20 内部的显式 anchor

目前 E24 被放在模型外用 gate 融合，gate 很难判断何时切换。更合理的方式是保持候选数 K=20：

- 用 1 个候选固定为或初始化为 E24 deterministic anchor；
- 其余 19 个为分层多模态残差候选；
- 统一送入 verifier 打分。

这不会把比较从 K=20 偷换成 K=21，同时为困难样本提供一个稳定默认分支。训练早期可冻结 anchor，后期只允许小残差。

### P5：评分损失只作为辅助，并统一 utility

如果保留原始 scorer，建议改为：

\[
q_k=\operatorname{softmax}\left(-C_k/\tau_{phys}\right),
\qquad
C_k=\operatorname{ADE}_k+\lambda_F\operatorname{FDE}_k,
\]

其中 `τ_phys` 是在 validation 上确定的固定物理尺度，不再除以每个样本的平均 gap。使用 `KL(q||p)` 加少量 hard-negative pairwise ranking，并对长期赢家频率做温和再平衡。主评分仍由独立 verifier 提供；此 scorer 只作为候选生成器内部的辅助信号。

### P6：SDD 阶段再做轻量场景晚融合

ETH/UCY 上先完成无场景因果链。迁移 SDD 前必须重新估计坐标尺度、速度、TTC、DCA、图半径和 RBF 中心，不能沿用“米制 ETH/UCY”的常数。

如果无场景模型在 SDD 的障碍/道路边界处明显失败，再加入：

- 冻结或轻量预训练的局部场景编码器；
- 以候选终点/关键点为 reference，仅采样少量局部特征；
- 只注入 endpoint proposer 或 verifier。

这比恢复完整 U-Net + scene Transformer 更符合当前轻量化目标，也与 CASPFormer 的 query/reference-point 场景采样思想一致。

## 七、建议的最小实验矩阵

### 阶段 A：只修协议和评分器，冻结 Q20

| 实验 | 改动 | 目的 | 继续条件 |
|---|---|---|---|
| A0 | 原始 scorer，正确 validation | 建立无泄漏基线 | 必跑 |
| A1 | fixed-temperature listwise loss | 检查 gap 归一化问题 | Top-1 ADE 至少稳定改善 3/3 seeds |
| A2 | 分支频率平衡 | 检查 mode-prior collapse | active modes、hit@1 同时改善 |
| A3 | ground-truth-only social verifier | 核心方案 | Top-1 ADE/FDE 均优于 A0，regret 显著下降 |
| A4 | verifier + 原始 logits 融合 | 检查互补性 | 融合系数只由 validation 决定 |

所有 A 组都报告：Top-1 ADE/FDE、minADE20、same-path FDE、minFDE20、regret、hit@1/3/5、active modes、概率 NLL/ECE、参数量和端到端延迟。

### 阶段 B：改候选生成

| 实验 | 改动 |
|---|---|
| B0 | 原始自由 Q20 + 最佳 A 组 verifier |
| B1 | 1 个 E24 anchor + 19 个自由 query |
| B2 | 5×4 endpoint-shape prototypes |
| B3 | B2 + 关键点 `t=4,8,12` |
| B4 | B3 + 分支容量/频率平衡 |

阶段 B 的主指标先看 minADE20 和 minFDE20，避免评分器掩盖生成质量；只有 oracle 明确改善后，再看 Top-1。

### 阶段 C：SDD 和场景

1. C0：重新标定尺度的无场景 B 组最佳模型；
2. C1：只给 verifier 加局部场景特征；
3. C2：只给 endpoint/关键点 proposer 加局部场景特征；
4. C3：两者都加。

只有 C1/C2 在多种子和场景分组上稳定有效，才考虑更大的场景编码器。

## 八、统计与报告规范

1. 当前三随机种子可支持“方向一致”判断，但不足以支撑精确显著性结论；报告均值、样本标准差和 3/3 改善比例即可。
2. 24,334 个测试窗口并非 24,334 个独立实验单位。普通逐样本 bootstrap 会因窗口重叠低估不确定性。
3. 建议用场景/连续时间块 bootstrap 估计样本结构不确定性，用随机种子标准差表达训练不确定性，两者分开报告。
4. 多次消融和超参数筛选已经产生选择偏差。最终方案应重新冻结配置，在未参与开发的测试场景/折上做确认性实验。
5. 稀疏图的 ADE 增益小于种子波动，宜表述为“效率/归纳偏置下无损或微幅改善”，不宜作为主要精度结论。
6. 配置中 `PRED_TIME: 4.6`，但 `ceil(4.6×2.5)=12` 实际对应 4.8 秒。应直接改成整数长度或 4.8 秒，避免复现实验时产生协议歧义。

## 九、应停止或暂缓的方向

- 暂停 motif、RWPE、Magnetic LapPE、替代 RWSE 等结构编码堆叠；
- 不再尝试 P6、全秩 P24 或恒速 + DCT 的同类变体，除非模型假设发生实质改变；
- 不再用更大的候选集合 Transformer 直接替代评分损失；
- 不在 ETH/UCY 评分器尚未解决前同时引入大场景分支；
- 不把 test window 数量当独立样本数报告狭窄置信区间；
- 不用 test 上事后选择的模式、阈值或融合权重作为有效结果。

## 十、建议形成的最终方法雏形

一个与现有证据最一致、也有清晰论文叙事的方法可以概括为：

> **Risk-aware Sparse Prototype Trajectory Predictor**：状态增强的风险感知有向稀疏交互编码器，结合数据驱动的终点—路径形状分层原型、E24 端点保持残差和稀疏关键点骨架，并使用独立条件似然 verifier 对候选与历史社会刺激的一致性进行评分。

其创新链条比“再加一种图位置编码”更完整：

1. 风险稀疏图控制交互计算；
2. 分层原型控制多模态候选空间；
3. 稀疏关键点控制中间路径形状；
4. 独立 verifier 解决生成分支先验与 Top-1 选择失效；
5. 场景特征仅在需要时晚融合，保持轻量。

## 十一、最终判断

后续最值得先跑的不是新图模块，而是两个非常具体的实验：

1. **在正确 validation 划分上，冻结 Q20，训练 ground-truth-only social stimulus verifier。** 这是最低风险、最直接验证当前首要归因的实验。
2. **若 verifier 有效，再做 `1 个 E24 anchor + 19 个分层数据原型`，随后增加 2–3 个关键点。** 这才开始解决候选终点和路径形状的第二层瓶颈。

若实验 1 仍不能显著降低 Top-1 regret，则应接受一个重要结论：ETH/UCY 的单一真实未来不足以可靠训练单轨迹概率排序，此时模型应明确定位为 K=20 集合预测器，或引入更丰富的场景/行为监督，而不是继续把 minADE20 的潜力强行转化成不可识别的 Top-1 类别任务。
