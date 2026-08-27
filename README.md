# 轻量化多模态行人轨迹预测研究项目

本仓库记录一个围绕行人轨迹预测开展的持续研究项目，核心目标是在保持模型轻量、稀疏交互和运动学先验的前提下，提高多候选轨迹预测的 minADE/minFDE、候选覆盖与排序能力。

仓库不仅包含最终方法代码，还保留了实验假设、批量运行脚本、逐轮结果、消融数据、失败方向和研究分析，便于完整理解模型是如何演化的。

## 1. 当前方法概览

当前模型可以拆成四个主要部分：

1. **历史运动编码**：从观测轨迹提取速度、方向、曲率和多尺度时间信息；
2. **稀疏社会交互**：只在运动学图选出的局部邻居之间进行时间—社会交叉注意力；
3. **多模态候选生成**：使用模式查询、终点/中间锚点和运动模板生成 20 条候选轨迹；
4. **候选评价与细化**：研究候选排序、锚点修正、模板细化和模式分组，但现有实验表明排序仍是主要短板。

模型不依赖恒速（CV）外推作为预测参考，运动学信息作为先验和关系约束进入网络。

详细结构见：[当前模型方法详解.md](./当前模型方法详解.md)。

## 2. 建议阅读顺序

如果希望快速理解整个项目，建议按以下顺序阅读：

1. [实验阶段总结_供其他模型查阅.md](./实验阶段总结_供其他模型查阅.md)
2. [ASTRA_Q20_模型问题与改进分析报告.md](./ASTRA_Q20_模型问题与改进分析报告.md)
3. [当前模型方法详解.md](./当前模型方法详解.md)
4. [全实验研究总结与后续改进参照_2026-08-23.md](./全实验研究总结与后续改进参照_2026-08-23.md)
5. [推荐方法_实验决策与下一步路线.md](./推荐方法_实验决策与下一步路线.md)
6. 各实验系列目录中的 `README.md`、`RESULT_ANALYSIS.md`、`all_runs.csv` 和 `multiseed_summary.json`

场景输入方向的专项调研位于：

- [SDD 场景输入文献评估](./SDD_SCENE_INPUT_LITERATURE_EVALUATION_2026-08-28.md)
- [近四年场景信息提取与轨迹融合深度调研](./SCENE_METHOD_DEEP_SURVEY_2026/近四年场景信息提取与轨迹融合深度调研_2026-08-28.md)
- [跨领域视觉场景提取与多模态融合调研](./SCENE_METHOD_DEEP_SURVEY_2026/跨领域视觉场景提取与多模态融合调研_自动驾驶目标检测视频理解_2026-08-28.md)

## 3. 实验系列目录

### 编码器、交互与注意力

- `FRONT_ENCODER_SERIES`：时间编码、社会交互和前端编码器基础消融；
- `ATTENTION_COMPUTE_SERIES`：注意力计算方式与效率实验；
- `ATTENTION_MECHANISM_SERIES_2`：第二阶段注意力机制实验；
- `PAIR_AWARE_ATTENTION_SERIES`：成对关系感知注意力；
- `ORACLE_INTERACTION_SERIES`：交互模块的 oracle 与上限诊断；
- `THREE_PART_ENCODER_SERIES`：历史、社会交互和状态分支的分块验证；
- `NON_DECODER_OPTIMIZATION_SERIES`：解码器之外的优化实验。

### 候选、锚点、模板与解码

- `HIERARCHICAL_DECODER_SERIES`：终点与三锚点两阶段解码；
- `ANCHOR_COUNT_ABLATION_SERIES`：2、4、6 等锚点数量消融；
- `REFINER_MECHANISM_SERIES`：轨迹细化器及训练/推理耦合；
- `CANDIDATE_GENERATION_SERIES`：候选生成机制；
- `CONTROLLED_ANCHOR_CORRECTION_SERIES`：受控锚点修正；
- `ANCHOR_CORRECTION_VALIDATION_SERIES`：锚点修正的泛化验证；
- `INTENT_ALLOCATION_SERIES`：候选意图分配；
- `MODE_QUERY_DISTILLATION_SERIES`：模式查询监督与蒸馏；
- `MODE_TEMPLATE_GROUP_SERIES`：锚点对齐运动模板与软模式分组。

### 泛化、完整数据与 SDD

- `FRONT_DOMAIN_ADAPTATION_SERIES`：跨划分和域适配诊断；
- `FRONT_SCALE_GENERALIZATION_SERIES`：尺度变化与归一化；
- `FRONT_TARGETED_GENERALIZATION_SERIES`：针对速度、曲率等困难样本的泛化；
- `ENCODER_TRAINING_PRECISION_SERIES`：训练精度和稳定性；
- `ETH_UCY_FULL_CROSSVAL_SERIES`：ETH/UCY 完整交叉验证；
- `SDD_LOW_COST_SERIES`：SDD 低成本可行性实验；
- `SDD_STANDARD_POTENTIAL_SERIES`：SDD 标准化评估和潜力验证。

### 调研与决策

- `ATTENTION_TRANSFORMER_LONG_RESEARCH_2026`：跨领域注意力与 Transformer 长调研；
- `SCENE_METHOD_DEEP_SURVEY_2026`：图像/视频场景提取与融合调研；
- `qoder_work`：外部建议和项目分析材料。

## 4. 最新 EXP-146 结论

`MODE_TEMPLATE_GROUP_SERIES` 验证了两个假设：锚点之间的线性插值是否限制完整轨迹，以及 20 个模式是否能通过软分组改善排序。

三随机种子结果表明：

- 锚点对齐模板使 General/Challenge 的 minADE 稳定下降约 0.8%—1.1%；
- minFDE 改善只有约 0.1%—0.35%，说明线性插值只是次要瓶颈；
- 五组模式发生明显选择塌缩，实际只使用一到两个组；
- 分组没有提高精确命中最佳候选的概率；
- 当前主要瓶颈仍是终点/分叉覆盖和组内完整路径排序。

详见：[MODE_TEMPLATE_GROUP_SERIES/RESULT_ANALYSIS.md](./MODE_TEMPLATE_GROUP_SERIES/RESULT_ANALYSIS.md)。

## 5. 结果文件约定

多数实验目录包含：

- `README.md`：假设、方法改动和运行命令；
- `RESULT_ANALYSIS.md`：实验结论和后续判断；
- `all_runs.csv`：逐随机种子、逐划分结果；
- `multiseed_summary.json`：均值、标准差和原始指标；
- `results/**/metrics.json`：单次训练的完整评价结果；
- `results/**/history.csv`：训练历史。

常用指标：

- `min_ade_20` / `min_fde_20`：20 个候选中的 oracle ADE/FDE；
- `same_path_fde`：取得最佳 ADE 的同一候选对应的 FDE；
- `top1_ade` / `top1_fde`：模型实际选择候选的指标；
- `hit_best_ade`：Top-1 是否命中 oracle ADE 候选；
- `regret`：实际选择与 oracle 候选之间的差距。

不同数据协议和坐标尺度下的绝对数值不可直接横向比较，优先查看同目录、同划分、同随机种子的配对消融。

## 6. 数据与模型权重

本仓库**不包含任何 ETH/UCY、SDD 原始数据、处理后数组、数据划分索引或视频/图像数据**，也不包含训练 checkpoint。

被排除的主要文件类型包括：

- `*.npy`、`*.npz`、`*.pkl`、`*.pickle`；
- `*.pth`、`*.pt`、`*.ckpt`；
- 数据集和 `processed_data*` 目录；
- 控制台日志、缓存和 smoke-test 临时输出。

仓库中的指标 JSON/CSV 是实验结果数据，不包含训练样本。

## 7. 运行环境与可移植性

当前实验脚本源自 Windows 本地研究环境，部分脚本仍包含本机绝对路径，例如外部 ASTRA 工程、Python 环境和数据集位置。复现实验前需要修改各批处理脚本顶部的：

```python
PROJECT = Path(...)
PYTHON = Path(...)
DATA = Path(...)
```

主要外部工程在原研究环境中位于 `ASTRA_ACCURACY_Q20`，本仓库保存的是围绕该骨干开发的模型插件、训练驱动、实验记录和分析，而不是复制原始数据或第三方工程。

典型批处理入口形如：

```powershell
python "<SERIES>/run_*_batch.py" --device cuda:0 --epochs 30 --batch-size 64 --seeds 1 2 3
```

运行前应先阅读对应实验目录的 `README.md`，确认来源 checkpoint、数据协议、划分和参数设置。

## 8. 当前研究判断

经过编码器、注意力、锚点、候选、细化和分组等多轮验证后，小范围结构微调通常只能带来约 0%—2% 的改善。后续更有潜力的方向是把结构化场景直接引入模式、终点和锚点生成：

```text
语义/可行走/SDF raster
    + 场景边界、通道、分叉和入口 vector tokens
    + 动态实例 tokens
    -> mode/anchor/candidate query 稀疏读取场景
    -> 场景条件候选生成、细化与评分
```

这一路线保留已有运动学和稀疏社会交互，同时补充纯轨迹输入无法判断的路径可行性与场景分叉信息。

