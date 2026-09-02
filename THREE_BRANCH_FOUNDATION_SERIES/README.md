# EXP-174A / EXP-175S0 / EXP-176A 第一批三分支实验

本目录落实 `THREE_BRANCH_EXPERIMENT_GUIDE_SCENE_EFFICIENCY_SCORING_2026-09-01.md` 的第一阶段，不提前执行需要依赖本轮结论的 EXP-174B、EXP-175A 或 EXP-176B。

## 实验内容

- **EXP-174A**：对 EXP-173 的 F16 Joint-V2 最终模型及保留的 E2 模型做统一 FP32 效率基线。报告参数量、Profiler 可计 FLOPs、均值/中位/P95 延迟、吞吐量、峰值显存；覆盖 batch=1/16/64 和 N=4/8/16/32/64。
- **EXP-175S0**：使用标准 SDD 31/12/17 视频划分，训练 F16 + H3 + Joint-V2 三种子无场景基线。输出总体、little/non-little 与逐视频的 minADE/minFDE、same-path、top1（像素）。视频 ID 只用于分组统计，不输入模型。
- **EXP-176A**：不训练模型，审计 EXP-173 F16 Joint-V2 五折三种子的原生分数。质量定义为 `ADE + 0.25 FDE`，输出相关性、命中率、top-k 恢复、regret、熵和 mode usage。

## 运行与恢复

单 GPU 使用 `run_three_branch_batch.py` 顺序执行，避免同时占用显卡。SDD 阶段会跳过已经存在 `metrics.json` 的种子/阶段，因此中断后直接重启同一命令即可续跑。正式结果分别写入三个 `EXP_*` 子目录；模型权重位于 EXP-175S0 的 `results` 下。

注意：EXP-174A 的 FLOPs 是 PyTorch Profiler 已支持算子的计数，不支持公式的算子不会被纳入，因此结果会同时保留稀疏边数与名义注意力 score 元素数，避免把 profiler 值误解为理论完整 FLOPs。

## 第二阶段：EXP-176B

`run_exp176b_cross_scene_feature_audit.py` 对 EXP-173 F16 Joint-V2 的冻结候选做零训练特征审计。它先用其他场景对原生 score 与联合误差的关系做交叉拟合控制，再逐样本计算 motion、social、V2 correction 与 proposal 特征对 residual quality 的 Spearman 关系。只有至少两个非原生分数特征达到 `|macro Spearman| >= 0.15` 且五折中至少四折同号，才允许进入 EXP-176C。社会特征中的恒速外推只用于估算与其他智能体的未来距离，不参与候选生成，也不是解码参考。

## 第三阶段：EXP-176C

`run_exp176c_regret_safe_scorer.py` 只使用 EXP-176B 通过的 `acceleration_mismatch` 与 `mean_social_clearance`。候选和原始 score 冻结；小型 `2→32→1` MLP 仅产生 `β=0.35` 的有界残差。每个 outer fold 使用其余四个 source scenes 做四轮 scene-held-out CV 选择 epoch，再用全部 source scenes 拟合并只在最后访问 target。训练使用 decisive pairwise loss（margin 为 source-train quality difference P50）和 `0.1 KL` 原始评分锚定。

## EXP-174B：无重训练运行时优化

`run_exp174b_no_retrain_runtime.py` 在最终 EXP-173 F16 Joint-V2 上比较 FP32 eager、FP16 autocast、BF16 autocast 和可用时的 `torch.compile`。每个正式延迟结果包含100次warmup和1000次逐次CUDA同步测量，并在ETH测试集核对minADE/minFDE与候选坐标数值误差。F16主社会注意力包含pair-specific relation bias/value，无法由融合SDPA无损替换；图构建、mask与三锚点插值已经批量向量化，因此B3/B4只输出结构审计，不伪造无效变体。
