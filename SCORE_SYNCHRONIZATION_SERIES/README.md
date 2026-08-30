# EXP-161 Frozen Candidate Score Synchronization

本实验只训练一个小型 listwise scorer。F16 编码器、V2/M2 解码器、原始模式嵌入和全部 K=20 候选轨迹均冻结。

训练目标为 `ADE + 0.25 × FDE` 的 soft listwise target；每个官方 fold 只使用其训练集训练和早停，held-out fold 仅用于最终评估。

脚本先分别输出 V2、M2 同步前后的 Top-1 ADE/FDE、regret、Spearman、entropy，并保证 oracle minADE/minFDE 不受评分训练影响。全部五折完成后，依据 `summary.json` 中的透明门槛推荐保留 V2、M2，或者标记为尚不能决策。
