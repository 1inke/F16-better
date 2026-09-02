# EXP-176D：Scene-Compatible Score

这是零训练、冻结候选集的SDD评分审计。固定使用EXP-175A-R2在validation选择的
`appearance_similarity`，不重新选择场景特征，也不搜索权重。

```text
corrected_logit
= original_logit
- 0.10 * within-sample score_std * (1 - appearance_similarity)
```

脚本先运行validation。只有综合Top-1代价和mean regret改善、P95 regret不恶化超过2%、
场景违背率下降并实际改变了候选选择时，才会读取test一次。

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\THREE_BRANCH_FOUNDATION_SERIES\EXP_176D_SCENE_COMPATIBLE_SCORE\run_exp176d_scene_score.py" --device cuda:0 --seed 1 --batch-size 64
```

本实验不改变候选轨迹，因此所有oracle minADE/minFDE必须保持不变。
