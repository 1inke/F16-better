# EXP-175C：Soft Appearance Feasibility Auxiliary Loss

该实验从与 EXP-175B-R2 相同的 S0 seed1 检查点和零初始化场景支路开始，唯一变化是训练时加入固定权重 `0.02` 的软路径外观可行性损失。

- 只约束按真实轨迹误差选出的近 Oracle Top-5 候选邻域。
- 候选权重停止梯度，不把未来轨迹作为模型输入。
- 不做硬删除或硬可通行约束。
- F16、H3、Joint-V2和original score继续冻结。
- 不进行测试集权重搜索。

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\THREE_BRANCH_FOUNDATION_SERIES\EXP_175C_SOFT_APPEARANCE_FEASIBILITY\run_exp175c_batch.py" --device cuda:0 --seed 1 --epochs 20 --batch-size 64 --scene-loss-weight 0.02
```

如果相对S0的minADE和minFDE均未达到1%改善，则停止该外观场景分支，不补跑更多种子。
