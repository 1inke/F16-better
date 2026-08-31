# Individual Motion Representation Audit

这是 EXP-169B 得到 B1（Full Social beneficial）后的预注册下一步。

实验不训练、不修改 checkpoint，也不改变 decoder。它复用 EXP-164 official V2 seed 1，审计 F16 四个阶段的目标行人表征：

1. `robust_state`
2. temporal feature
3. post-social agent feature
4. final encoder output

输出内容包括高加速度分桶误差、表征范数/时间变化/饱和率、表征与加速度和未来误差的相关性，以及用 source-inner 特征拟合、在 outer fold 测试的线性可解码性。线性 probe 只是只读诊断，不更新轨迹模型。

正式运行：

```powershell
python "C:\Users\linke\Documents\New project\INDIVIDUAL_MOTION_REPRESENTATION_AUDIT\run_motion_representation_audit.py" --device cuda:0 --folds all --seed 1 --batch-size 64
```

正式输出位于 `MOTION_REPRESENTATION_AUDIT/`。
