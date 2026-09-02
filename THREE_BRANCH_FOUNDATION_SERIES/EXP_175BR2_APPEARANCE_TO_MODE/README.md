# EXP-175B-R2：Observation-Conditioned Appearance-to-Mode

本实验是通过 EXP-175A-R2 后的第一项训练验证。它加载并冻结 EXP-175S0 的
`F16 + H3 + Joint-V2 + original score`，只训练一个基于观测外观的锚点残差支路。

## 因果约束

- 场景原型只来自已经观测到的轨迹坐标，不使用未来轨迹输入。
- 不使用失败的 ADE20K 固定语义标签。
- 不修改 F16、H3、Joint-V2 或 original score 参数。
- 图像对齐要求关闭旋转、反射和速度几何增强。
- 场景修正层零初始化，初始预测必须与 S0 完全一致。
- t4/t8/t12 的额外修正上限分别为 `0.05/0.08/0.15`。
- 不加入场景辅助损失，仍使用原轨迹损失。

## 正式运行

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\THREE_BRANCH_FOUNDATION_SERIES\EXP_175BR2_APPEARANCE_TO_MODE\run_exp175br2_batch.py" --device cuda:0 --seed 1 --epochs 20 --batch-size 64
```

正式训练前可独立执行真实 SDD 小批次前向/反向检查：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\THREE_BRANCH_FOUNDATION_SERIES\EXP_175BR2_APPEARANCE_TO_MODE\run_exp175br2_batch.py" --device cuda:0 --seed 1 --preflight-only
```

输出包括 `results/.../metrics.json`、根目录 `summary.json`、逐视频指标
`per_video_metrics.csv` 和控制台日志。第一轮只运行 seed 1；通过精度门槛后再补 seeds 2/3。
