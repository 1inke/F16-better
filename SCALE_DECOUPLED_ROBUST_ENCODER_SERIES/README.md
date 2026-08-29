# SDRE1：Scale-Decoupled Robust Encoder

SDRE1保持F16 sparse temporal-social encoder、K20及模式+三锚点解码器不变。它新增两条零初始化残差：

- 基于每位行人速度尺度归一化的motion-shape adapter；
- 基于局部邻居距离与双方速度归一化的relation branch。

绝对speed、acceleration、neighbour count及local distance不被删除；它们以6维condition控制两条normalized residual的强度。因两个残差的末层均为零初始化，训练开始时SDRE1与F16计算等价。

F16原有social/output LayerNorm保持不变；第一轮不额外插入强制LayerNorm，以避免把“表示重标定”混入尺度解耦实验。

## 正式运行

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\SCALE_DECOUPLED_ROBUST_ENCODER_SERIES\run_sdre1_batch.py" --device cuda:0 --seed 1 --epochs 12 --val-freq 3 --batch-size 64
```

运行General与Challenge上的B0/SDRE1共4次训练。预计约2～5小时，取决于GPU和既有12-epoch实验耗时。

若只运行SDRE1：`--variants sdre1`。若只运行Challenge：`--splits challenge`。

运行前可验证模型接口：

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\SCALE_DECOUPLED_ROBUST_ENCODER_SERIES\smoke_check.py"
```

## 预注册判据

- Challenge minADE20或minFDE20改善≥0.5%，General不能明显反向退化；
- 后续诊断中Challenge encoder domain AUROC应从0.9226降低，individual/social norm SMD应从约1.51/1.48明显下降；
- 不训练tail gate，也不修改decoder。
