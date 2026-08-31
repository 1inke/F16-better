# Horizon-Resolved Motion Conditioning Series

本轮只训练小型运动条件adapter，F16、稀疏社交交互、H3主体、原V2 refiner和score全部冻结。

- `p1`：H3 source anchors之后、V2 refiner之前的逐mode三锚点二维残差。
- `c1`：H3 proposal不变，在不扩大原V2上限的前提下调整现有correction。

两者均使用连续、无量纲、仅观测历史可得的运动特征；不使用hard high-acc gate、scene ID、CV reference或全局轨迹缩放。

```powershell
python "C:\Users\linke\Documents\New project\HORIZON_RESOLVED_MOTION_CONDITIONING_SERIES\run_hrmc_batch.py" --device cuda:0 --folds all --variants all --seed 1 --epochs 20 --batch-size 64
```

完成训练后，驱动脚本自动执行官方outer-fold以及high-acc/non-high-acc评估。
