# EXP-117 MATKS

Motion-Adaptive Temporal Key Smoothing。保持 F16 的完整 8 帧、运动学稀疏图、relation bias/value 和 Kinematic K20，只在 temporal-social attention 的 K/V 投影前做局部三点平滑。

左右邻帧混合系数由当前 temporal feature 预测，上限各为 0.20；二阶运动变化越大，平滑越弱，最后一帧的平滑上限额外降为四分之一，以保护最新速度和转向。首筛为 seed=1、scene 1/3、30 epochs。

