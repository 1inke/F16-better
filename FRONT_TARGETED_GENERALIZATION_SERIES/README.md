# EXP-138：针对性泛化实验

固定 F16、Top-K=4 稀疏图、K20 解码器和已有训练设置，验证四个单变量：

1. `residual_shape_scale`：保留 Baseline robust state，增加零初始化 Shape–Scale 小残差；
2. `future_difficulty_loss`：未来 GT 仅用于训练期逐样本损失权重，不作为模型输入；
3. `wide_speed_augmentation`：仅把速度增强范围从 0.9–1.1 扩至 0.75–1.25；
4. `curvature_augmentation`：在原增强上增加最大 0.025 rad/step 的共享平滑曲率扰动。

默认运行两个冻结开发划分与 seed 1/2/3，共 24 次新训练。已有 Baseline 直接复用，不访问官方测试折。

## 运行命令

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\FRONT_TARGETED_GENERALIZATION_SERIES\run_targeted_generalization_batch.py" --device cuda:0 --epochs 30 --batch-size 64
```

支持断点续跑；不要加入 `--force`，除非需要覆盖全部结果。

## 前置验证

- 所有脚本语法检查和批量命令 dry-run 通过；
- EXP-138A 初始残差严格为零，与 Baseline 状态输出最大差为 0；
- 全 1 样本权重时，新损失与旧损失数值完全一致；
- 未来困难度权重已接入真实模型预测并完成反向传播；
- 曲率增强保持每个时刻的智能体间欧氏距离，测试最大误差为 4.77×10⁻⁷；
- EXP-138A 参数量为 290,859，其余三个实验推理模型均保持 284,642 参数。

## 时间估计

上一轮 18 次训练实测约 4 小时。本轮共 24 次，预计约 5 小时 20 分至 6 小时；建议预留 6.5 小时。未来困难度加权只增加训练期损失计算，不改变推理模型。
