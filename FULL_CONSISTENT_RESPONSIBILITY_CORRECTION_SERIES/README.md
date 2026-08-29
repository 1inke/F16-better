# FCRA1：Full-Consistent Responsibility-Aware Correction

## 目的

FCRA1直接验证CSRS1暴露的两个问题：残差相对Self-Only定义却施加到Full候选，以及mode gate缺少候选分配监督。

## 关键设计

- 每个训练批次在线生成当前尚未经过FCRA修正的Full Social候选。
- 对同一候选计算`ADE + 0.5 × FDE`，温度0.10并保留Top-3 soft responsibility。
- 每个mode预测自己的三锚点残差，target为`GT anchor-current Full anchor`。
- Assignment head使用Top-3 soft cross entropy显式监督。
- Need标签复用CSNA真实Full-vs-Self gain，阈值0.01；不使用interaction potential。
- Assignment和correction只监督actual-needed正样本；negative样本使用no-op损失。
- 推理仅修正Assignment预测的Top-3，其余17个候选保持不变。
- 不使用CV候选或CV参考。

在线target由当前模型生成并detach，因此编码器更新后不会出现离线Full候选过期造成的新target/base mismatch。

## 基线策略

不重复训练B0。直接复用CSRS1中的B0，因为它与本实验具有完全相同的初始化、seed、划分、训练轮数、优化器和关闭增强设置。重新运行会得到确定性重复结果，只增加时间。

## 运行

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\FULL_CONSISTENT_RESPONSIBILITY_CORRECTION_SERIES\run_fcra1_batch.py" --device cuda:0 --seed 1 --splits all --epochs 12
```

将运行General/Challenge两个FCRA1训练，并自动与既有严格匹配B0比较及生成Assignment、正负修正量、Residual、diversity和coverage诊断。

